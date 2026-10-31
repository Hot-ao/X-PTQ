"""
PD-Quant(Liu et al., CVPR 2023) 기준선과 head 캡처 도우미.

head 캡처 도우미(promptcal.py와 diag_w4_sensitivity.py가 쓴다):
  _find_head(model)  : detection head(model.model[-1])를 찾는다.
  _CV4Capture        : head.cv4(레벨별 contrastive head)의 출력(유사도 맵)과, capture_input=True면
                       입력(region feature)도 grad를 유지한 채 캡처한다.

PD-Quant (10-28, 공식 코드 github.com/hustvl/PD-Quant e12038e의 quant/block_recon.py, data_utils.py를 따름):
QDrop의 block 재구성 위에 두 가지를 더한다. brecq.optimize_brecq(pd=..., dc_stats=...)가 이 파일의 도우미를 쓴다.
  (1) Prediction Difference(PD) 손실: 재구성 중인 block의 출력(양자화 입력)을 나머지 FP 네트워크에 흘려 얻은
      최종 예측과 FP 예측의 KL(온도 T)을 더한다(/lamb_r). 공식: KLDivLoss(batchmean)(log_softmax(q/T), softmax(fp/T)).
  (2) Distribution Correction(DC): block의 FP 입력을, block 안 BN의 running 통계(평균·표준편차)에 맞도록
      500 iter 보정한다(입력에서 너무 멀어지지 않게 lp 제약/lamb_c). 보정된 입력의 FP block 출력이 재구성 목표가
      되고, 보정된 입력은 QDrop 입력 drop의 FP 쪽 값으로 쓰인다.
검출기에 옮기며 정한 것:
  - "예측" = head의 원시 출력(박스 DFL 분포 4×reg_max + 클래스 logit)을 anchor 단위로 모은 것(decode 전, sigmoid 전).
    클래스는 어휘(calibration = COCO 80) 위의 softmax, 박스는 변마다 reg_max 구간 위의 softmax로 KL을 잰다.
    공식은 분류(ImageNet 1000클래스)만 다뤄 박스 항이 없다 -- 검출의 예측에는 박스도 포함되므로 같은 가중치로 더했다.
  - 나머지 네트워크: FP 모델 전체를 다시 돌리면서 재구성 대상의 FP 짝 모듈 출력을 양자화 경로 출력으로 바꿔 끼운다
    (forward hook). 공식(ResNet류)은 뒤쪽 블록을 순서대로 통과시키는데, YOLO는 앞 층 출력을 뒤에서 다시 쓰는
    skip 연결이 있어 이렇게 했다. skip으로 들어오는 앞 층 값은 FP다.
  - BN: ultralytics는 양자화 전에 BN을 conv에 접는다(fuse). 접힌 conv 출력은 BN 출력과 같으므로, BN 입력이
    (running_mean, sqrt(running_var+eps))를 따르게 하는 공식 목표는 접힌 출력이 (beta, |gamma|·sqrt(var/(var+eps)))를
    따르게 하는 것과 같다. 접기 전 모델에서 이 값을 모아 둔다(collect_bn_targets).
"""

from __future__ import annotations
import os
import torch
import torch.nn.functional as F


def _find_head(model):
    core = getattr(model, "model", model)
    return core[-1] if hasattr(core, "__getitem__") else core.model[-1]


class _CV4Capture:
    """head.cv4의 레벨별 출력을 grad 유지한 채 캡처.

    capture_input=True(09-25)면 cv4의 **입력**(ContrastiveHead.forward(x, w)의 x =
    region feature)도 같이 모은다. ContrastiveHead는 sim_j = x̂ · ŵ_j 이므로,
    x̂(region embedding의 단위 방향)를 보존하면 프롬프트를 하나도 참조하지 않고
    seen/unseen 모든 vocabulary의 유사도가 함께 보존된다 -- vocabulary-agnostic
    정규화의 자연스러운 대상이다(V3 §6 방향 2)."""
    def __init__(self, head, capture_input=False):
        self.head = head
        self.buf = {}
        self.inbuf = {}
        self.capture_input = capture_input
        self.handles = []
        for i, sub in enumerate(head.cv4):
            def make(idx):
                def hook(_m, _inp, out):
                    self.buf[idx] = out          # grad 유지 (detach 안 함)
                    if self.capture_input:
                        self.inbuf[idx] = _inp[0]
                return hook
            self.handles.append(sub.register_forward_hook(make(i)))

    def assemble_input(self):
        """레벨별 region feature를 [anchors, C]로 조립(정규화 전)."""
        parts = []
        for i in sorted(self.inbuf):
            t = self.inbuf[i]
            B, C, H, W = t.shape
            parts.append(t.reshape(B, C, H * W))
        return torch.cat(parts, dim=2)[0].transpose(0, 1)

    def clear(self): self.buf = {}; self.inbuf = {}
    def close(self):
        for h in self.handles: h.remove()
        self.handles = []


# ---------------------------------------------------------------- PD-Quant (10-28)
PD_T = 4.0          # 공식 main_imagenet.py --T 기본값(run_script.py도 ResNet/RegNet은 4.0)
PD_LAMB_R = float(__import__("os").environ.get("PDQ_LAMB_R", "0.1"))   # 공식 --lamb_r 기본값 0.1 (pd 손실 / lamb_r). PDQ_LAMB_R로 진단 변경
DC_LAMB_C = 0.02    # 공식 --lamb_c (ResNet 설정)
DC_LR = 1e-3        # 공식 --bn_lr
DC_ITERS = 500      # 공식 GetDcFpLayerInpOut의 iters
DC_BATCH = 32       # 공식 batch_size(DC는 배치 단위로 보정)
_EPS = 1e-6
# 10-28 진단용 변형(환경변수, 기본값이면 위 공식 설정 그대로):
#   PDQ_PD_REDUCE=sum|mean  PD KL을 anchor에 대해 합(공식 batchmean 그대로) 또는 평균
#   PDQ_LAMB_R=<값>          PD 손실 가중치(나누는 값). 공식 0.1
#   PDQ_DC_SKIP_FIRST=1      첫 블록(입력=이미지)에는 DC를 걸지 않는다
#   PDQ_NO_PD=1 / PDQ_NO_DC=1  PD 손실 / DC를 끈다(run_comparison의 pdquant 분기에서 읽음)
PD_REDUCE = os.environ.get("PDQ_PD_REDUCE", "sum")
PD_OFF = os.environ.get("PDQ_NO_PD", "0") == "1"
DC_OFF = os.environ.get("PDQ_NO_DC", "0") == "1"
DC_SKIP_FIRST = os.environ.get("PDQ_DC_SKIP_FIRST", "0") == "1"   # 입력이 이미지인 첫 블록(model.0)은 DC 제외


def collect_bn_targets(unfused_det_model):
    """접기 전 DetectionModel에서 Conv(conv+bn) 모듈마다, 접힌 conv 출력이 따라야 할 채널별 (평균, 표준편차)를 모은다.
    반환: {"model.2.cv1.conv": (mean[C], std[C])} -- 이름은 접은 뒤 모델에서도 같은 nn.Conv2d를 가리킨다."""
    out = {}
    for name, m in unfused_det_model.named_modules():
        conv, bn = getattr(m, "conv", None), getattr(m, "bn", None)
        if isinstance(conv, torch.nn.Conv2d) and isinstance(bn, torch.nn.BatchNorm2d):
            var = bn.running_var.detach().float()
            g = bn.weight.detach().float().abs() * torch.sqrt(var / (var + bn.eps))
            out[f"{name}.conv"] = (bn.bias.detach().float().clone(), g.clone())
    return out


def _lp(pred, tgt, p=2.0):
    """공식 lp_loss(reduction='none'): 채널축 sum, 나머지 mean."""
    return (pred - tgt).abs().pow(p).sum(1).mean()


def dc_correct(fm, fp_inputs, bn_map, device, lamb=DC_LAMB_C, lr=DC_LR, iters=DC_ITERS, batch=DC_BATCH):
    """공식 GetDcFpLayerInpOut: block FP 입력(첫 텐서 인자)을 BN 통계에 맞게 보정한다.
    fp_inputs: 이미지별 입력 튜플(_to_cpu 형식). bn_map: {nn.Conv2d 모듈: (mean, std)}.
    fm 안에 BN 통계가 있는 conv가 없으면 공식에서도 제약 항만 남아 입력이 그대로이므로 None을 돌려준다.
    반환: (보정된 입력 튜플 목록, 보정된 입력에 대한 FP 출력 목록) -- 둘 다 _to_cpu 형식."""
    from .brecq import _collate, _to_dev, _to_cpu
    convs = [(m, bn_map[m]) for m in fm.modules() if m in bn_map]
    if not convs:
        return None
    caps = {}
    hs = [m.register_forward_hook(lambda mod, i, o: caps.__setitem__(mod, o)) for m, _ in convs]
    stats = [(m, mu.to(device), sd.to(device)) for m, (mu, sd) in convs]
    new_in, new_out = [], []
    try:
        for b0 in range(0, len(fp_inputs), batch):
            args = list(_to_dev(_collate(fp_inputs[b0:b0 + batch]), device))
            x0 = args[0].detach()
            para = x0.clone().requires_grad_(True)
            opt = torch.optim.Adam([para], lr=lr)
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, min_lr=1e-5, patience=100)
            for _ in range(iters):
                opt.zero_grad()
                caps.clear()
                fm(para, *args[1:])
                loss = 0.0
                for m, mu, sd in stats:
                    o = caps[m]
                    o = o.reshape(o.size(0), o.size(1), -1)
                    t_mu = o.mean(2)
                    t_sd = torch.sqrt(o.var(2) + _EPS)
                    loss = loss + (mu - t_mu).norm() ** 2 / t_mu.size(0) + (sd - t_sd).norm() ** 2 / t_sd.size(0)
                loss = loss + _lp(para, x0) / lamb
                loss.backward()
                opt.step()
                sch.step(float(loss.detach()))
            with torch.no_grad():
                out = fm(para.detach(), *args[1:])
            for k in range(para.size(0)):
                new_in.append(_to_cpu(tuple([para[k:k + 1].detach()] + [a[k:k + 1] if torch.is_tensor(a) else a
                                                                          for a in args[1:]])))
                new_out.append(out[k:k + 1].detach().half().cpu())
    finally:
        for h in hs:
            h.remove()
    return new_in, new_out


class PDContext:
    """PD 손실 계산기. FP 모델 전체(fp_full = DetectionModel)와 calibration 이미지별 FP 예측을 들고 있다."""

    def __init__(self, fp_full, calib_list, device, T=PD_T, lamb_r=PD_LAMB_R):
        self.fp, self.calib, self.device, self.T, self.lamb_r = fp_full, calib_list, device, T, lamb_r
        for p in fp_full.parameters():
            p.requires_grad_(False)          # 뒤쪽 FP 네트워크는 grad 통로일 뿐 학습 대상이 아니다
        self.head = _find_head(fp_full)
        self.nbox = 4 * int(getattr(self.head, "reg_max", 16))
        self._cap = None
        self._hh = self.head.register_forward_hook(self._head_hook)
        with torch.no_grad():
            self.fp_pred = [self._run(t).detach().half().cpu() for t in calib_list]

    def _head_hook(self, m, i, o):
        self._cap = o[1] if isinstance(o, tuple) else o

    def _run(self, t, replace=None):
        """fp_full을 돌려 head 원시 출력을 [B, A, no]로 모은다. replace=(module, tensor)면 그 모듈의 출력을 바꿔 끼운다."""
        h = None
        if replace is not None:
            mod, val = replace
            h = mod.register_forward_hook(lambda m, i, o: val)
        try:
            self._cap = None
            self.fp(t.to(self.device))
        finally:
            if h is not None:
                h.remove()
        xs = self._cap
        if isinstance(xs, dict):         # 현재 ultralytics: {"boxes": [B, 4*reg_max, A], "scores": [B, nc, A], "feats": ...}
            return torch.cat([xs["boxes"], xs["scores"]], dim=1).transpose(1, 2)
        b = xs[0].size(0)                # 이전 ultralytics: 레벨별 [B, 4*reg_max + nc, H, W] 목록
        return torch.cat([x.reshape(b, x.size(1), -1) for x in xs], dim=2).transpose(1, 2)

    def _kl(self, q, f):
        """공식 KLDivLoss(reduction='batchmean')를 그대로 옮김: 분포 축(마지막)으로 softmax, 배치 축을 뺀 나머지
        (anchor 등) 전부 합한 뒤 배치 크기로 나눈다. 즉 anchor에 대해서는 평균이 아니라 합이다."""
        lq = F.log_softmax(q / self.T, dim=-1)
        pf = F.softmax(f / self.T, dim=-1)
        kl = pf * (torch.log(pf + 1e-12) - lq)
        if PD_REDUCE == "mean":          # 진단 변형: anchor 평균(분포 축은 합)
            return kl.sum(-1).reshape(kl.size(0), -1).mean(1).mean()
        return kl.reshape(kl.size(0), -1).sum(1).mean()

    def loss(self, fm, out_q, idx):
        """fm(재구성 대상의 FP 짝 모듈)의 출력을 out_q로 바꿔 FP 네트워크를 통과시킨 예측과, 같은 이미지의 FP 예측의 KL."""
        q = self._run(torch.cat([self.calib[j] for j in idx], 0), replace=(fm, out_q))
        f = torch.cat([self.fp_pred[j] for j in idx], 0).to(self.device).float()
        nb = self.nbox
        kl_cls = self._kl(q[..., nb:], f[..., nb:])
        rm = nb // 4
        kl_box = self._kl(q[..., :nb].reshape(*q.shape[:2], 4, rm), f[..., :nb].reshape(*f.shape[:2], 4, rm))
        return (kl_cls + kl_box) / self.lamb_r

    def close(self):
        self._hh.remove()
