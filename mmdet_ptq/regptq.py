"""11-16 Reg-PTQ(Ding et al., CVPR 2024, "Regression-specialized Post-training Quantization for Fully Quantized Object
Detector") 재현: mmdet RetinaNet R50 / Faster R-CNN R50 위의 QDrop과 Reg-PTQ. .venv_mmdet 전용.
공식 코드가 없어 논문 본문(4-5절, 알고리즘 1)만 보고 옮겼다. 양자화 모듈은 메인 실험 구현(quant/)을 그대로 쓴다.

논문 5.1절 프로토콜(full quantization):
  - QDrop 파이프라인을 따른다: AdaRound weight 반올림 + activation step(LSQ)을 단위마다 함께 재구성, QDrop drop 0.5.
  - backbone은 block 단위, 그 밖(stem, FPN, head)은 layer 단위. 단위마다 2k iters.
  - 모든 중간층을 WwAa로(head 포함). 첫 층은 8bit, 마지막 예측층(RetinaNet retina_cls/reg, RPN rpn_cls/reg,
    RoI fc_cls/fc_reg)은 FP. 덧셈(residual)은 FP. BN은 conv에 접는다.
  - 순서(알고리즘 1): 단위를 앞에서부터 하나씩 연다(open_quant). 앞 단위는 재구성이 끝난 양자화 상태, 뒤 단위는 아직 FP.
Reg-PTQ = QDrop + FGIC(+ LLAQ):
  - FGIC 손실(식 3-5, 8): L = L_H + L_G.
      L_H = sum (Ô-O)^2 g^2 (식 3, 대각 Fisher). g = dL_G/dÔ, 단위를 열 때 calibration 이미지마다 한 번 잰다(알고리즘 1 4-7행).
            공식 BRECQ 코드(같은 연구 그룹)의 Fisher 구현을 따라 g <- |g| + 1, 손실은 채널 합의 평균.
      L_G = (1/n) sum_i [ CE(y_i, ŷ_i) + λ L_p(b_i·I, b̂_i·I) ] (식 8). y, b = FP 검출기 출력, ŷ, b̂ = 양자화 검출기 출력.
            I = I_HC·I_HI: FP 신뢰도 >= θ_C(2e-4) 그리고 IoU(b, b̂) >= θ_I(0.1)인 박스만(논문 기본값, 표 2).
  - LLAQ(4.3절, head weight의 로그-아핀 양자화): --method regptq_llaq. 보충자료에도 세부 식이 없어 본문 식 9-10으로 정했다(LLAQ 클래스).
논문에 없는 값과 정한 것:
  - calibration: COCO train2017 앞 256장(test 파이프라인, 증강 없음). 논문은 "batch 256"만 적었다.
    이미지 크기가 서로 달라 batch 1로 2k iters(논문과 다름).
  - 재구성 하이퍼파라미터: 메인 실험의 QDrop과 같다(alpha Adam 1e-3, 반올림 정규화 0.01, warmup 0.2, β 20->2,
    LSQ step Adam 4e-5 + cosine, QDrop 손실 lp p=2).
  - L_G의 박스: 회귀 출력(anchor/proposal 대비 delta)의 L1(p=1), λ = 1. IoU 필터는 delta를 박스로 풀어서 계산.
    CE: RetinaNet과 RPN은 sigmoid라 FP 확률을 목표로 한 BCE(클래스 합), RoI head는 FP softmax를 목표로 한 CE.
    식 8 그대로 CE는 모든 박스, L_p만 필터를 곱한다. Faster R-CNN은 L_G = RPN 항 + RoI 항이고, RoI head는 두 모델 모두
    FP 모델의 proposal을 쓴다(박스 짝을 맞추기 위해). RoI 회귀는 FP가 고른 클래스의 delta.
  - weight: 채널별 비대칭 MSE 초기화, activation: 텐서별 비대칭 MSE 초기화(공식 QDrop과 같다).
"""
import os, sys, json, time, argparse, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common import build_runner, COCO                                  # noqa: E402
from ptq import fold_bn, load_calib, to_batch                          # noqa: E402
from quant.fake_quant import QuantConv2d                               # noqa: E402
from quant.quant_model import wrap_convs, set_mode                     # noqa: E402
from quant.adaround import (convert_to_adaround, list_adaround_convs, lp_rec_loss, temp_decay,   # noqa: E402
                            free_cpu_mem, AdaRoundQuantConv2d)

W_LR, REG_W, WARMUP, A_LR, DROP = 1e-3, 0.01, 0.2, 4e-5, 0.5
THETA_C, THETA_I, LAMBDA = 2e-4, 0.1, 1.0


# ------------------------------------------------------------------ 모델 준비
class LinearAsConv(nn.Module):
    """RoI head의 nn.Linear를 1x1 conv로 바꿔 conv 양자화 모듈을 그대로 쓴다(수학적으로 같은 함수)."""
    def __init__(self, lin):
        super().__init__()
        self.conv = nn.Conv2d(lin.in_features, lin.out_features, 1, bias=lin.bias is not None)
        with torch.no_grad():
            self.conv.weight.copy_(lin.weight.view(lin.out_features, lin.in_features, 1, 1))
            if lin.bias is not None:
                self.conv.bias.copy_(lin.bias)

    def forward(self, x):
        return self.conv(x.reshape(x.size(0), -1, 1, 1)).flatten(1)


_orig_forward = AdaRoundQuantConv2d.forward


class LLAQ(nn.Module):
    """11-16 Reg-PTQ 4.3절 Learnable Logarithmic-Affine Quantizer(head weight 전용). 본문 식 9-10과 추론 식
    X^T W ~ X^T 2^{round(psi(W))} s_w만 있고 보충자료에도 세부 식이 없어서 아래처럼 정했다.
      - 위치 mu: 출력 채널별 중앙값(라플라스 분포의 위치 모수). 크기 r = |w - mu| / s, s = 채널별 max |w - mu| (r in [0, 1]).
      - psi(r) = k * log2(r) + a. w >= mu 쪽은 (k+, a+), 반대쪽은 (k-, a-)로 따로 학습한다(식 9).
      - 로그 공간에서 균일 양자화: q = clamp(round(psi), 0, n), n = 2^(b-1) - 1(부호 1bit).
        q = 0은 0(= mu), q >= 1은 r_hat = 2^((q - a) / k). 복원 w_hat = mu + sign * s * r_hat.
      - 초기값: 층마다 weight MSE가 가장 작은 (k, a)를 격자 탐색(k 1~4, a n-1~n+4; 양쪽 공통). round는 STE,
        k, a는 재구성 손실로 학습한다(AdaRound 반올림 대신).
    """

    def __init__(self, w, bits):
        super().__init__()
        co = w.shape[0]
        wf = w.detach().reshape(co, -1)
        mu = wf.median(1).values
        s = (wf - mu[:, None]).abs().max(1).values.clamp(min=1e-8)
        shp = (co,) + (1,) * (w.dim() - 1)
        self.register_buffer("mu", mu.view(shp))
        self.register_buffer("s", s.view(shp))
        self.n = 2 ** (bits - 1) - 1
        self.k = nn.Parameter(torch.ones(2, device=w.device))            # [k+, k-]
        self.a = nn.Parameter(torch.full((2,), float(self.n), device=w.device))
        # 초기화: "scaling factor는 통상적인 탐색으로 초기화"(4.1절). 층마다 weight 오차(MSE)가 가장 작은 (k, a)를 격자 탐색.
        with torch.no_grad():
            best = None
            for kk in torch.arange(1.0, 4.01, 0.25).tolist():
                for aa in torch.arange(self.n - 1.0, self.n + 4.01, 0.5).tolist():
                    self.k.fill_(kk); self.a.fill_(aa)
                    err = float((self(w.detach()) - w.detach()).pow(2).mean())
                    if best is None or err < best[0]:
                        best = (err, kk, aa)
            self.k.fill_(best[1]); self.a.fill_(best[2])

    def forward(self, w):
        d = w - self.mu
        pos = d >= 0
        k = torch.where(pos, self.k[0], self.k[1]).clamp(min=0.05)
        a = torch.where(pos, self.a[0], self.a[1])
        r = (d.abs() / self.s).clamp(min=1e-12)
        psi = k * torch.log2(r) + a
        q = psi + (torch.round(psi) - psi).detach()                         # round STE
        q = torch.clamp(q, 0, self.n)
        rh = torch.where(q >= 0.5, torch.exp2((q - a) / k), torch.zeros_like(q))
        return self.mu + torch.where(pos, 1.0, -1.0) * self.s * rh


def _forward_open(self, x):
    """알고리즘 1의 open_quant: 아직 열지 않은 단위는 FP(원래 conv) 그대로. llaq가 붙은 conv(head)는 weight를 LLAQ로."""
    if not getattr(self, "opened", True):
        return self.conv(x)
    lq = getattr(self, "llaq", None)
    if lq is None:
        return _orig_forward(self, x)
    if self.quantized and self.a_obs.ready and self.act_quant_enabled:
        x = self.quant_act(x)
    return F.conv2d(x, lq(self.conv.weight), self.conv.bias, self.conv.stride,
                    self.conv.padding, self.conv.dilation, self.conv.groups)


AdaRoundQuantConv2d.forward = _forward_open


def quant_targets(model):
    """양자화할 하위 모듈(경로, 모듈)과 재구성 단위 [(이름, 경로, block 여부)]. 마지막 예측층은 제외."""
    units = [("stem", "backbone.conv1", False)]
    for li in range(1, 5):
        for bi in range(len(getattr(model.backbone, f"layer{li}"))):
            units.append((f"layer{li}.{bi}", f"backbone.layer{li}.{bi}", True))
    for k in range(len(model.neck.lateral_convs)):
        units.append((f"fpn.lat{k}", f"neck.lateral_convs.{k}.conv", False))
    for k in range(len(model.neck.fpn_convs)):
        units.append((f"fpn.out{k}", f"neck.fpn_convs.{k}.conv", False))
    if hasattr(model, "bbox_head"):
        h = model.bbox_head
        units += [(f"head.cls{i}", f"bbox_head.cls_convs.{i}.conv", False) for i in range(len(h.cls_convs))]
        units += [(f"head.reg{i}", f"bbox_head.reg_convs.{i}.conv", False) for i in range(len(h.reg_convs))]
    else:
        units.append(("rpn.conv", "rpn_head.rpn_conv", False))
        # LinearAsConv 자체를 단위로(FP 쪽은 Linear 그대로라 경로와 출력 모양 [N, 1024]이 같다)
        units += [(f"roi.fc{i}", f"roi_head.bbox_head.shared_fcs.{i}", True)
                  for i in range(len(model.roi_head.bbox_head.shared_fcs))]
    return units


def prepare_q(q_model, w_bits, a_bits):
    if hasattr(q_model, "roi_head"):
        fcs = q_model.roi_head.bbox_head.shared_fcs
        for i in range(len(fcs)):
            fcs[i] = LinearAsConv(fcs[i])
    n = wrap_convs(q_model.backbone, w_bits, a_bits) + wrap_convs(q_model.neck, w_bits, a_bits)
    if hasattr(q_model, "bbox_head"):
        h = q_model.bbox_head
        n += wrap_convs(h.cls_convs, w_bits, a_bits) + wrap_convs(h.reg_convs, w_bits, a_bits)
    else:
        n += wrap_convs(q_model.rpn_head, w_bits, a_bits, skip_names={"rpn_cls", "rpn_reg"})
        n += wrap_convs(q_model.roi_head.bbox_head.shared_fcs, w_bits, a_bits)
    stem = q_model.backbone.conv1
    stem.w_bits, stem.a_obs.bits = 8, 8
    return n


# ------------------------------------------------------------------ 검출기 출력(전역 손실용)
def _dense(scores, preds, C):
    """[1, A*C, H, W] 목록 -> [N, C], [N, 4] (mmdet 순서: 위치마다 anchor)."""
    s = torch.cat([x.permute(0, 2, 3, 1).reshape(-1, C) for x in scores])
    b = torch.cat([x.permute(0, 2, 3, 1).reshape(-1, 4) for x in preds])
    return s, b


def head_out(model, feats, ds, rois=None):
    """전역 손실에 쓰는 출력 dict. Faster R-CNN은 rois(FP proposal)를 받아 RoI head를 그 위에서 돌린다."""
    if hasattr(model, "bbox_head"):
        h = model.bbox_head
        cs, bp = h(feats)
        s, b = _dense(cs, bp, h.cls_out_channels)
        return dict(dense=(s, b), sizes=[c.shape[-2:] for c in cs])
    rh = model.rpn_head
    cs, bp = rh(feats)
    s, b = _dense(cs, bp, rh.cls_out_channels)
    out = dict(dense=(s, b), sizes=[c.shape[-2:] for c in cs], rpn_raw=(cs, bp))
    if rois is not None:
        r = model.roi_head._bbox_forward(feats, rois)
        out["roi"] = (r["cls_score"], r["bbox_pred"])
    return out


def fp_rois(model, out, ds):
    from mmdet.structures.bbox import bbox2roi
    cs, bp = out["rpn_raw"]
    props = model.rpn_head.predict_by_feat(cs, bp, batch_img_metas=[d.metainfo for d in ds], rescale=False)
    return bbox2roi([p.bboxes for p in props])


def _iou(a, b):
    lt = torch.max(a[:, :2], b[:, :2]); rb = torch.min(a[:, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(1)
    area = lambda x: (x[:, 2:] - x[:, :2]).clamp(min=0).prod(1)
    return inter / (area(a) + area(b) - inter).clamp(min=1e-6)


class GlobalLoss:
    """식 8. anchors는 이미지마다 다시 만든다(특징 크기가 이미지마다 다름)."""

    def __init__(self, fp_model, device):
        self.m, self.device = fp_model, device
        self.dense_head = fp_model.bbox_head if hasattr(fp_model, "bbox_head") else fp_model.rpn_head

    def _anchors(self, sizes):
        return torch.cat(self.dense_head.prior_generator.grid_priors(sizes, device=self.device))

    def dense(self, fo, qo):
        (sf, bf), (sq, bq) = fo["dense"], qo["dense"]
        yf = sf.float().sigmoid()
        ce = F.binary_cross_entropy_with_logits(sq.float(), yf, reduction="none").sum(1)
        with torch.no_grad():
            anc = self._anchors(fo["sizes"])
            coder = self.dense_head.bbox_coder
            keep = (yf.max(1).values >= THETA_C) & (_iou(coder.decode(anc, bf.float()), coder.decode(anc, bq.detach().float())) >= THETA_I)
        lp = (bq.float() - bf.float()).abs().sum(1) * keep
        return (ce + LAMBDA * lp).mean(), float(keep.float().mean())

    def roi(self, fo, qo, rois):
        (cf, bf), (cq, bq) = fo["roi"], qo["roi"]
        pf = F.softmax(cf.float(), 1)
        ce = -(pf * F.log_softmax(cq.float(), 1)).sum(1)
        n, C = pf.size(0), pf.size(1) - 1
        idx = pf[:, :C].argmax(1)
        ar = torch.arange(n, device=pf.device)
        df, dq = bf.float().view(n, C, 4)[ar, idx], bq.float().view(n, C, 4)[ar, idx]
        with torch.no_grad():
            coder = self.m.roi_head.bbox_head.bbox_coder
            keep = (pf[:, :C].max(1).values >= THETA_C) & (_iou(coder.decode(rois[:, 1:], df), coder.decode(rois[:, 1:], dq.detach())) >= THETA_I)
        lp = (dq - df).abs().sum(1) * keep
        return (ce + LAMBDA * lp).mean(), float(keep.float().mean())

    def __call__(self, fo, qo, rois=None):
        L, k = self.dense(fo, qo)
        if rois is not None:
            L2, k2 = self.roi(fo, qo, rois)
            return L + L2, (k, k2)
        return L, (k,)


# ------------------------------------------------------------------ 순전파 도우미
class _Stop(Exception):
    pass


class Tap:
    """단위 출력 수집 hook. stop=True면 첫 호출 뒤 순전파를 끊는다(QDrop 국소 손실만 쓸 때 계산 절약)."""

    def __init__(self, mod):
        self.outs, self.stop = [], False
        self.h = mod.register_forward_hook(self._hook)

    def _hook(self, m, i, o):
        self.outs.append(o)
        if self.stop:
            raise _Stop

    def reset(self, stop=False):
        self.outs, self.stop = [], stop

    def remove(self):
        self.h.remove()


def run(model, x, ds, tap, need_head, rois=None, fp=False):
    """need_head=False면 tap 지점에서 멈춘다. FP 쪽은 Faster R-CNN proposal도 만든다."""
    tap.reset(stop=not need_head)
    try:
        feats = model.extract_feat(x)
        out = head_out(model, feats, ds, None if fp else rois)
        if fp and "rpn_raw" in out:
            rois = fp_rois(model, out, ds)
            r = model.roi_head._bbox_forward(feats, rois)
            out["roi"] = (r["cls_score"], r["bbox_pred"])
        return out, rois
    except _Stop:
        return None, rois


# ------------------------------------------------------------------ 재구성
def local_fisher(outs, tgts, gs):
    """식 3(대각 Fisher). 공식 BRECQ 구현처럼 (Ô-O)^2 g^2를 채널로 합하고 나머지 축 평균, 호출(레벨) 평균."""
    ls = [((o - t).pow(2) * g.pow(2)).sum(1).mean() for o, t, g in zip(outs, tgts, gs)]
    return sum(ls) / len(ls)


def local_lp(outs, tgts):
    return sum(lp_rec_loss(o, t) for o, t in zip(outs, tgts)) / len(outs)


def recon(q_model, fp_model, units, calib, device, iters, method, log=print):
    head_paths = ("bbox_head", "rpn_head", "roi_head")
    gl = GlobalLoss(fp_model, device)
    for ti, (label, path, is_block) in enumerate(units):
        qm, fm = q_model.get_submodule(path), fp_model.get_submodule(path)
        convs = list_adaround_convs(qm) if is_block else [qm]
        for c in convs:
            c.opened = True
            c.alpha.requires_grad_(True); c.lsq_act.delta.requires_grad_(True)   # 이 단위만 학습(앞 단위로 역전파하지 않게)
        qtap, ftap = Tap(qm), Tap(fm)
        fg = method.startswith("regptq")                                  # FGIC(regptq, regptq_llaq)
        lqs = [c.llaq for c in convs if getattr(c, "llaq", None) is not None]
        for lq in lqs:
            lq.k.requires_grad_(True); lq.a.requires_grad_(True)
        need_head = fg or path.startswith(head_paths)
        t0 = time.time()
        # 알고리즘 1 3-8행: 단위를 연 상태(초기 반올림, activation 양자화, drop 없음)에서 g = dL_G/dÔ를 이미지마다 잰다.
        gbuf = None
        if fg:
            gbuf, kf = [], []
            for x, ds in calib:
                x = x.to(device)
                with torch.no_grad():
                    fo, rois = run(fp_model, x, ds, ftap, True, fp=True)
                with torch.enable_grad():
                    qo, _ = run(q_model, x, ds, qtap, True, rois=rois)
                    L, k = gl(fo, qo, rois)
                    gs = torch.autograd.grad(L, qtap.outs, allow_unused=True)
                gbuf.append([(g.abs() if g is not None else torch.zeros_like(o)).add(1.0).half().cpu()
                             for g, o in zip(gs, qtap.outs)])
                kf.append(k[-1])
            tp = time.time() - t0
        for c in convs:
            c.soft, c.use_lsq, c.qdrop_prob = True, c.a_obs.bits < 16, DROP
        wp = [c.alpha for c in convs if getattr(c, "llaq", None) is None] + [p for lq in lqs for p in (lq.k, lq.a)]
        ap = [c.lsq_act.delta for c in convs if c.use_lsq]
        w_opt = torch.optim.Adam(wp, lr=W_LR)
        a_opt = torch.optim.Adam(ap, lr=A_LR) if ap else None
        a_sch = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=iters, eta_min=0.0) if ap else None
        sl, sg = 0.0, 0.0
        for it in range(iters):
            j = random.randrange(len(calib))
            x, ds = calib[j]
            x = x.to(device)
            with torch.no_grad():
                fo, rois = run(fp_model, x, ds, ftap, need_head, fp=True)
            tgts = [t.detach() for t in ftap.outs]
            qo, _ = run(q_model, x, ds, qtap, need_head, rois=rois)
            outs = qtap.outs
            if fg:
                Lh = local_fisher(outs, tgts, [g.to(device).float() for g in gbuf[j]])
                Lg, _ = gl(fo, qo, rois)
                loss = Lh + Lg
                sg += float(Lg)
            else:
                Lh = local_lp(outs, tgts)
                loss = Lh
            sl += float(Lh)
            if it >= WARMUP * iters:
                beta = temp_decay(it, iters, WARMUP)
                loss = loss + REG_W * sum(c.reg_loss(beta) for c in convs if getattr(c, "llaq", None) is None)
            w_opt.zero_grad()
            if a_opt is not None:
                a_opt.zero_grad()
            loss.backward()
            w_opt.step()
            if a_opt is not None:
                a_opt.step(); a_sch.step()
        for c in convs:
            c.soft, c.qdrop_prob = False, 0.0
            c.alpha.requires_grad_(False); c.lsq_act.delta.requires_grad_(False)
        for lq in lqs:
            lq.k.requires_grad_(False); lq.a.requires_grad_(False)
        qtap.remove(); ftap.remove()
        msg = f"  [{method}] {ti+1}/{len(units)} {label} ({len(convs)} conv) {time.time()-t0:.0f}s L_H {sl/iters:.4g}"
        if fg:
            msg += f" L_G {sg/iters:.4g} (g {tp:.0f}s, 필터 통과 {100*float(np.mean(kf)):.2f}%)"
        if lqs:
            msg += f" | LLAQ k+/k- {lqs[0].k[0]:.3f}/{lqs[0].k[1]:.3f} a+/a- {lqs[0].a[0]:.2f}/{lqs[0].a[1]:.2f}"
        log(msg, flush=True)
        del gbuf
        free_cpu_mem()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--w-bits", type=int, default=4)
    ap.add_argument("--a-bits", type=int, default=4)
    ap.add_argument("--method", choices=["naive", "qdrop", "regptq", "regptq_llaq"], default="regptq")
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--eval-cap", type=int, default=0, help="0이면 val2017 전체")
    ap.add_argument("--units-cap", type=int, default=0, help="시험용: 앞 N개 단위만 재구성(나머지는 RTN)")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda:0"
    os.makedirs(args.work_dir, exist_ok=True)
    print("[args]", vars(args), flush=True)

    runner = build_runner(args.model, args.work_dir, batch=2)   # 평가 batch 2(두 작업이 GPU 하나를 나눠 쓰게)
    fp_model = runner.model.to(device).eval()
    q_model = copy.deepcopy(fp_model)
    for p in fp_model.parameters():
        p.requires_grad_(False)
    print(f"[bn] 접은 BN: fp {fold_bn(fp_model.backbone)}, q {fold_bn(q_model.backbone)}", flush=True)
    nq = prepare_q(q_model, args.w_bits, args.a_bits)
    q_model.to(device).eval()
    for p in q_model.parameters():
        p.requires_grad_(False)

    _, items = load_calib(runner.cfg, args.calib)
    calib = [to_batch(fp_model, it, "cpu") for it in items]
    del items

    # activation 초기 범위: MSE(L2.4) 탐색(공식 QDrop). FP 네트워크 위에서 한 번에 모은다.
    for m in q_model.modules():
        if isinstance(m, QuantConv2d):
            m.a_obs.method = "mse"
    set_mode(q_model, calibrating=True, quantized=False)
    with torch.no_grad():
        for x, ds in calib:
            x = x.to(device)
            feats = q_model.extract_feat(x)
            o = head_out(q_model, feats, ds)
            if "rpn_raw" in o:
                q_model.roi_head._bbox_forward(feats, fp_rois(q_model, o, ds))
    for m in q_model.modules():
        if isinstance(m, QuantConv2d):
            m.a_obs.freeze(); m.freeze_weight_quant()
    set_mode(q_model, calibrating=False, quantized=True)
    convert_to_adaround(q_model, channelwise_smult=False)
    convs = list_adaround_convs(q_model)
    for p in q_model.parameters():
        p.requires_grad_(False)
    if args.method == "regptq_llaq":
        # LLAQ는 head weight에만(논문 표 3: "applying ... LLAQ quantizers on detection head only")
        hp = ("bbox_head", "rpn_head", "roi_head")
        nl = 0
        for name, c in q_model.named_modules():
            if isinstance(c, AdaRoundQuantConv2d) and name.startswith(hp):
                c.llaq = LLAQ(c.conv.weight, c.w_bits); nl += 1
        print(f"[llaq] head conv {nl}개에 LLAQ", flush=True)
    units = quant_targets(q_model)
    covered = {id(c) for _, p, b in units for c in (list_adaround_convs(q_model.get_submodule(p)) if b else [q_model.get_submodule(p)])}
    assert covered == {id(c) for c in convs}, f"단위에 안 들어간 양자화 conv: {len(convs) - len(covered)}"
    print(f"[quant] conv {nq}개 양자화(head 포함, 마지막 예측층 FP), stem W8A8, W{args.w_bits}A{args.a_bits}, "
          f"단위 {len(units)}개", flush=True)

    t0 = time.time()
    if args.method == "naive":
        for c in convs:
            c.soft, c.opened = False, True
    else:
        for c in convs:
            c.opened = False
        ru = units[:args.units_cap] if args.units_cap else units
        recon(q_model, fp_model, ru, calib, device, args.iters, args.method)
        for c in convs:
            if not c.opened:
                c.soft, c.opened = False, True
    calib_s = time.time() - t0
    print(f"[build] {args.method} 완료 {calib_s:.0f}s", flush=True)
    del calib
    free_cpu_mem()

    runner.model = q_model
    if args.eval_cap:
        full = json.load(open(COCO + "annotations/instances_val2017.json"))
        keep = {im["id"] for im in full["images"][:args.eval_cap]}
        sub = dict(full, images=[im for im in full["images"] if im["id"] in keep],
                   annotations=[a for a in full["annotations"] if a["image_id"] in keep])
        sub_path = os.path.abspath(os.path.join(args.work_dir, f"val_cap{args.eval_cap}.json"))
        json.dump(sub, open(sub_path, "w"))
        runner._test_dataloader["dataset"]["ann_file"] = sub_path
        runner._test_evaluator["ann_file"] = sub_path
    with torch.no_grad():
        m = runner.test()
    res = {k: round(float(v) * 100, 1) for k, v in m.items() if "bbox" in k}
    print("[result]", args.model, f"W{args.w_bits}A{args.a_bits}", args.method, json.dumps(res), flush=True)
    json.dump(dict(args=vars(args), result=res, calib_s=calib_s),
              open(os.path.join(args.work_dir, f"result_{args.model}_w{args.w_bits}a{args.a_bits}_{args.method}.json"), "w"))


if __name__ == "__main__":
    main()
