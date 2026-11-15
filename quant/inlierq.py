"""
InlierQ(Kim et al., ICLR 2026, arXiv 2602.03472, "Inlier-Centric Post-Training Quantization for Object Detection
Models") 기준선. 11-15 재구현. 공식 코드가 없어서 논문 본문(3-4절, 부록 A, 알고리즘 1)만 보고 옮겼다.

논문의 방법:
  (식 7)  heatmap 대리 손실 L = -(1/KC) sum_c sum_{k<=K} log H_[k],c  (클래스 c마다 heatmap 상위 K개 값의 NLL)
  (식 9)  위치(volume) saliency h(x) = sum_m |dL/dx_m|  (그 위치 activation 벡터에 대한 L의 gradient L1)
  (식 10-11) h에 2성분 GMM을 EM으로 맞추고, inlier 성분의 posterior >= tau인 위치를 inlier 집합 I로 둔다
  (식 12) argmin_S E_{x in I}[dx^T H dx], H는 FIM(E[g g^T])으로 근사
  BRECQ(층별 순차, FIM 기반 재구성) 위에서 동작하고, BRECQ와 같은 순서로 대상 단위를 하나씩 처리한다.

이 저장소의 BRECQ(quant/brecq.optimize_brecq) 위에 옮기며 정한 것(논문에 값이 없거나 검출기 구조가 달라서):
  - 손실 대상: 재구성 단위(block 또는 head conv)의 출력. BRECQ의 재구성 손실(lp_rec_loss)을 아래 가중 손실로
    바꾸고, 나머지(alpha 반올림 + LSQ activation step 공동 최적화, iters, batch, 순서)는 brecq와 그대로 같다.
    논문은 head를 FP16으로 두지만 이 저장소 프로토콜은 head까지 양자화하므로 head conv에도 같은 손실을 쓴다.
  - heatmap: head의 클래스 logit(어휘 = calibration 어휘, COCO 80) 위의 sigmoid. anchor 축에서 클래스마다 상위 K.
    YOLO-World의 클래스 점수 맵이 곧 heatmap이다(논문은 2D 검출기의 heatmap 정의를 적지 않았다).
  - gradient: FP 점에서 잰다. 대상의 FP 짝 모듈 출력을 grad를 받는 leaf로 바꿔 끼우고 나머지 FP 네트워크를 통과시켜
    L을 역전파한다(PD-Quant의 PDContext와 같은 바꿔 끼우기). 이미지마다 한 번, 대상마다 한 번.
  - FIM: 대각 근사 w = g^2 (BRECQ의 Fisher 대각과 같은 형태). 손실 = inlier 위치 평균 sum_c w_c (q - fp)_c^2.
    w는 inlier 원소 평균이 1이 되게 정규화한다(w = 1이면 BRECQ lp_rec_loss와 같은 규모 -> 반올림 정규화와의 균형 유지).
  - GMM: 대상마다 calibration 전체 위치의 log h에 1차원 2성분 GMM(EM 100회, 사분위수 초기화). 평균이 큰 성분이
    inlier(판정에 영향이 큰 위치). h 그대로는 꼬리가 무거워 한 성분이 0 근처에 몰려서 log를 쓴다.
  - tau = 0.5(베이즈 결정 경계), K = 300(논문은 학습 때 값을 기본으로 둔다: DETR3D 300 query. YOLO 계열 추론의
    max_det 300에 맞춤). 환경변수 IQ_TAU, IQ_K로 바꿀 수 있다.
  - 논문 표 1의 Ours는 표 2의 inlier만 쓴 행과 같으므로 anomaly 항(lambda_A)은 두지 않는다.
"""

from __future__ import annotations
import os
import torch
import torch.nn.functional as F

from .pdquant import _find_head

IQ_TAU = float(os.environ.get("IQ_TAU", "0.5"))
IQ_K = int(os.environ.get("IQ_K", "300"))
IQ_EM_ITERS = 100


def gmm2_posterior(v: torch.Tensor, iters: int = IQ_EM_ITERS):
    """1차원 2성분 GMM(EM). v: [N] float. 반환: (평균이 큰 성분의 posterior [N], (mu, sigma, pi) 요약)."""
    v = v.double()
    q = torch.quantile(v[:: max(1, v.numel() // 1_000_000)], torch.tensor([0.25, 0.75], dtype=v.dtype))
    mu = q.clone()
    sd = torch.full((2,), float(v.std().clamp(min=1e-6)), dtype=v.dtype)
    pi = torch.tensor([0.5, 0.5], dtype=v.dtype)
    for _ in range(iters):
        lp = -0.5 * ((v[:, None] - mu) / sd) ** 2 - torch.log(sd) + torch.log(pi)
        r = torch.softmax(lp, dim=1)
        nk = r.sum(0).clamp(min=1e-9)
        mu = (r * v[:, None]).sum(0) / nk
        sd = torch.sqrt((r * (v[:, None] - mu) ** 2).sum(0) / nk).clamp(min=1e-6)
        pi = nk / nk.sum()
    hi = int(torch.argmax(mu))
    return r[:, hi].float(), (mu.tolist(), sd.tolist(), pi.tolist())


class InlierQContext:
    """FP 모델 전체(fp_full = DetectionModel)로 대상별 inlier 마스크 x FIM 가중치를 만든다."""

    def __init__(self, fp_full, calib_list, device, K=IQ_K, tau=IQ_TAU):
        self.fp, self.calib, self.device, self.K, self.tau = fp_full, calib_list, device, K, tau
        for p in fp_full.parameters():
            p.requires_grad_(False)
        self.head = _find_head(fp_full)
        self.nbox = 4 * int(getattr(self.head, "reg_max", 16))
        self._cap = None
        self._hh = self.head.register_forward_hook(self._head_hook)

    def _head_hook(self, m, i, o):
        self._cap = o[1] if isinstance(o, tuple) else o

    def _scores(self):
        xs = self._cap
        if isinstance(xs, dict):
            return xs["scores"].transpose(1, 2)                   # [B, A, nc] logit
        b = xs[0].size(0)
        raw = torch.cat([x.reshape(b, x.size(1), -1) for x in xs], dim=2).transpose(1, 2)
        return raw[..., self.nbox:]

    def heat_loss(self, logits):
        """식 7: 클래스마다 anchor 상위 K개 heatmap 값의 NLL 평균. sigmoid는 단조라 logit에서 top-K를 고른다."""
        k = min(self.K, logits.size(1))
        top = logits.topk(k, dim=1).values                        # [B, K, nc]
        return -F.logsigmoid(top).mean()

    def weights(self, fm, n, label=""):
        """대상 fm(FP 짝 모듈)의 출력 위치별 가중치 목록(이미지별 [1, C, H, W] fp16 CPU; inlier 밖은 0).
        출력이 텐서가 아니거나 4차원이 아니면 None(그 대상은 BRECQ 손실 그대로)."""
        gs, hs, c = [], [], None
        state = {}

        def hook(m, i, o):
            if not torch.is_tensor(o) or o.dim() != 4:
                state["bad"] = True
                return o
            z = o.detach().float().requires_grad_(True)
            state["z"] = z
            return z + 0.0                                        # leaf 그대로 내보내면 뒤의 in-place 활성화가 막힌다
        h = fm.register_forward_hook(hook)
        try:
            for j in range(n):
                state.clear()
                self._cap = None
                with torch.enable_grad():
                    self.fp(self.calib[j].to(self.device))
                    if state.get("bad") or "z" not in state:
                        return None
                    L = self.heat_loss(self._scores().float())
                    g, = torch.autograd.grad(L, state["z"])
                if c is None:                                     # fp16 저장용 규모 맞춤(대상마다 상수 하나)
                    c = 1.0 / float(g.pow(2).mean().sqrt().clamp(min=1e-30))
                g = g * c
                gs.append(g.half().cpu())
                hs.append(g.abs().sum(1).flatten().cpu())         # 식 9 (상수 c는 GMM on log h에서 평행이동일 뿐)
        finally:
            h.remove()
        hv = torch.cat(hs)
        # gradient가 정확히 0인 위치(top-K anchor에 닿지 않는 위치)는 판정에 영향이 없으므로 inlier가 아니다.
        # log(0)이 GMM 한 성분을 독차지하지 않도록 h > 0인 위치에만 GMM을 맞춘다.
        nz = hv > 0
        if int(nz.sum()) < 2:
            # head 박스 갈래(cv2)처럼 heatmap에 닿지 않는 대상: 식 7 손실의 gradient가 전부 0이라 inlier가 정의되지 않는다.
            # 논문은 head를 FP16으로 둬서 이 경우가 없다. BRECQ 재구성 손실을 그대로 쓴다.
            print(f"    [inlierq] {label} heatmap gradient 0 -> BRECQ 손실 그대로", flush=True)
            return None
        mask = torch.zeros_like(nz)
        post, (mu, sd, pi) = gmm2_posterior(torch.log(hv[nz]))
        mask[nz] = post >= self.tau
        frac = float(mask.float().mean())
        out, off, tot, cnt = [], 0, 0.0, 0
        for g in gs:
            hw = g.shape[2] * g.shape[3]
            mk = mask[off:off + hw].view(1, 1, g.shape[2], g.shape[3]); off += hw
            w = g.float() * mk                                      # 제곱근 형태로 둔다(제곱은 손실에서)
            tot += float(w.pow(2).sum()); cnt += int(mk.sum()) * g.shape[1]
            out.append(w)
        norm = (tot / max(cnt, 1)) ** 0.5                          # 11-15: g^2를 fp16으로 두면 inf가 나서 g/rms로 저장
        out = [(w / max(norm, 1e-30)).half() for w in out]
        print(f"    [inlierq] {label} inlier {frac*100:.1f}% (h=0 {float((~nz).float().mean())*100:.1f}%, GMM mu {mu[0]:.2f}/{mu[1]:.2f}, pi {pi[0]:.2f}/{pi[1]:.2f})",
              flush=True)
        return out

    @staticmethod
    def loss(pred, target, w):
        """inlier 위치 평균 sum_c w_c^2 (pred - target)_c^2. w = sqrt(FIM 대각) [B, C, H, W](inlier 밖 0).
        11-15: 이전에는 w에 g^2를 fp16으로 담아 손실에서 그대로 곱했다(runs/264). 값이 fp16 범위를 넘으면 inf가 된다."""
        d = ((pred - target) * w).pow(2)
        npos = (w != 0).any(1).sum().clamp(min=1)
        return d.sum() / npos

    def close(self):
        self._hh.remove()
