"""
입력 채널별 scale 이전(migration) 탐색 (09-29).

근거: runs/128 W4 민감도 진단에서 W4 손상이 C2f concat을 받는 1x1 conv(12.cv2 등)에 몰려 있고,
그 conv들이 입력 채널별 weight 크기 편차가 가장 큰 conv와 겹친다(12.cv2 13.3배). 출력 채널별
weight 양자화는 입력 채널 방향 편차를 흡수하지 못하므로, 편차를 activation 쪽(A8은 여유가 있음)으로
옮긴다.

  conv(x, W) = conv(x / s, W * diag(s))       (FP로는 항등)
  양자화:     conv(Q_a(x / s), Q_w(W * diag(s)))   Q_a: per-tensor, Q_w: 출력 채널별(기존 그대로)

s는 SmoothQuant 형태 s_j = max|x_j|^a / max|W_:j|^(1-a) 로 두고, conv마다 a를 grid search로
고른다(AWQ처럼 그 conv 출력의 재구성 오차 최소). 후보에는 항상 "이전 없음(s=1)"을 포함해 어떤
conv도 기존보다 나빠지지 않게 한다. 입력은 FP 모델 기준(앞 conv들의 이전은 FP에서 항등이라
순서와 무관).

주의(배포): 한 텐서를 여러 conv가 소비하면(C2f split 등) 비용 0 배포에는 그 소비자들이 같은 s를
공유해야 한다. 그 제약을 지키는 탐색·적용은 channel_graph.py(search_and_apply_tied)에 있고, 이 파일에는
거기서 쓰는 도우미(_act_quant_mse, _conv)만 남아 있다. conv마다 독립 s를 쓰던 1차 구현(search_and_apply)과
채널 단위 혼합 정밀도(select_hi_cols)는 확정 방법에 쓰이지 않아 제거했다(옛 구현은 promptcal-ptq의 src/quant/migrate.py).

무작위성을 쓰지 않는다(표본 추출은 stride) -- 전역 RNG 스트림 불변.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


@torch.no_grad()
def _act_quant_mse(x: torch.Tensor, bits: int, p: float = 2.4, n_cand: int = 80, max_elems: int = 2_000_000):
    """ActObserver(method='mse')와 같은 탐색을 RNG 없이(stride 표본) 수행하고 x를 fake-quant."""
    v = x.detach().flatten().float()
    if v.numel() > max_elems:
        v = v[:: v.numel() // max_elems + 1]
    x_min = torch.minimum(v.min(), torch.zeros((), device=v.device))
    x_max = torch.maximum(v.max(), torch.zeros((), device=v.device))
    qmax = 2 ** bits - 1
    best, rng = None, (x_min, x_max)
    for i in range(n_cand):
        f = 1.0 - i * 0.01
        mn, mx = x_min * f, x_max * f
        sc = ((mx - mn) / qmax).clamp(min=1e-8)
        zp = torch.round(-mn / sc)
        vq = (torch.clamp(torch.round(v / sc) + zp, 0, qmax) - zp) * sc
        score = (vq - v).abs().pow(p).mean()
        if best is None or score < best:
            best, rng = score, (mn, mx)
    mn, mx = rng
    sc = ((mx - mn) / qmax).clamp(min=1e-8)
    zp = torch.round(-mn / sc)
    return (torch.clamp(torch.round(x / sc) + zp, 0, qmax) - zp) * sc


def _conv(c, x, w):
    k = c.conv
    return F.conv2d(x, w, k.bias, k.stride, k.padding, k.dilation, k.groups)
