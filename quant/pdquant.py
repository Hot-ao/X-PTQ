"""
head 캡처 도우미.

  _find_head(model)  : detection head(model.model[-1])를 찾는다.
  _CV4Capture        : head.cv4(레벨별 contrastive head)의 출력(유사도 맵)과, capture_input=True면
                       입력(region feature)도 grad를 유지한 채 캡처한다.

파일 이름은 원래 여기 있던 PD-Quant 기준선 구현에서 왔다(지금 파이프라인에서는 쓰지 않아 제거,
옛 구현은 src/quant/pdquant.py). promptcal.py와 diag_w4_sensitivity.py가 위 두 도우미를 쓴다.
"""

from __future__ import annotations
import torch


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
