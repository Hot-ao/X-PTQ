"""모델 종류별 생성·어휘 전환·텍스트 bank (10-04: YOLOE 지원).

- YOLO-World(yolov8*-world*.pt): ultralytics YOLOWorld 그대로.
- YOLOE(yoloe-*.pt): 공개 가중치는 분할(seg) 모델이라, 같은 이름의 검출 구성(yoloe-v8s.yaml 등)을 만들고
  가중치를 불러온다(head의 mask 분기는 버린다). 텍스트는 MobileCLIP → RepRTA(head.reprta)를 거친 상수(pe)로 head의
  contrastive에만 쓰인다. neck에 텍스트 융합이 없어서 게이트 교환(G)은 적용되지 않는다(gate_commute_all이 아무것도 안 함).
- 11-05: YOLOE_SEG(run_comparison --yoloe-seg)를 켜면 seg 모델 그대로 만든다(mask 분기 포함, 양자화·재구성 대상).
  head를 SegHeadDetView로 바꿔 평소에는 검출 head와 같은 출력(mask 계수를 뺀 것)을 내므로 flip·진단·재구성 코드는
  그대로다. mask 분기는 계산은 하므로 재구성 hook에 잡힌다. LVIS predict 때만 seg_outputs()로 mask를 낸다.
- 11-05: YW_SEG(run_comparison --yw-seg <mask 가중치>)를 주면 YOLO-World의 WorldDetect에 YOLO-World-Seg(mmyolo
  YOLOWv8InsHeadModule)의 mask 분기를 붙인다(WorldSegHeadView). 검출 부분은 그대로라 G·P·M이 모두 적용된다.
  mask 가중치는 scripts/convert_yw_seg.py로 변환한다. 원본과 다른 점은 proto 업샘플(최근접, ConvTranspose 아님)뿐.
"""
from __future__ import annotations
import contextlib
import os
import torch

YOLOE_SEG = False
YW_SEG = None


from ultralytics.nn.modules.block import Proto
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import WorldDetect, YOLOESegment


class SegHeadDetView(YOLOESegment):
    """YOLOE-seg head. det_only면 검출 head(YOLOEDetect)와 같은 형식 (y[B, 4+nc, A], preds)을 낸다."""
    det_only = True

    def forward(self, x):
        out = super().forward(x)
        if self.training or self.export or not self.det_only:
            return out
        (y, _proto), preds = out
        return y[:, :-self.nm], preds


class WorldProto(Proto):
    """mmyolo ProtoModule: conv1 -> 최근접 2배 업샘플 -> conv2 -> conv3 (ultralytics Proto는 ConvTranspose2d)."""

    def __init__(self, c1, c_=256, c2=32):
        super().__init__(c1, c_, c2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")


class WorldSegHeadView(WorldDetect):
    """YOLO-World-Seg head = WorldDetect + mask 계수(cv5) + prototype(proto). det_only면 WorldDetect와 같은 출력."""
    det_only = True
    nm = 32

    def forward(self, x, text):
        bs = x[0].shape[0]
        mc = torch.cat([self.cv5[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)   # WorldDetect가 x를 바꾸기 전에
        proto = self.proto(x[0])
        out = super().forward(x, text)
        if self.training or self.export or self.det_only:
            return out
        y, preds = out
        return (torch.cat([y, mc], 1), proto), preds


def attach_world_seg(m, path):
    """YOLOWorld 래퍼 m의 head에 mask 분기를 붙이고 가중치(convert_yw_seg.py 출력)를 넣는다."""
    head = m.model.model[-1]
    ch = [lvl[0].conv.in_channels for lvl in head.cv2]
    c4 = max(ch[0] // 4, WorldSegHeadView.nm)
    head.cv5 = torch.nn.ModuleList(torch.nn.Sequential(Conv(c, c4, 3), Conv(c4, c4, 3), torch.nn.Conv2d(c4, WorldSegHeadView.nm, 1))
                                   for c in ch)
    head.proto = WorldProto(ch[0], 256, WorldSegHeadView.nm)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    own = {k for k in head.state_dict() if k.startswith(("cv5.", "proto."))}
    if set(sd) != own:
        raise ValueError(f"mask 가중치 이름 불일치: 없음 {sorted(own - set(sd))[:5]}, 남음 {sorted(set(sd) - own)[:5]}")
    head.load_state_dict(sd, strict=False)
    p0 = next(head.cv2.parameters())
    for mod in (head.cv5, head.proto):
        mod.to(p0.device, p0.dtype).train(head.training)          # 새 모듈은 train 모드로 생기므로 head 상태에 맞춘다
    head.__class__ = WorldSegHeadView
    return m


def is_seg(model) -> bool:
    return isinstance(model.model.model[-1], (SegHeadDetView, WorldSegHeadView))


@contextlib.contextmanager
def seg_outputs(model):
    """이 안에서만 head가 seg 출력((det+mask 계수, proto), preds)을 낸다(ultralytics predict용)."""
    head = model.model.model[-1]
    head.det_only = False
    try:
        yield
    finally:
        head.det_only = True


def is_yoloe(path: str) -> bool:
    return os.path.basename(str(path)).lower().startswith("yoloe")


class DetAdaptiveMaxPool2d(torch.nn.Module):
    """nn.AdaptiveMaxPool2d와 forward 값·argmax가 같고 backward가 결정적인 대체.

    10-16: ImagePoolingAttn의 AdaptiveMaxPool2d backward(adaptive_max_pool2d_backward_cuda)는 겹치는 창의 gradient를
    atomicAdd로 더해서 --deterministic에서도 실행마다 값이 달라진다(경고만 뜬다). BRECQ/QDrop 재구성이 이 경로로
    역전파하므로 같은 seed·같은 조건에서도 결과가 갈렸다(s W4A8 seed 0 brecq+GPM: 6회 중 3가지 결과, 10-09에
    '조건 묶음 의존성'으로 보였던 현상). argmax는 원래 연산으로 구하고 값은 gather로 뽑는다 -- gather의 backward는
    결정적 모드에서 결정적 scatter_add를 쓴다. 원래 연산과의 gradient 차이는 float 반올림 수준(<= 2.4e-7)."""

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        with torch.no_grad():
            _, idx = torch.nn.functional.adaptive_max_pool2d(x, self.output_size, return_indices=True)
        return x.flatten(2).gather(2, idx.flatten(2)).view(*x.shape[:2], *idx.shape[2:])


def make_pools_deterministic(model):
    """Model 래퍼 안의 nn.AdaptiveMaxPool2d를 모두 DetAdaptiveMaxPool2d로 바꾼다(없으면 아무것도 안 함)."""
    for parent in model.model.modules():
        for name, child in parent.named_children():
            if type(child) is torch.nn.AdaptiveMaxPool2d:
                setattr(parent, name, DetAdaptiveMaxPool2d(child.output_size))
    return model


def model_factory(path: str):
    """build()의 model_cls 자리에 넣는 생성자. 호출하면 ultralytics Model 래퍼를 돌려준다."""
    if is_yoloe(path):
        from ultralytics import YOLOE

        def make(w):
            stem = os.path.splitext(os.path.basename(str(w)))[0]
            if YOLOE_SEG and stem.endswith("-seg"):
                m = YOLOE(f"{stem}.yaml").load(w)
                m.model.model[-1].__class__ = SegHeadDetView
                return make_pools_deterministic(m)
            stem = stem[:-4] if stem.endswith("-seg") else stem          # yoloe-v8s-seg -> yoloe-v8s
            return make_pools_deterministic(YOLOE(f"{stem}.yaml").load(w))
        make.__name__ = "YOLOE"
        return make
    from ultralytics import YOLOWorld

    def make(w):
        m = YOLOWorld(w)
        return make_pools_deterministic(attach_world_seg(m, YW_SEG) if YW_SEG else m)
    make.__name__ = "YOLOWorld"
    return make


def prompt_names(names):
    """ultralytics 공식 YOLOE/YOLO-World 평가와 같은 텍스트 프롬프트: 'a/b' 동의어 표기에서 첫 이름만 쓴다
    (models/yolo/yoloe/val.py). 10-04: YOLOE(MobileCLIP)는 전체 문자열을 넣으면 LVIS FP AP가 11.6으로 크게 떨어졌다.
    YOLO-World는 기존 결과와의 일관성을 위해 전체 문자열을 그대로 쓴다(이 함수는 YOLOE 경로에서만 쓴다)."""
    return [n.split("/", 1)[0] for n in names]


def switch_vocab_yoloe(model, names, device):
    """평가 어휘 전환(예: LVIS 1203). 텍스트 임베딩을 새로 만들어 head의 pe로 넣는다."""
    model.model.to("cpu")
    with torch.no_grad():
        pe = model.model.get_text_pe(prompt_names(names))
    model.model.set_classes(list(names), pe)
    model.model.names = list(names)
    model.predictor = None
    model.model.to(device).eval()


@torch.no_grad()
def text_bank_yoloe(model_cls, weights, names, device, batch=256):
    """유사도 bank [N, D]: head가 실제로 쓰는 텍스트 임베딩(RepRTA 뒤, L2 정규화)."""
    m = model_cls(weights)
    feats = []
    for i in range(0, len(names), batch):
        pe = m.model.get_text_pe(prompt_names(names[i:i + batch]))   # [1, n, D]
        feats.append(torch.nn.functional.normalize(pe[0].float(), dim=-1).cpu())
    del m
    return torch.cat(feats).to(device)
