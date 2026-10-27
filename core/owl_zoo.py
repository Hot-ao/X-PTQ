"""OWLv2(ViT 기반 OVOD)를 이 저장소의 양자화 파이프라인에 올리기 위한 래퍼 (10-17).

YOLO 계열과 같은 코드(QuantConv2d, calibrate, AdaRound/BRECQ, 보호 목록)를 쓰려고 두 가지를 맞춘다.
  1) Linear -> 1x1 Conv2d(LinearConv). 토큰 [.., C]를 [1, C, M, 1]로 바꿔 흘린다. 수학적으로 같은 연산이고
     activation은 per-tensor, weight는 출력 채널별이라 양자화 단위도 Linear와 같다.
  2) 모델을 블록 리스트(OwlDet.model)로 펼친다: [stem(패치 conv + 위치 임베딩 + pre-LN), ViT 층 12개, head].
     BRECQ(quant/brecq.py)가 YOLO의 DetectionModel.model처럼 블록 단위로 재구성하고, 마지막 블록(head)은 conv 단위로 한다.
텍스트 쪽(text encoder)은 YOLO-World의 텍스트 상수처럼 FP로 미리 계산해 두고 양자화하지 않는다(어휘 고정 배포).
head 출력은 [logits(Q) | box(4) | objectness(1)]를 마지막 축으로 이어 붙인 텐서 하나다.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# 10-18: self-training(ST)만 한 체크포인트. -ensemble/-finetuned는 LVIS base로 fine-tuning돼 LVIS가 held-out 어휘가 아니다
# (scenic owl_vit README: ST LVIS AP 26.5 / ST+FT 41.4 / ensemble 43.9, 뒤 둘은 ft-lvisbase).
OWL_NAME = "google/owlv2-base-patch16"
OWL_IMGSZ = 960
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
PROMPT = "a photo of a {}"


class LinearConv(nn.Module):
    """nn.Linear와 같은 함수를 1x1 Conv2d로. wrap_convs가 안의 conv를 QuantConv2d로 바꾼다."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.conv = nn.Conv2d(lin.in_features, lin.out_features, 1, bias=lin.bias is not None)
        with torch.no_grad():
            self.conv.weight.copy_(lin.weight[:, :, None, None])
            if lin.bias is not None:
                self.conv.bias.copy_(lin.bias)

    def forward(self, x):
        shp = x.shape
        y = self.conv(x.reshape(-1, shp[-1]).t()[None, :, :, None])
        return y[0, :, :, 0].t().reshape(*shp[:-1], -1)


def _linear_to_conv(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LinearConv(child))
        else:
            _linear_to_conv(child)


class Stem(nn.Module):
    def __init__(self, emb, pre_ln):
        super().__init__()
        self.embeddings, self.pre_layernorm = emb, pre_ln

    def forward(self, px):
        return self.pre_layernorm(self.embeddings(px))


class EncLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, h):
        return self.layer(h, None, None)[0]


class Head(nn.Module):
    """post-LN -> class 토큰 곱 -> LN -> class/box/objectness head. queries: 정규화된 텍스트 임베딩 [Q, D]."""

    def __init__(self, det):
        super().__init__()
        self.post_layernorm = det.owlv2.vision_model.post_layernorm
        self.layer_norm = det.layer_norm
        self.class_head, self.box_head, self.objectness_head = det.class_head, det.box_head, det.objectness_head
        self.register_buffer("box_bias", det.box_bias.clone(), persistent=False)
        self.register_buffer("queries", torch.zeros(1, det.config.text_config.hidden_size), persistent=False)
        self.cls_q = nn.Identity()           # attn_quant 자리: region 임베딩(QPoint)·텍스트 상수(ConstQ)
        self.txt_q = nn.Identity()

    def forward(self, h):
        e = self.post_layernorm(h)
        e = e[:, 1:] * e[:, :1]
        e = self.layer_norm(e)
        ch = self.class_head
        ce = ch.dense0(e)
        ce = ce / (torch.linalg.norm(ce, dim=-1, keepdim=True) + 1e-6)
        logits = torch.einsum("bpd,qd->bpq", self.cls_q(ce), self.txt_q(self.queries))
        logits = (logits + ch.logit_shift(e)) * (F.elu(ch.logit_scale(e)) + 1)
        boxes = torch.sigmoid(self.box_head(e) + self.box_bias)
        obj = self.objectness_head(e)
        return torch.cat([logits, boxes, obj], dim=-1)


class OwlDet(nn.Module):
    """OWLv2 검출기를 블록 리스트로 펼친 것. forward(px [B,3,960,960]) -> [B, 3600, Q+5]."""

    def __init__(self, det, text_fn):
        super().__init__()
        vm = det.owlv2.vision_model
        blocks = [Stem(vm.embeddings, vm.pre_layernorm)] + [EncLayer(l) for l in vm.encoder.layers] + [Head(det)]
        self.model = nn.Sequential(*blocks)
        _linear_to_conv(self.model)
        self._text_fn = text_fn
        self.names = []

    @property
    def head(self):
        return self.model[-1]

    def set_classes(self, names):
        self.head.queries = self._text_fn(names).to(self.head.box_bias.device)
        self.names = list(names)

    def forward(self, px):
        return self.model(px)


_TEXT_CACHE = {}
_TEXT_FN = {}


def _text_fn(name, owl, device):
    """텍스트 임베딩 함수(프로세스당 모델 이름마다 하나 -- 빌드마다 text encoder가 GPU에 쌓이지 않게)."""
    if name not in _TEXT_FN:
        from transformers import Owlv2Processor
        tok = Owlv2Processor.from_pretrained(name).tokenizer
        tm, tp = owl.text_model.to(device).eval(), owl.text_projection.to(device).eval()

        @torch.no_grad()
        def fn(names):
            key = (name, tuple(names))
            if key not in _TEXT_CACHE:
                feats = []
                for i in range(0, len(names), 256):
                    t = tok([PROMPT.format(n) for n in names[i:i + 256]], padding="max_length", truncation=True,
                            max_length=16, return_tensors="pt").to(device)
                    f = tp(tm(input_ids=t.input_ids, attention_mask=t.attention_mask)[1])
                    feats.append(f / (torch.linalg.norm(f, dim=-1, keepdim=True) + 1e-6))
                _TEXT_CACHE[key] = torch.cat(feats).float().cpu()
            return _TEXT_CACHE[key]
        _TEXT_FN[name] = fn
    return _TEXT_FN[name]


def build_owl(device, name=None):
    """FP OwlDet. 텍스트 임베딩은 원 모델의 text encoder(FP)로 만들고 어휘별로 캐시한다."""
    from transformers import Owlv2ForObjectDetection
    name = name or OWL_NAME
    det = Owlv2ForObjectDetection.from_pretrained(name).eval()
    m = OwlDet(det, _text_fn(name, det.owlv2, device)).to(device).eval()
    return m


def preprocess(path, imgsz=OWL_IMGSZ):
    """OWLv2 전처리: 아래·오른쪽을 0.5 회색으로 채워 정사각형 -> imgsz로 resize(antialias) -> CLIP 정규화.
    반환: ([1,3,imgsz,imgsz] 텐서, 패딩 뒤 한 변 길이(원 이미지 픽셀))."""
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    h, w = im.shape[:2]
    s = max(h, w)
    pad = np.full((s, s, 3), 0.5, dtype=np.float32)
    pad[:h, :w] = im
    t = torch.from_numpy(pad).permute(2, 0, 1)[None]
    t = F.interpolate(t, size=(imgsz, imgsz), mode="bilinear", align_corners=False, antialias=True)
    mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
    return (t - mean) / std, s


def split_out(out, nq):
    """head 출력 -> (logits [B,P,Q], boxes cxcywh(0~1) [B,P,4], objectness [B,P])."""
    return out[..., :nq], out[..., nq:nq + 4], out[..., nq + 4]
