"""Grounding DINO(DETR 계열, early-fusion OVOD)를 이 저장소의 양자화 파이프라인에 올리기 위한 래퍼 (10-29).

owl_zoo.py와 같은 원칙을 따른다.
  1) Linear -> 1x1 Conv2d(owl_zoo.LinearConv). 양자화 단위(activation per-tensor, weight 출력 채널별)가 Linear와 같다.
     HF 구현은 decoder.bbox_embed(6층)와 det.bbox_embed가 같은 MLP 하나를 공유하므로 바꿀 때 memo로 공유를 유지한다.
  2) HF forward는 그대로 쓴다(멀티스케일 마스크, two-stage query 선택, iterative box refinement를 다시 구현하지 않음).
텍스트 쪽(BERT text_backbone + text_projection)은 어휘가 고정되면 미리 계산할 수 있으므로 FP로 둔다(YOLO-World의
CLIP 텍스트, OWLv2 text encoder와 같은 처리). encoder의 text enhancer·fusion 층은 이미지마다 값이 달라 양자화 대상이다.

어휘: 토큰 한도(max_text_len 256) 때문에 클래스 이름을 "a. b. c." 프롬프트 여러 개(chunk)로 나눈다. COCO 80은 1개,
LVIS 1203은 32개. 클래스 점수는 공식 COCO 평가(PostProcessCocoGrounding)처럼 sigmoid(token logits)를 클래스별
positive map(그 클래스 토큰들에 1/n)으로 평균낸 값이다.
forward(px) 출력: [1, 900, C_chunk + 4] = [클래스 점수(0~1) | box cxcywh(0~1)] (현재 chunk 하나).
"""
from __future__ import annotations
import torch
import torch.nn as nn
from PIL import Image

from owl_zoo import LinearConv

GDINO_NAME = "IDEA-Research/grounding-dino-tiny"
HF_CACHE = "/data/taeho/hf_cache/hub"
MAX_TOK = 200            # chunk 하나의 토큰 상한(공식 한도 256 안에서 여유)

_PROC = {}


def processor(name=GDINO_NAME):
    if name not in _PROC:
        from transformers import AutoProcessor
        _PROC[name] = AutoProcessor.from_pretrained(name, cache_dir=HF_CACHE)
    return _PROC[name]


def linear_to_conv(module: nn.Module, memo=None):
    """하위 nn.Linear를 LinearConv로(제자리). 같은 Linear 객체는 같은 LinearConv로 바꿔 공유를 유지한다."""
    memo = {} if memo is None else memo
    for name, child in list(module.named_children()):
        if id(child) in memo:
            setattr(module, name, memo[id(child)])
        elif isinstance(child, nn.Linear):
            memo[id(child)] = LinearConv(child)
            setattr(module, name, memo[id(child)])
        elif not isinstance(child, LinearConv):
            memo[id(child)] = child
            linear_to_conv(child, memo)
    return memo


def chunk_names(names, tok, max_tok=MAX_TOK):
    """클래스 이름 목록 -> 프롬프트 chunk 목록(각 chunk는 이름 목록). 이름 순서를 유지한다."""
    out, cur = [], []
    for n in names:
        if cur and len(tok(". ".join(cur + [n]) + ".").input_ids) > max_tok:
            out.append(cur)
            cur = []
        cur.append(n)
    return out + [cur]


def positive_map(names, tok, length=256):
    """프롬프트 "a. b. c." -> (토큰 입력, [C, length] 클래스별 토큰 평균 가중치)."""
    text = ". ".join(names) + "."
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
    offs = enc.pop("offset_mapping")[0].tolist()
    pos = torch.zeros(len(names), length)
    s = 0
    for i, n in enumerate(names):
        e = s + len(n)
        toks = [t for t, (a, b) in enumerate(offs) if b > a and a >= s and b <= e]
        assert toks, f"토큰을 찾지 못함: {n}"
        pos[i, toks] = 1.0 / len(toks)
        s = e + 2
    return dict(enc), pos


class GdinoDet(nn.Module):
    """HF GroundingDinoForObjectDetection + 고정 어휘. Linear는 모두 LinearConv(텍스트 쪽 제외)."""

    def __init__(self, det, name=GDINO_NAME):
        super().__init__()
        self.det = det
        self.tok = processor(name).tokenizer
        memo = {}
        for sub in (det.model.backbone, det.model.input_proj_vision, det.model.encoder, det.model.decoder,
                    det.model.encoder_output_bbox_embed):
            linear_to_conv(sub, memo)
        det.model.enc_output = LinearConv(det.model.enc_output)
        self.names, self.chunks, self._tok, self._pos = [], [], [], []
        self.chunk = 0

    def set_classes(self, names):
        dev = next(self.parameters()).device
        self.names = list(names)
        self.chunks = chunk_names(self.names, self.tok)
        self._tok, self._pos = [], []
        for c in self.chunks:
            enc, pos = positive_map(c, self.tok)
            self._tok.append({k: v.to(dev) for k, v in enc.items()})
            self._pos.append(pos.to(dev))
        self.chunk = 0

    def forward(self, px):
        out = self.det(pixel_values=px, **self._tok[self.chunk])
        sc = out.logits.sigmoid() @ self._pos[self.chunk].T
        return torch.cat([sc, out.pred_boxes], dim=-1)


def build_gdino(device, name=GDINO_NAME):
    from transformers import AutoModelForZeroShotObjectDetection
    det = AutoModelForZeroShotObjectDetection.from_pretrained(name, cache_dir=HF_CACHE, use_safetensors=True).eval()
    return GdinoDet(det, name).to(device).eval()


def preprocess(path, name=GDINO_NAME):
    """공식 전처리(짧은 변 800, 긴 변 1333 이하, ImageNet 정규화). 반환: ([1,3,H,W], (원 W, 원 H))."""
    img = Image.open(path).convert("RGB")
    return processor(name).image_processor(images=img, return_tensors="pt").pixel_values, img.size
