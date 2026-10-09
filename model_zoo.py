"""모델 종류별 생성·어휘 전환·텍스트 bank (10-04: YOLOE 지원).

- YOLO-World(yolov8*-world*.pt): ultralytics YOLOWorld 그대로.
- YOLOE(yoloe-*.pt): 공개 가중치는 분할(seg) 모델이라, 같은 이름의 검출 구성(yoloe-v8s.yaml 등)을 만들고
  가중치를 불러온다(head의 mask 분기는 버린다). 텍스트는 MobileCLIP → RepRTA(head.reprta)를 거친 상수(pe)로 head의
  contrastive에만 쓰인다. neck에 텍스트 융합이 없어서 게이트 교환(G)은 적용되지 않는다(gate_commute_all이 아무것도 안 함).
"""
from __future__ import annotations
import os
import torch


def is_yoloe(path: str) -> bool:
    return os.path.basename(str(path)).lower().startswith("yoloe")


def model_factory(path: str):
    """build()의 model_cls 자리에 넣는 생성자. 호출하면 ultralytics Model 래퍼를 돌려준다."""
    if is_yoloe(path):
        from ultralytics import YOLOE

        def make(w):
            stem = os.path.splitext(os.path.basename(str(w)))[0]
            stem = stem[:-4] if stem.endswith("-seg") else stem          # yoloe-v8s-seg -> yoloe-v8s
            return YOLOE(f"{stem}.yaml").load(w)
        make.__name__ = "YOLOE"
        return make
    from ultralytics import YOLOWorld
    return YOLOWorld


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
