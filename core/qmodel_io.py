"""양자화 모델 저장/불러오기 (10-27).

모델은 nn.Module을 통째로 torch.save한다(커스텀 모듈 QuantConv2d, AdaRound, 게이트 교환 구조, 이전 scale mig가 그대로
들어간다). 문제는 양자화 지점을 끼우려고 인스턴스에 메서드를 붙인 곳들이다(attn_quant의 forward 교체, owl_quant의
attention forward, fusion_quant의 게이트 교환 forward 등, types.MethodType). pickle은 이걸 "같은 이름의 속성을 찾아라"로
저장해서 불러올 때 AttributeError가 난다. 그래서 저장할 때 이런 메서드를 떼어 내고 (모듈 경로, 속성, 함수의 모듈·이름)을
같이 저장했다가, 불러올 때 다시 붙인다.
함수 안에서 정의한 지역 함수는 다시 찾을 수 없어서 저장을 거부한다. 단 calibrate() 전에만 쓰는 freeze_weight_quant
(fusion_quant의 분기별 weight scale)는 빌드가 끝나면 다시 부르지 않으므로 떼어 내고 버린다.
"""
from __future__ import annotations
import importlib, types
import torch

DROP_OK = {"freeze_weight_quant"}          # 빌드 뒤에는 쓰지 않는 지역 함수 메서드


def _bound_methods(module):
    for path, m in module.named_modules(remove_duplicate=False):
        for k, v in list(vars(m).items()):
            if isinstance(v, types.MethodType) and v.__self__ is m:
                yield path, m, k, v.__func__


def save_module(module, path):
    patched, removed = [], []
    for p, m, k, f in _bound_methods(module):
        removed.append((m, k, m.__dict__.pop(k)))
        if "<locals>" in f.__qualname__:
            if k in DROP_OK:
                continue
            for mm, kk, vv in removed:                       # 되돌리고 거부
                mm.__dict__[kk] = vv
            raise RuntimeError(f"{p}.{k}: 지역 함수({f.__module__}.{f.__qualname__})는 저장할 수 없다. 모듈 수준 함수로 옮길 것")
        patched.append((p, k, f.__module__, f.__qualname__))
    try:
        torch.save({"module": module, "patched": patched}, path)
    finally:
        for m, k, v in removed:
            m.__dict__[k] = v
    return patched


def load_module(path, map_location="cpu"):
    obj = torch.load(path, map_location=map_location, weights_only=False)
    module = obj["module"]
    for p, k, modname, qualname in obj["patched"]:
        f = importlib.import_module(modname)
        for part in qualname.split("."):
            f = getattr(f, part)
        m = module.get_submodule(p) if p else module
        setattr(m, k, types.MethodType(f, m))
    return module
