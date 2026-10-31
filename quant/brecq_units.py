"""
BRECQ를 HF 계열 모델(Grounding DINO 등)에 쓰기 위한 경로 단위 재구성 (10-30).

quant/brecq.py의 optimize_brecq는 DetectionModel.model / OwlDet.model 같은 블록 리스트를 전제로 하고, 블록 입력을
위치 인자로만 잡으며 출력이 텐서가 아니면 건너뛴다. HF 층은 키워드 인자(마스크, 위치 임베딩, reference point,
spatial shape 등)를 받고 tuple을 돌려주므로 그대로 쓸 수 없다. 여기서는 같은 알고리즘(블록 joint 재구성, 누적 오차
반영, lp 손실, warmup 뒤 rounding 정규화, LSQ activation step 공동 학습, QDrop 선택)을 유지하고 다음만 바꾼다.
  - 재구성 단위를 모듈 경로 목록(units)으로 받는다. kind="block"은 그 안의 AdaRound conv 전체를 joint로,
    kind="conv"는 AdaRound conv 하나(layer-wise)다.
  - 입력은 forward hook(with_kwargs)으로 위치 인자와 키워드 인자를 함께 저장한다. float 텐서만 fp16으로 내리고
    bool/정수 텐서(마스크, spatial shape, level index)는 dtype을 유지한다.
  - 출력은 단위가 돌려주는 주 출력만 쓴다: 텐서면 그대로, tuple이면 첫 원소(그것이 tuple이면 그 안의 텐서들 --
    Grounding DINO encoder 층의 (이미지, 텍스트)). attention 가중치 같은 부가 출력은 손실에 넣지 않는다.
  - 토큰 출력 [B, N, C]는 채널(마지막 축) sum, 나머지 mean(lp_rec_loss의 채널 sum 규칙을 토큰 배치에 맞춘 것).
  - fp_input=True인 단위는 입력도 FP 모델에서 잡는다(대칭 재구성). DETR류의 two-stage query 선택 뒤에서는 양자화
    모델이 FP와 다른 query를 고를 수 있어, 양자화 경로 입력과 FP 목표가 같은 query를 가리킨다는 보장이 없다.
  - 이미지마다 해상도가 달라 batch=1로만 돈다(공유 모듈은 한 forward에 여러 번 불리면 호출마다 표본 하나).
YOLO/OWLv2 결과는 기존 optimize_brecq로 낸 것이고 이 파일은 그 경로를 건드리지 않는다.
"""
from __future__ import annotations
import torch

from .adaround import (AdaRoundQuantConv2d, list_adaround_convs, h_alpha, free_cpu_mem, lp_rec_loss, temp_decay,
                       DEFAULT_LR, DEFAULT_REG_WEIGHT, DEFAULT_WARMUP, DEFAULT_ACT_LR_BRECQ, DEFAULT_ACT_LR_QDROP)
from .brecq import LSQ_MAX_BITS


def _cpu(x):
    if torch.is_tensor(x):
        x = x.detach()
        return (x.half() if x.is_floating_point() else x).cpu()
    if isinstance(x, (list, tuple)):
        return type(x)(_cpu(v) for v in x)
    if isinstance(x, dict):
        return {k: _cpu(v) for k, v in x.items()}
    return x


def _dev(x, device):
    if torch.is_tensor(x):
        return x.to(device).float() if x.is_floating_point() else x.to(device)
    if isinstance(x, (list, tuple)):
        return type(x)(_dev(v, device) for v in x)
    if isinstance(x, dict):
        return {k: _dev(v, device) for k, v in x.items()}
    return x


def _mix(q, f, prob):
    """QDrop input_prob: float 텐서만 원소별로 섞고 나머지는 양자화 경로 값을 쓴다."""
    if torch.is_tensor(q):
        if torch.is_tensor(f) and q.is_floating_point() and q.shape == f.shape:
            return torch.where(torch.rand_like(q) < prob, q, f)
        return q
    if isinstance(q, (list, tuple)):
        return type(q)(_mix(a, b, prob) for a, b in zip(q, f))
    if isinstance(q, dict):
        return {k: _mix(v, f[k], prob) for k, v in q.items()}
    return q


def main_outputs(o):
    """단위 출력 -> 손실에 쓸 텐서 목록."""
    if torch.is_tensor(o):
        return [o]
    o = o[0]
    if torch.is_tensor(o):
        return [o]
    return [t for t in o if torch.is_tensor(t)]


def rec_loss(pred, tgt):
    if pred.dim() == 3:                                  # 토큰 [B, N, C]
        return (pred - tgt).abs().pow(2).sum(-1).mean()
    return lp_rec_loss(pred, tgt)


def _get(root, path):
    return root.get_submodule(path) if path else root


def _capture(root, mod, calib, device, want_in, want_out):
    """root에 calib을 흘리며 mod의 (args, kwargs)와 주 출력을 호출마다 저장."""
    ins, outs = [], []

    def hook(m, args, kwargs, out):
        if want_in:
            ins.append(_cpu((args, kwargs)))
        if want_out:
            outs.append([_cpu(t) for t in main_outputs(out)])
    h = mod.register_forward_hook(hook, with_kwargs=True)
    with torch.no_grad():
        for t in calib:
            root(t.to(device))
    h.remove()
    return ins, outs


def exec_order(root, paths, sample, device):
    """paths를 forward 첫 호출 순서로 정렬(누적 오차 반영 재구성은 실행 순서대로 해야 한다)."""
    first, hs = {}, []

    def mark(p):
        def hook(m, i, o):                                   # None을 돌려줘야 출력을 바꾸지 않는다
            first.setdefault(p, len(first))
        return hook
    for p in paths:
        hs.append(_get(root, p).register_forward_hook(mark(p)))
    with torch.no_grad():
        root(sample.to(device))
    for h in hs:
        h.remove()
    missing = [p for p in paths if p not in first]
    assert not missing, f"forward에서 불리지 않는 단위: {missing[:3]}"
    return sorted(paths, key=first.get)


@torch.no_grad()
def _rel_err(qm, in_buf, out_buf, device, k=4):
    """앞 k개 표본에서 주 출력의 상대 제곱 오차 sum((q-fp)^2)/sum(fp^2)."""
    num = den = 0.0
    for j in range(min(k, len(in_buf))):
        args, kwargs = _dev(in_buf[j], device)
        for p, t in zip(main_outputs(qm(*args, **kwargs)), out_buf[j]):
            t = _dev(t, device)
            num += float((p - t).pow(2).sum()); den += float(t.pow(2).sum())
    return num / max(den, 1e-12)


def optimize_brecq_units(quant_root, fp_root, units, calib, device, iters=2000, lr=DEFAULT_LR,
                         reg_weight=DEFAULT_REG_WEIGHT, warmup=DEFAULT_WARMUP, learn_act_scale=True, act_lr=None,
                         qdrop_prob=0.0, verbose=True):
    """units: [(label, path, kind('block'|'conv'), fp_input)] 실행 순서. quant_root/fp_root는 같은 구조의 양자화/FP 모델
    (calib 텐서 하나를 받는 forward). quant_root는 convert_to_adaround를 거친 상태. 반환: 단위별 nearest 대비 flip(%)."""
    if act_lr is None:
        act_lr = DEFAULT_ACT_LR_QDROP if qdrop_prob > 0 else DEFAULT_ACT_LR_BRECQ
    calib = list(calib)
    flips = []
    for ti, (label, path, kind, fp_input) in enumerate(units):
        qm, fm = _get(quant_root, path), _get(fp_root, path)
        convs = list_adaround_convs(qm) if kind == "block" else [qm]
        assert convs and all(isinstance(c, AdaRoundQuantConv2d) for c in convs), label
        need_fp_in = fp_input or qdrop_prob > 0
        fp_in, out_buf = _capture(fp_root, fm, calib, device, need_fp_in, True)
        in_buf = fp_in if fp_input else _capture(quant_root, qm, calib, device, True, False)[0]
        n = min(len(in_buf), len(out_buf))
        assert n > 0, f"{label}: 입력/출력 미포착"

        if verbose:
            for c in convs:
                c.soft = False
            e0 = _rel_err(qm, in_buf, out_buf, device)
        for c in convs:
            c.soft, c.ste, c.qdrop_prob = True, True, qdrop_prob
            if learn_act_scale:
                c.use_lsq = c.a_obs.bits < LSQ_MAX_BITS
        opt = torch.optim.Adam([c.alpha for c in convs], lr=lr)
        lsq = [c.lsq_act.delta for c in convs if c.use_lsq] if learn_act_scale else []
        opt_a = torch.optim.Adam(lsq, lr=act_lr) if lsq else None
        sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=iters, eta_min=0.0) if lsq else None
        for it in range(iters):
            j = int(torch.randint(0, n, (1,)))
            args, kwargs = _dev(in_buf[j], device)
            if qdrop_prob > 0 and not fp_input:
                fa, fk = _dev(fp_in[j], device)
                args, kwargs = _mix(args, fa, qdrop_prob), _mix(kwargs, fk, qdrop_prob)
            opt.zero_grad()
            if opt_a is not None:
                opt_a.zero_grad()
            loss = sum(rec_loss(p, _dev(t, device)) for p, t in zip(main_outputs(qm(*args, **kwargs)), out_buf[j]))
            if it >= warmup * iters:
                beta = temp_decay(it, iters, warmup)
                loss = loss + reg_weight * sum(c.reg_loss(beta) for c in convs)
            loss.backward()
            opt.step()
            if opt_a is not None:
                opt_a.step()
                sched_a.step()

        for c in convs:
            c.soft, c.ste, c.qdrop_prob = False, False, 0.0
        fl = sum(c.flip_rate() for c in convs) / len(convs)
        flips.append(fl)
        if verbose:
            e1 = _rel_err(qm, in_buf, out_buf, device)
            hc = sum(float(((h_alpha(c.alpha) < 0.05) | (h_alpha(c.alpha) > 0.95)).float().mean()) for c in convs)
            print(f"  [{ti+1}/{len(units)}] {label} ({len(convs)} conv, 표본 {n}{', FP 입력' if fp_input else ''}) "
                  f"h->0/1 {100*hc/len(convs):.0f}%, flip {fl:.3f}%, 상대오차 {e0:.4f} -> {e1:.4f}", flush=True)
        del in_buf, out_buf, fp_in
        free_cpu_mem()
    name = "qdrop" if qdrop_prob > 0 else "brecq"
    print(f"[{name}-units] 재구성 완료 {len(units)}단위, nearest 대비 평균 flip {sum(flips)/len(flips):.3f}% "
          f"(max {max(flips):.3f}%)", flush=True)
    return flips
