"""
채널 계보 추적 + 공유 제약을 지키는 scale 이전 (09-29).

배경: conv마다 독립 s를 쓰는 1차 구현(지금은 제거, promptcal-ptq의 src/quant/migrate.py::search_and_apply)은 배포 제약이 없는 상한이다. 배포에서 x/s는 생산 conv가
SiLU 뒤 requant 단계에서 한 번 만들어 모든 소비자에게 보내므로, 같은 생산 채널을 받는 소비 conv들은
같은 s를 써야 하고, residual add로 합쳐지는 생산 채널들도 같은 s여야 한다.

1) trace_channel_producers: 소비 conv(QuantConv2d)의 입력 채널 c가 어느 생산자(ultralytics Conv 모듈)의
   어느 출력 채널 j에서 왔는지 찾는다. 생산자 P의 출력을 채널별 배율로 나눠 두 번 흘려보낸다
   (run1: r1_j = 1.5 + j/C, run2: r2 = 2). concat/chunk/upsample/maxpool은 양수 채널 배율과 교환 가능하므로
   소비 입력의 변화 d_k = x0 - x_k 는 (1 - 1/r_k) * a (a = 그 입력 중 P 채널 j에서 온 성분)이고,
   d1/d2 = 2 (1 - 1/r1_j) 는 a와 무관하게 j만으로 정해진다 -- residual add(a + b)로 섞여도 성립.
   비선형 경로(attention의 aw = f(x) 등)로 새는 의존은 원소별 d1/d2가 일정하지 않아 걸러진다.
2) tie_groups: 한 소비 채널이 여러 (P, j)에 의존하면(residual add) 그 생산 채널들을 한 그룹으로 묶는다.
   그룹 = 배포에서 s 하나를 공유하는 단위.
3) search_and_apply_tied: 생산자(add로 묶인 생산자들은 한 단위)마다 SmoothQuant 형태
   s_g = max|x_g|^a / max|W_:g|^(1-a) 의 a를 그 단위가 영향을 주는 모든 소비 conv의 상대 출력 오차 합으로
   고르고("이전 없음" 포함), 소비 conv마다 자기 입력 채널의 그룹 s로 apply_migration 한다.
   계보가 없는 입력 채널(stem 이미지 입력 등)은 s=1.

시뮬레이션은 migrate.py와 같은 형태(소비 conv 안에서 x/s, W*s)라 BRECQ 등 이후 경로는 그대로다.
check_deploy_equivalence: 생산자 출력을 그룹 s로 나누고 소비 weight만 W*s로 둔 "배포형" 모델이 FP와 같은
함수인지 확인한다(공유 제약이 하나라도 틀리면 여기서 어긋난다). RNG 미사용.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from .fake_quant import QuantConv2d, quantize_weight_per_channel
from .migrate import _act_quant_mse, _conv


def find_producers(model_module):
    return [(n, m) for n, m in model_module.named_modules()
            if type(m).__name__ == "Conv" and isinstance(getattr(m, "conv", None), QuantConv2d)]


def find_consumers(model_module):
    return [(n, m) for n, m in model_module.named_modules()
            if isinstance(m, QuantConv2d) and m.conv.groups == 1]


@torch.no_grad()
def _consumer_inputs(model_module, consumers, image):
    buf = {}
    hs = [qc.register_forward_pre_hook(lambda _m, inp, k=i: buf.__setitem__(k, inp[0].detach().float()))
          for i, (_, qc) in enumerate(consumers)]
    model_module(image)
    for h in hs:
        h.remove()
    return buf


@torch.no_grad()
def trace_channel_producers(model_module, image, device, tol=1e-3):
    """반환: deps[consumer_idx] = [set((producer_idx, j)), ...] (입력 채널별), producers, consumers."""
    producers, consumers = find_producers(model_module), find_consumers(model_module)
    for _, qc in consumers:
        qc.calibrating, qc.quantized = False, False
    image = image.to(device)
    # 기준 실행에서 모든 QuantConv2d 출력을 저장해 두고, 교란 실행에서는 그 값으로 고정한다 -- 교란이
    # conv를 통과해 퍼지면(예: outlier 채널 하나가 뒤 conv 출력 변화를 지배) 비율 검사를 우연히 통과해
    # 가짜 계보가 생긴다(09-29 실측: 12.attn.proj_conv가 neck 전체에 368채널 그룹을 만듦).
    conv_out = {}
    hs = [qc.register_forward_hook(lambda _m, _i, out, k=i: conv_out.__setitem__(k, out.detach().clone()))
          for i, (_, qc) in enumerate(consumers)]
    base = _consumer_inputs(model_module, consumers, image)
    for h in hs:
        h.remove()
    deps = {ci: [set() for _ in range(base[ci].shape[1])] for ci in base}
    for pi, (_, P) in enumerate(producers):
        C = P.conv.conv.out_channels
        r1 = (1.5 + torch.arange(C, device=device, dtype=torch.float32) / C).view(1, -1, 1, 1)
        outs = []
        for r in (r1, torch.full_like(r1, 2.0)):
            freeze = [qc.register_forward_hook(lambda _m, _i, _o, k=i: conv_out[k].clone())
                      for i, (_, qc) in enumerate(consumers)]
            h = P.register_forward_hook(lambda _m, _i, out, r=r: out / r.to(out.dtype))
            outs.append(_consumer_inputs(model_module, consumers, image))
            h.remove()
            for f in freeze:
                f.remove()
        for ci, x0 in base.items():
            d1, d2 = x0 - outs[0][ci], x0 - outs[1][ci]
            mag = d2.abs().flatten(2).amax(-1)[0]                         # [Cin]
            ref = x0.abs().flatten(2).amax(-1)[0].clamp(min=1e-12)
            hit = (mag > 1e-4 * ref).nonzero().flatten()
            for c in hit.tolist():
                a, b = d1[0, c].flatten(), d2[0, c].flatten()
                sel = b.abs() > 0.05 * b.abs().max()
                q = a[sel] / b[sel]
                if q.numel() == 0 or float(q.std()) > tol * max(1.0, float(q.abs().mean())) and q.numel() > 1:
                    continue                                              # 비선형 누수 -- 계보 아님
                r1_j = 1.0 / (1.0 - float(q.mean()) / 2.0)
                jf = (r1_j - 1.5) * C
                j = round(jf)
                if 0 <= j < C and abs(jf - j) < 0.1:
                    deps[ci][c].add((pi, j))
    return deps, producers, consumers


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def tie_groups(deps):
    """(producer_idx, j) -> 그룹 대표. 한 소비 채널이 여러 생산 채널에 의존하면 묶는다."""
    uf = _UF()
    for chans in deps.values():
        for s in chans:
            s = list(s)
            for node in s:
                uf.find(node)
            for a, b in zip(s, s[1:]):
                uf.union(a, b)
    return uf


SMOOTH_ALPHA = 0.5                                  # rule="smooth": SmoothQuant 기본 이전 강도
AWQ_GRID = tuple(i / 20 for i in range(20))         # rule="awq": AWQ 공식 구현(n_grid=20)의 a 후보, a=0이 s=1


@torch.no_grad()
def search_and_apply_tied(model_module, images, device, alphas, verbose=True, rule="tied"):
    """공유 제약을 지키는 scale 이전. calibrate() 전 호출. 반환: (deps, producers, consumers, s_of, uf).

    rule (10-17, 기준선 비교용 -- 세 규칙 모두 같은 공유 제약·같은 이미지·같은 적용 방식을 쓴다):
      "tied"   우리 M. s = max|x|^a / max|W|^(1-a), 단위마다 a ∈ alphas를 W·A 양자화 출력 오차로 고른다(s=1 포함).
      "smooth" SmoothQuant(Xiao et al., ICML'23). 같은 식에 a = SMOOTH_ALPHA 고정, 탐색 없이 모든 단위에 적용.
      "awq"    AWQ(Lin et al., MLSys'24). s = mean|x|^a(activation 통계만), a ∈ AWQ_GRID를 weight만 양자화한
               출력 오차(activation은 FP)로 고른다. AWQ가 같은 입력을 받는 층들(q/k/v)에 s 하나를 쓰는 것이
               여기의 공유 그룹과 같다. weight clipping 탐색은 기존 weight 양자화기(MSE)가 맡는다."""
    assert rule in ("tied", "smooth", "awq"), rule
    deps, producers, consumers = trace_channel_producers(model_module, images[0], device)
    uf = tie_groups(deps)
    # 소비 conv별: 입력 채널 -> 그룹 대표(None = 계보 없음, s=1 고정)
    cmap = {ci: [uf.find(next(iter(s))) if s else None for s in chans] for ci, chans in deps.items()}
    # 단위(unit) = add로 묶인 생산자들의 집합. 그룹 대표의 producer_idx들로 연결요소를 만든다.
    uf_p = _UF()
    for chans in deps.values():
        for s in chans:
            ps = sorted({p for p, _ in s})
            for a, b in zip(ps, ps[1:]):
                uf_p.union(a, b)
    units = {}
    for ci, gs in cmap.items():
        for g in gs:
            if g is not None:
                units.setdefault(uf_p.find(g[0]), set()).add(g)
    # FP 입력 수집(이 시점 모델은 FP 모드)
    xs = {ci: [] for ci in range(len(consumers))}
    hs = [qc.register_forward_pre_hook(lambda _m, inp, k=i: xs[k].append(inp[0].detach()))
          for i, (_, qc) in enumerate(consumers)]
    for t in images:
        model_module(t.to(device))
    for h in hs:
        h.remove()
    W = {ci: qc.conv.weight.detach().float() for ci, (_, qc) in enumerate(consumers)}
    xmax_c = {ci: torch.stack([x.abs().amax(dim=(0, 2, 3)) for x in xs[ci]]).amax(0).float() for ci in xs}
    wmax_c = {ci: W[ci].abs().amax(dim=(0, 2, 3)) for ci in W}
    # 그룹별 max|x|, max|W| (그룹에 속한 모든 소비 채널에 대해)
    gx, gw = {}, {}
    for ci, gs in cmap.items():
        for c, g in enumerate(gs):
            if g is None:
                continue
            gx[g] = max(gx.get(g, 0.0), float(xmax_c[ci][c]))
            gw[g] = max(gw.get(g, 0.0), float(wmax_c[ci][c]))
    if rule == "awq":
        # 그룹별 mean|x|: 같은 생산 채널을 받는 소비자들은 같은 값을 보고, add로 묶인 그룹은 그중 최댓값을 쓴다.
        gm = {}
        for ci, gs in cmap.items():
            xmean = torch.stack([x.abs().float().mean(dim=(0, 2, 3)) for x in xs[ci]]).mean(0)
            for c, g in enumerate(gs):
                if g is not None:
                    gm[g] = max(gm.get(g, 0.0), float(xmean[c]))
    s_of = {}                                                             # 그룹 -> 확정된 s

    def s_vec(ci, override):
        v = torch.ones(len(cmap[ci]), device=device)
        for c, g in enumerate(cmap[ci]):
            if g is not None:
                v[c] = override.get(g, s_of.get(g, 1.0))
        return v

    def cons_err(ci, s):
        qc = consumers[ci][1]
        Wq = quantize_weight_per_channel(W[ci] * s.view(1, -1, 1, 1), qc.w_bits)
        e = r = 0.0
        for x in xs[ci]:
            x = x.float()
            ref = _conv(qc, x, W[ci])
            e += float((_conv(qc, _act_quant_mse(x / s.view(1, -1, 1, 1), qc.a_obs.bits), Wq) - ref).pow(2).sum())
            r += float(ref.pow(2).sum())
        return e / max(r, 1e-20)

    def cons_err_wonly(ci, s):
        # AWQ 목적함수: weight만 양자화(Q_w(W·s)), 입력 x/s는 FP 그대로.
        qc = consumers[ci][1]
        Wq = quantize_weight_per_channel(W[ci] * s.view(1, -1, 1, 1), qc.w_bits)
        e = r = 0.0
        for x in xs[ci]:
            x = x.float()
            ref = _conv(qc, x, W[ci])
            e += float((_conv(qc, x / s.view(1, -1, 1, 1), Wq) - ref).pow(2).sum())
            r += float(ref.pow(2).sum())
        return e / max(r, 1e-20)

    for u, groups in units.items():
        groups = sorted(groups)
        affected = sorted({ci for ci, gs in cmap.items() if any(g in groups for g in gs)})
        gxt = torch.tensor([gx[g] for g in groups]).clamp(min=1e-5)
        gwt = torch.tensor([gw[g] for g in groups]).clamp(min=1e-8)
        if rule != "tied":
            if rule == "smooth":
                s = gxt.pow(SMOOTH_ALPHA) / gwt.pow(1.0 - SMOOTH_ALPHA)
                s = s / s.log().mean().exp()
                best = (None, SMOOTH_ALPHA, {g: float(v) for g, v in zip(groups, s)})
            else:
                gmt = torch.tensor([gm[g] for g in groups]).clamp(min=1e-4)
                best = (float("inf"), None, None)
                for a in AWQ_GRID:
                    s = gmt.pow(a)
                    s = s / (s.max() * s.min()).sqrt()
                    ov = {g: float(v) for g, v in zip(groups, s)}
                    e = sum(cons_err_wonly(ci, s_vec(ci, ov)) for ci in affected)
                    if e < best[0]:
                        best = (e, a, ov)
            s_of.update(best[2])
            if verbose:
                names = sorted({producers[g[0]][0] for g in groups})
                print(f"  [{rule}] {','.join(names)[:60]:60s} ({len(groups)}그룹, 소비 {len(affected)}) "
                      f"a={best[1]:.2f}", flush=True)
            continue
        best = (sum(cons_err(ci, s_vec(ci, {})) for ci in affected), None, None)
        base_err = best[0]
        for a in alphas:
            s = gxt.pow(a) / gwt.pow(1.0 - a)
            s = s / s.log().mean().exp()
            ov = {g: float(v) for g, v in zip(groups, s)}
            e = sum(cons_err(ci, s_vec(ci, ov)) for ci in affected)
            if e < best[0]:
                best = (e, a, ov)
        if best[2] is not None:
            s_of.update(best[2])
        if verbose:
            names = sorted({producers[g[0]][0] for g in groups})
            print(f"  [tied] {','.join(names)[:60]:60s} ({len(groups)}그룹, 소비 {len(affected)}) "
                  f"a={'-' if best[1] is None else f'{best[1]:.2f}':>5} err x{best[0] / max(base_err, 1e-20):.3f}",
                  flush=True)
    for ci, (_, qc) in enumerate(consumers):
        s = s_vec(ci, {})
        if not torch.allclose(s, torch.ones_like(s)):
            qc.apply_migration(s)
    del xs
    return deps, producers, consumers, s_of, uf


@torch.no_grad()
def check_deploy_equivalence(fp_module, q_module, producers, s_of, uf, image, device):
    """배포형: 생산자 출력을 그룹 s로 나누고, 소비 conv는 입력 나눗셈 없이 W*s만(apply_migration이 이미 곱해둠).
    공유 제약이 맞으면 FP와 같은 함수. 반환: 최종 출력 max|diff| / max|ref|."""
    hs = []
    for pi, (_, P) in enumerate(producers):
        C = P.conv.conv.out_channels
        s = torch.ones(C, device=device)
        for j in range(C):
            g = uf.find((pi, j)) if (pi, j) in uf.p else None
            if g is not None and g in s_of:
                s[j] = s_of[g]
        if not torch.allclose(s, torch.ones_like(s)):
            hs.append(P.register_forward_hook(lambda _m, _i, out, s=s.view(1, -1, 1, 1): out / s.to(out.dtype)))
    # attention(MaxSigmoidAttnBlock)은 입력 x를 conv 없이 FP로 직접 쓴다(embed = x). 배포에서는 그 앞에서
    # 채널별 s로 역양자화하므로 여기서도 x*s로 되돌리고, 그 x를 받는 proj_conv는 입력 나눗셈(mig)을 유지한다.
    keep = set()
    for n, m in q_module.named_modules():
        if type(m).__name__ == "MaxSigmoidAttnBlock":
            pc = m.proj_conv.conv
            if pc.mig is not None:
                keep.add(id(pc))
                hs.append(m.register_forward_pre_hook(
                    lambda _m, args, s=pc.mig: (args[0] * s.to(args[0].dtype),) + tuple(args[1:])))
    saved = {}
    for n, m in q_module.named_modules():
        if isinstance(m, QuantConv2d) and m.mig is not None and id(m) not in keep:
            saved[n] = m.mig
            m.mig = None                                                  # 입력 나눗셈 끔(생산자가 대신)
    try:
        a = fp_module(image.to(device))[0]
        b = q_module(image.to(device))[0]
    finally:
        for h in hs:
            h.remove()
        for n, m in q_module.named_modules():
            if n in saved:
                m.mig = saved[n]
    return float((a - b).abs().max() / a.abs().max())
