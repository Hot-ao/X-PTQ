"""OWLv2 양자화 구성 (10-17). YOLO 쪽 확정 프로토콜(scripts/protocol.sh)과 같은 규칙을 ViT에 옮긴다.

  - 양자화 범위: 비전 쪽 모든 Linear(LinearConv) + 패치 conv. 텍스트 인코더는 FP 상수(owl_zoo 참고).
  - 첫 레이어: 패치 conv W8A8(입력이 이미지 픽셀). 마지막 레이어: head의 출력층(class dense0, logit_shift, logit_scale,
    box/objectness dense2)은 W8, 입력 A16(YOLO의 head 마지막 1x1 W8 + 입력 A16과 같은 규칙).
  - attention(attn_cls와 같은 범위): QK^T의 q, k와 v는 per-tensor 8bit(QPoint), softmax 출력은 16bit.
    10-18 runs/182_owl_ablate: 키가 3601개라 softmax 확률 대부분이 1/3601 근처인데 8bit 균일 격자 간격은 1/255라서
    거의 0으로 반올림된다(W8A16 naive에서 softmax만 8bit로 해도 COCO AP 36.7 -> 19.3). YOLO 쪽 "head 마지막 입력 A16"과
    같은 성격의 예외로, 구조 때문에 균일 8bit로 표현할 수 없는 지점만 비트로 푼다. YOLO-World attention은 키가 텍스트
    토큰 수십 개라 해당 없다.
    region-텍스트 내적은 이미지 쪽 per-tensor 8bit, 텍스트 상수는 행별 8bit(ConstQ).
  - FP: LayerNorm, softmax, GELU, residual add, sigmoid, elu.
G(텍스트 게이트 교환)는 OWLv2에 텍스트 게이트가 없어 해당 없음(YOLOE와 같다).
M(공유 제약 이전)은 ViT 구조가 고정이라 채널 계보를 측정하지 않고 공유 단위를 직접 적는다(owl_units):
  LN1 -> q/k/v, LN2 -> fc1, fc1+GELU -> fc2, head LN -> head 입력층 5개, box/objectness GELU -> 다음 층.
  LN 뒤 단위는 1/s를 LN affine에, GELU 뒤 단위는 앞 층 requant에 접어 넣을 수 있어 배포 가능하다.
  attention 출력 -> out_proj는 1/s를 v_proj에 접어야 하는데 그러면 v 양자화 지점이 보는 값이 바뀌므로 넣지 않는다(s=1).
"""
from __future__ import annotations
import types
import torch
import torch.nn as nn

from quant.quant_model import wrap_convs, calibrate, set_conv_wbits
from quant.fake_quant import QuantConv2d, quantize_weight_per_channel
from quant.attn_quant import QPoint, ConstQ
from quant.migrate import _act_quant_mse, _conv

HEAD_OUT = ("class_head.dense0", "class_head.logit_shift", "class_head.logit_scale",
            "box_head.dense2", "objectness_head.dense2")


def qconv(mod):
    """LinearConv/Conv 경로 -> 안의 QuantConv2d."""
    return mod if isinstance(mod, QuantConv2d) else mod.conv


def _attn_forward(self, hidden_states, attention_mask=None, causal_attention_mask=None, output_attentions=False):
    bsz, tgt, d = hidden_states.size()
    hd = self.head_dim
    q = self._aq["q"](self.q_proj(hidden_states) * self.scale)
    k = self._aq["k"](self.k_proj(hidden_states))
    v = self._aq["v"](self.v_proj(hidden_states))
    sh = lambda t: t.view(bsz, tgt, self.num_heads, hd).transpose(1, 2).reshape(bsz * self.num_heads, tgt, hd)
    w = torch.softmax(torch.bmm(sh(q), sh(k).transpose(1, 2)), dim=-1)
    o = torch.bmm(self._aq["p"](w), sh(v))
    o = o.view(bsz, self.num_heads, tgt, hd).transpose(1, 2).reshape(bsz, tgt, d)
    return self.out_proj(o), None


def apply_protocol(m, w_bits, a_bits, attn_bits=8, last_abits=16, p_bits=16, qkv_attn=True):
    """OwlDet m(FP)을 양자화 구성으로 바꾼다. calibrate() 전에 호출. 반환: 설명 문자열 목록."""
    n = wrap_convs(m.model, w_bits, a_bits)
    seq = m.model
    stem = qconv(seq[0].embeddings.patch_embedding)
    stem.w_bits = 8; stem.a_obs.bits = 8
    head = seq[-1]
    for nm in HEAD_OUT:
        qc = qconv(head.get_submodule(nm))
        qc.w_bits = 8; qc.a_obs.bits = last_abits
    if not attn_bits:                                 # 10-18 진단용: attention·region-텍스트 내적을 FP로 둔다
        return [f"양자화 Linear/conv {n}개", "패치 conv W8A8", f"head 출력층 W8 A{last_abits}", "attention FP"]
    for blk in seq[1:-1]:
        a = blk.layer.self_attn
        # p_bits: softmax 출력 비트(None이면 attn_bits). qkv_attn=False면 q/k/v는 FP(진단용).
        a._aq_mods = nn.ModuleDict({k: (QPoint(attn_bits) if qkv_attn else nn.Identity()) for k in ("q", "k", "v")})
        a._aq_mods["p"] = QPoint(p_bits or attn_bits)
        a._aq = dict(a._aq_mods)
        a.forward = types.MethodType(_attn_forward, a)
    head.cls_q = QPoint(attn_bits)
    head.txt_q = ConstQ(attn_bits)
    return [f"양자화 Linear/conv {n}개", "패치 conv W8A8", f"head 출력층 W8 A{last_abits}",
            f"attention q/k/v {attn_bits if qkv_attn else 'FP'}, softmax {p_bits or attn_bits}bit x {len(seq) - 2}층", "region-텍스트 내적 8bit"]


def owl_units(m):
    """M 공유 단위: [[소비 QuantConv2d 경로...], ...]. 경로는 m.model 기준(set_conv_wbits와 같은 형식)."""
    units = []
    for i in range(1, len(m.model) - 1):
        p = f"{i}.layer"
        units += [[f"{p}.self_attn.q_proj", f"{p}.self_attn.k_proj", f"{p}.self_attn.v_proj"],
                  [f"{p}.mlp.fc1"], [f"{p}.mlp.fc2"]]
    h = f"{len(m.model) - 1}"
    units += [[f"{h}.class_head.dense0", f"{h}.class_head.logit_shift", f"{h}.class_head.logit_scale",
               f"{h}.box_head.dense0", f"{h}.objectness_head.dense0"],
              [f"{h}.box_head.dense1"], [f"{h}.box_head.dense2"],
              [f"{h}.objectness_head.dense1"], [f"{h}.objectness_head.dense2"]]
    return units


def _get(m, path):
    idx, _, rest = path.partition(".")
    return qconv(m.model[int(idx)].get_submodule(rest))


@torch.no_grad()
def migrate(m, images, device, alphas, rule="tied", verbose=True):
    """owl_units 단위마다 s를 정해 apply_migration. 규칙은 quant/channel_graph.search_and_apply_tied와 같다:
    tied(우리 M: max|x|^a/max|W|^(1-a), a 탐색, s=1 포함, W·A 양자화 출력 오차), smooth(a=0.5 고정),
    awq(mean|x|^a, weight만 양자화한 오차로 a 탐색). calibrate() 전에 호출(모델은 FP 모드)."""
    from quant.channel_graph import SMOOTH_ALPHA, AWQ_GRID
    units = [[(p, _get(m, p)) for p in u] for u in owl_units(m)]
    allc = [c for u in units for c in u]
    xs = {p: [] for p, _ in allc}
    hs = [qc.register_forward_pre_hook(lambda _m, inp, k=p: xs[k].append(inp[0].detach())) for p, qc in allc]
    for t in images:
        m(t.to(device))
    for h in hs:
        h.remove()

    def err(p, qc, s, wonly):
        W = qc.conv.weight.detach().float()
        Wq = quantize_weight_per_channel(W * s.view(1, -1, 1, 1), qc.w_bits)
        e = r = 0.0
        for x in xs[p]:
            x = x.float()
            ref = _conv(qc, x, W)
            xi = x / s.view(1, -1, 1, 1)
            if not wonly:
                xi = _act_quant_mse(xi, qc.a_obs.bits)
            e += float((_conv(qc, xi, Wq) - ref).pow(2).sum())
            r += float(ref.pow(2).sum())
        return e / max(r, 1e-20)

    n_on = 0
    for u in units:
        xmax = torch.stack([torch.stack([x.abs().amax(dim=(0, 2, 3)) for x in xs[p]]).amax(0) for p, _ in u]).amax(0).float()
        wmax = torch.stack([qc.conv.weight.detach().abs().amax(dim=(0, 2, 3)) for _, qc in u]).amax(0).float()
        xmax, wmax = xmax.clamp(min=1e-5), wmax.clamp(min=1e-8)
        one = torch.ones_like(xmax)
        if rule == "smooth":
            s = xmax.pow(SMOOTH_ALPHA) / wmax.pow(1 - SMOOTH_ALPHA)
            best = (None, SMOOTH_ALPHA, s / s.log().mean().exp())
        elif rule == "awq":
            xmean = torch.stack([torch.stack([x.abs().float().mean(dim=(0, 2, 3)) for x in xs[p]]).mean(0)
                                 for p, _ in u]).amax(0).clamp(min=1e-4)
            best = (float("inf"), None, one)
            for a in AWQ_GRID:
                s = xmean.pow(a); s = s / (s.max() * s.min()).sqrt()
                e = sum(err(p, qc, s, True) for p, qc in u)
                if e < best[0]:
                    best = (e, a, s)
        else:
            best = (sum(err(p, qc, one, False) for p, qc in u), None, one)
            base = best[0]
            for a in alphas:
                s = xmax.pow(a) / wmax.pow(1 - a); s = s / s.log().mean().exp()
                e = sum(err(p, qc, s, False) for p, qc in u)
                if e < best[0]:
                    best = (e, a, s)
        if best[1] is not None and not (rule == "awq" and best[1] == 0.0):
            for _, qc in u:
                qc.apply_migration(best[2].to(qc.conv.weight))
            n_on += 1
        if verbose:
            print(f"  [{rule}] {u[0][0]:40s} (소비 {len(u)}) a={'-' if best[1] is None else f'{best[1]:.2f}'}"
                  + (f" err x{best[0] / max(base, 1e-20):.3f}" if rule == "tied" else ""), flush=True)
    del xs
    return n_on


def linear_groups(m):
    """보호 진단 단위: QuantConv2d 하나씩(패치 conv와 head 출력층(이미 8bit) 제외). [(경로, qc)]."""
    out = []
    skip = {id(qconv(m.model[0].embeddings.patch_embedding))} | {id(qconv(m.model[-1].get_submodule(n))) for n in HEAD_OUT}
    for i, blk in enumerate(m.model):
        for n, mod in blk.named_modules():
            if isinstance(mod, QuantConv2d) and id(mod) not in skip:
                out.append((f"{i}.{n[:-5] if n.endswith('.conv') else n}", mod))
    return out


def set_bits_now(qc, bits):
    """calibrate 뒤 weight 비트를 바꾸고 weight scale을 다시 잡는다(진단용)."""
    qc.w_bits = bits
    qc.freeze_weight_quant()
