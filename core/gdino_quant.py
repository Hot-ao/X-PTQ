"""Grounding DINO 양자화 구성 (10-29). owl_quant.py와 같은 규칙을 DETR 계열에 옮긴다.

  - 양자화 범위: Swin 백본, 입력 projection conv, encoder(text enhancer·이미지-텍스트 fusion·deformable 층),
    decoder(self-attn, 텍스트 cross-attn, deformable cross-attn, FFN), query 선택(enc_output, encoder box head),
    box head의 모든 Linear(LinearConv)/conv. 텍스트 인코더(BERT + text_projection)는 FP 상수(gdino_zoo 참고).
  - 첫 레이어: Swin 패치 conv W8A8(입력이 이미지 픽셀). 마지막 레이어: box head 출력층(decoder box MLP와 encoder
    box MLP의 마지막 Linear)은 W8, 입력 A16(YOLO의 head 마지막 1x1, OWLv2 head 출력층과 같은 규칙).
  - FP: LayerNorm, softmax, GELU/ReLU, residual add, sigmoid, deformable 샘플링(grid_sample), 위치 임베딩.
    (Reg-PTQ·InlierQ의 DETR/DETR3D 설정처럼 Linear/conv의 weight와 입력 activation을 양자화한다.)
  - attention(attn_cls와 같은 범위, 10-30): matmul 피연산자 q/k/v는 per-tensor attn_bits(QPoint를 q/k/v Linear 출력에
    forward hook으로), softmax 출력은 종류별 p_bits(아래 P_KINDS). region-텍스트 내적은 두 피연산자 모두 per-tensor
    attn_bits -- 여기 텍스트는 encoder가 이미지마다 갱신한 값이라 OWLv2처럼 상수(ConstQ)가 아니다.
    deformable attention은 value(value_proj 출력)와 샘플링 가중치(softmax 출력)를 양자화하고, 샘플링 위치(offset)는 FP.
    softmax 키 개수가 지점마다 49(Swin 창)~1.7만(텍스트->이미지)이라 OWLv2의 "softmax 16bit" 예외를 종류별로 따진다.
  hook/교체 함수는 모듈 속성(m.out_q 등)만 참조하므로 deepcopy한 모델도 자기 QPoint를 쓴다.
"""
from __future__ import annotations
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant.fake_quant import QuantConv2d
from quant.attn_quant import QPoint
from quant.brecq_units import exec_order

# softmax 출력 양자화 지점 종류 -> 키 개수(COCO 800x1333 입력, 텍스트 약 200토큰 기준)
P_KINDS = {
    "swin": "Swin 창 attention(키 49)",
    "text_self": "text enhancer self-attn(키 = 텍스트 토큰)",
    "bi_v": "fusion 이미지->텍스트(키 = 텍스트 토큰)",
    "bi_t": "fusion 텍스트->이미지(키 = 이미지 토큰 약 1.7만)",
    "deform": "deformable 샘플링 가중치(4 level x 4 point = 16)",
    "dec_self": "decoder self-attn(키 = query 900)",
    "dec_text": "decoder 텍스트 cross-attn(키 = 텍스트 토큰)",
}


def wrap_convs_shared(module: nn.Module, w_bits: int, a_bits: int, memo=None) -> int:
    """하위 nn.Conv2d를 QuantConv2d로(제자리). quant_model.wrap_convs와 같지만 공유 모듈(box MLP)을 한 번만 감싸고,
    이미 감싼 QuantConv2d 안으로 다시 내려가지 않는다. 반환: 새로 감싼 개수."""
    memo = {} if memo is None else memo
    n = 0
    for name, child in list(module.named_children()):
        if id(child) in memo:
            setattr(module, name, memo[id(child)])
        elif isinstance(child, QuantConv2d):
            continue
        elif isinstance(child, nn.Conv2d):
            memo[id(child)] = QuantConv2d(child, w_bits, a_bits)
            setattr(module, name, memo[id(child)])
            n += 1
        else:
            memo[id(child)] = child
            n += wrap_convs_shared(child, w_bits, a_bits, memo)
    return n


def qconv(mod):
    return mod if isinstance(mod, QuantConv2d) else mod.conv


def first_conv(m):
    return qconv(m.det.model.backbone.conv_encoder.model.embeddings.patch_embeddings.projection)


def last_convs(m):
    return [qconv(m.det.model.decoder.bbox_embed[0].layers[-1]), qconv(m.det.model.encoder_output_bbox_embed.layers[-1])]


def _out_hook(mod, inp, out):
    return mod.out_q(out)


def _attach_out_q(mod, bits):
    """mod(LinearConv) 출력에 QPoint. 반환: QPoint."""
    mod.out_q = QPoint(bits)
    mod.register_forward_hook(_out_hook)
    return mod.out_q


def _deform_pre(mod, args):
    """MultiScaleDeformableAttention(value, shapes, shapes_list, start, locations, weights, step): weights 양자화."""
    args = list(args)
    args[5] = mod.p_q(args[5])
    return tuple(args)


def _contrast_pre(mod, args, kwargs):
    """ContrastiveEmbedding(vision_hidden_state, text_hidden_state, text_token_mask): 두 피연산자 양자화.
    HF는 위치 인자(encoder_output_class_embed)와 키워드 인자(decoder class_embed)로 모두 부른다."""
    args = list(args)
    for i, (k, q) in enumerate((("vision_hidden_state", mod.img_q), ("text_hidden_state", mod.txt_q))):
        if i < len(args):
            args[i] = q(args[i])
        else:
            kwargs[k] = q(kwargs[k])
    return tuple(args), kwargs


def _bi_forward(self, vision_features, text_features, vision_attention_mask=None, text_attention_mask=None):
    """GroundingDinoBiMultiHeadAttention.forward와 같은 계산(크기 검사 생략) + softmax 출력 양자화(self.pv_q, self.pt_q).
    원본은 F.dropout(함수)이라 모듈 교체로 끼울 자리가 없다. q/k/v는 Linear 출력 hook이 양자화한다."""
    bsz, tgt_len, _ = vision_features.size()
    vq = self._reshape(self.vision_proj(vision_features) * self.scale, tgt_len, bsz)
    tk = self._reshape(self.text_proj(text_features), -1, bsz)
    vv = self._reshape(self.values_vision_proj(vision_features), -1, bsz)
    tv = self._reshape(self.values_text_proj(text_features), -1, bsz)
    shp = (bsz * self.num_heads, -1, self.head_dim)
    vq, tk, vv, tv = vq.view(*shp), tk.view(*shp), vv.view(*shp), tv.view(*shp)
    src_len = tk.size(1)
    aw = torch.bmm(vq, tk.transpose(1, 2))
    aw = torch.clamp(aw - aw.max(), min=-50000, max=50000)
    awt = aw.transpose(1, 2)
    taw = torch.clamp(awt - torch.max(awt, dim=-1, keepdim=True)[0], min=-50000, max=50000)
    if vision_attention_mask is not None:
        vm = vision_attention_mask[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
        taw.masked_fill_(vm, float("-inf"))
    taw = taw.softmax(dim=-1)
    if text_attention_mask is not None:
        tm = text_attention_mask[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
        aw.masked_fill_(tm, float("-inf"))
    vaw = aw.softmax(dim=-1)
    vo = torch.bmm(self.pv_q(vaw), tv)
    to = torch.bmm(self.pt_q(taw), vv)
    vo = vo.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
    to = to.view(bsz, self.num_heads, src_len, self.head_dim).transpose(1, 2).reshape(bsz, src_len, self.embed_dim)
    return (self.out_vision_proj(vo), vaw), (self.out_text_proj(to), taw)


def _mha_kind(name):
    if "text_enhancer_layer" in name:
        return "text_self"
    return "dec_text" if name.endswith("encoder_attn_text") else "dec_self"


def add_attn_qpoints(m, attn_bits=8, p_bits=16):
    """attention matmul 피연산자, softmax 출력, region-텍스트 내적에 QPoint를 단다. p_bits: int 또는 {종류: 비트}
    (빠진 종류는 16). 반환: {종류: 개수}."""
    pb = p_bits if isinstance(p_bits, dict) else {k: p_bits for k in P_KINDS}
    pb = {k: pb.get(k, 16) for k in P_KINDS}
    cnt = {}
    for name, mod in list(m.det.named_modules()):
        t = type(mod).__name__
        if t == "SwinSelfAttention":
            kind, qkv = "swin", ("query", "key", "value")
            mod.dropout = QPoint(pb[kind])
        elif t == "GroundingDinoMultiheadAttention":
            kind, qkv = _mha_kind(name), ("query", "key", "value")
            mod.dropout = QPoint(pb[kind])
        elif t == "GroundingDinoBiMultiHeadAttention":
            kind, qkv = "bi_v", ("vision_proj", "text_proj", "values_vision_proj", "values_text_proj")
            mod.pv_q, mod.pt_q = QPoint(pb["bi_v"]), QPoint(pb["bi_t"])
            mod.forward = types.MethodType(_bi_forward, mod)
            cnt["bi_t"] = cnt.get("bi_t", 0) + 1
        elif t == "GroundingDinoMultiscaleDeformableAttention":
            kind, qkv = "deform", ("value_proj",)
            mod.attn.p_q = QPoint(pb[kind])
            mod.attn.register_forward_pre_hook(_deform_pre)
        elif t == "GroundingDinoContrastiveEmbedding":
            if not hasattr(mod, "img_q"):                       # decoder 6층이 한 인스턴스를 공유
                mod.img_q, mod.txt_q = QPoint(attn_bits), QPoint(attn_bits)
                mod.register_forward_pre_hook(_contrast_pre, with_kwargs=True)
                cnt["region_text"] = cnt.get("region_text", 0) + 1
            continue
        else:
            continue
        for a in qkv:
            _attach_out_q(getattr(mod, a), attn_bits)
        cnt[kind] = cnt.get(kind, 0) + 1
    return cnt, pb


def apply_protocol(m, w_bits, a_bits, last_abits=16, attn_bits=8, p_bits=16):
    """GdinoDet m(FP)을 양자화 구성으로 바꾼다. calibrate() 전에 호출. attn_bits=0이면 attention·region-텍스트는 FP.
    반환: 설명 문자열 목록."""
    d = m.det.model
    memo = {}
    n = sum(wrap_convs_shared(sub, w_bits, a_bits, memo)
            for sub in (d.backbone, d.input_proj_vision, d.encoder, d.decoder, d.encoder_output_bbox_embed))
    d.enc_output.conv = QuantConv2d(d.enc_output.conv, w_bits, a_bits)
    n += 1
    fc = first_conv(m)
    fc.w_bits = 8; fc.a_obs.bits = 8
    for qc in last_convs(m):
        qc.w_bits = 8; qc.a_obs.bits = last_abits
    desc = [f"양자화 Linear/conv {n}개", "Swin 패치 conv W8A8", f"box head 출력층 W8 A{last_abits}"]
    if not attn_bits:
        return desc + ["attention·region-텍스트 내적 FP"]
    cnt, pb = add_attn_qpoints(m, attn_bits, p_bits)
    return desc + [f"attention q/k/v·region-텍스트 {attn_bits}bit", "softmax " + " ".join(f"{k}{pb[k]}" for k in P_KINDS),
                   "지점 수 " + " ".join(f"{k}{v}" for k, v in cnt.items())]


def quant_layers(m):
    """진단 단위: QuantConv2d 하나씩(첫·마지막 층 제외). [(이름, qc)]. 이름은 m.det 기준 경로."""
    skip = {id(first_conv(m))} | {id(q) for q in last_convs(m)}
    out, seen = [], set()
    for n, mod in m.det.named_modules():
        if isinstance(mod, QuantConv2d) and id(mod) not in skip and id(mod) not in seen:
            seen.add(id(mod))
            out.append((n[:-5] if n.endswith(".conv") else n, mod))
    return out


def recon_units(m, sample, device):
    """BRECQ 재구성 단위(quant/brecq_units.optimize_brecq_units 형식). convert_to_adaround 뒤의 GdinoDet m을 받는다.
    블록(joint): Swin 블록 12, encoder 층 6(text enhancer + fusion + deformable), decoder 층 6.
    conv(layer-wise): 그 밖의 AdaRound conv -- 패치 conv, patch merging, 입력 projection, enc_output, box MLP.
    decoder 쪽(경로가 det.model.decoder로 시작; 공유 box MLP 포함)은 two-stage query 선택 뒤라 FP 입력으로 재구성한다.
    반환: 실행 순서로 정렬한 [(label, path, kind, fp_input)]."""
    from quant.adaround import AdaRoundQuantConv2d
    d = m.det.model
    blocks = [f"det.model.backbone.conv_encoder.model.encoder.layers.{i}.blocks.{j}"
              for i, st in enumerate(d.backbone.conv_encoder.model.encoder.layers) for j in range(len(st.blocks))]
    blocks += [f"det.model.encoder.layers.{i}" for i in range(len(d.encoder.layers))]
    blocks += [f"det.model.decoder.layers.{i}" for i in range(len(d.decoder.layers))]
    inside = {id(c) for p in blocks for c in m.get_submodule(p).modules()}
    convs = [n for n, c in m.named_modules() if isinstance(c, AdaRoundQuantConv2d) and id(c) not in inside]
    kind = {p: "block" for p in blocks} | {p: "conv" for p in convs}
    order = exec_order(m, list(kind), sample, device)
    short = lambda p: p.replace("det.model.", "").replace("backbone.conv_encoder.model.encoder.", "swin.")
    return [(short(p), p, kind[p], p.startswith("det.model.decoder")) for p in order]


def mig_units(m):
    """M 공유 단위(owl_quant.owl_units의 Grounding DINO판): [[소비 경로...]], 경로는 GdinoDet 기준. 생산 쪽에 1/s를
    접어 넣을 수 있는(배포 가능한) 곳만 넣는다.
      - Swin(pre-norm): layernorm_before -> q/k/v, layernorm_after -> fc1(intermediate), GELU -> fc2(output),
        patch merging norm -> reduction. LN 출력은 그 Linear들만 받으므로 1/s를 LN affine에 접는다.
      - encoder(text enhancer, deformable)·decoder는 post-norm(DETR)이라 LN 출력이 residual로도 흘러간다. LN 뒤 Linear에
        이전을 넣으면 residual 값이 바뀌므로 넣지 않는다. fusion 층도 LN 출력에 residual을 더해 같은 이유로 제외.
        ReLU 뒤 fc2만 넣는다(앞 층 requant에 접음, owl_quant의 GELU 뒤 단위와 같은 근거).
      - box MLP(encoder·decoder 공유, reference point head): ReLU 뒤 Linear."""
    d = m.det.model
    sw = "det.model.backbone.conv_encoder.model.encoder.layers"
    units = []
    for i, st in enumerate(d.backbone.conv_encoder.model.encoder.layers):
        for j in range(len(st.blocks)):
            p = f"{sw}.{i}.blocks.{j}"
            units += [[f"{p}.attention.self.query", f"{p}.attention.self.key", f"{p}.attention.self.value"],
                      [f"{p}.intermediate.dense"], [f"{p}.output.dense"]]
        if st.downsample is not None:
            units.append([f"{sw}.{i}.downsample.reduction"])
    for i in range(len(d.encoder.layers)):
        units += [[f"det.model.encoder.layers.{i}.text_enhancer_layer.fc2"],
                  [f"det.model.encoder.layers.{i}.deformable_layer.fc2"]]
    units += [[f"det.model.decoder.layers.{i}.fc2"] for i in range(len(d.decoder.layers))]
    for mlp in ("det.model.encoder_output_bbox_embed", "det.model.decoder.bbox_embed.0",
                "det.model.decoder.reference_points_head"):
        n = len(m.get_submodule(mlp).layers)
        units += [[f"{mlp}.layers.{k}"] for k in range(1, n)]
    return units


XMAX_FLOOR = 3e-3   # 채널 xmax 하한(단위 최댓값 대비). 10-31: 1%는 YOLO-World-S·OWL의 α를 1개씩 바꿔 0.3%로(runs/217_mig_rule)


@torch.no_grad()
def migrate(m, images, device, alphas, rule="tied", verbose=True, stat_images=None, xmax_floor=XMAX_FLOOR):
    """mig_units 단위마다 s를 정해 apply_migration. 규칙과 오차 계산은 owl_quant.migrate와 같다(tied/smooth/awq).
    다른 점: 한 단위의 소비 Linear는 같은 입력 텐서를 받으므로 입력을 단위당 한 번만, CPU에 모은다(Swin 1단계 GELU
    출력은 이미지당 약 100MB라 GPU에 다 들고 있을 수 없다). calibrate() 전에 호출(모델은 FP 모드).
    10-30 수정: ReLU 뒤 단위(encoder/decoder FFN, box MLP)는 채널이 희소해서, α 탐색용 몇 장에서 0이던 채널이 다른
    이미지에서 켜진다(8장 대비 최대 ~1e5배). 그 채널은 s가 ~1e-5로 잡혀 x/s가 폭주했다(brecq+M W4A5 COCO AP 0.01).
    그래서 xmax(s 공식)는 stat_images(calibration 전체)에서 최댓값만 누적해 잡고, 단위 최댓값의 XMAX_FLOOR 아래로
    내려가지 않게 한다. 오차 평가(α 선택)는 그대로 images에서."""
    from quant.channel_graph import SMOOTH_ALPHA, AWQ_GRID
    from quant.migrate import _act_quant_mse, _conv
    from quant.fake_quant import quantize_weight_per_channel
    units = [[(p, qconv(m.get_submodule(p))) for p in u] for u in mig_units(m)]
    xs = [[] for _ in units]
    hs = [u[0][1].register_forward_pre_hook(lambda _m, inp, k=k: xs[k].append(inp[0].detach().cpu()))
          for k, u in enumerate(units)]
    for t in images:
        m(t.to(device))
    for h in hs:
        h.remove()
    stat = [x and torch.stack([v.abs().amax(dim=(0, 2, 3)) for v in x]).amax(0).float().to(device) for x in xs]

    def acc(k):
        def h(_m, inp):
            stat[k] = torch.maximum(stat[k], inp[0].detach().abs().amax(dim=(0, 2, 3)).float())
        return h
    if stat_images is not None:
        hs = [u[0][1].register_forward_pre_hook(acc(k)) for k, u in enumerate(units)]
        for t in stat_images:
            m(t.to(device))
        for h in hs:
            h.remove()

    def err(k, qc, s, wonly):
        W = qc.conv.weight.detach().float()
        Wq = quantize_weight_per_channel(W * s.view(1, -1, 1, 1), qc.w_bits)
        e = r = 0.0
        for x in xs[k]:
            x = x.to(device).float()
            ref = _conv(qc, x, W)
            xi = x / s.view(1, -1, 1, 1)
            if not wonly:
                xi = _act_quant_mse(xi, qc.a_obs.bits)
            e += float((_conv(qc, xi, Wq) - ref).pow(2).sum())
            r += float(ref.pow(2).sum())
        return e / max(r, 1e-20)

    n_on = 0
    for k, u in enumerate(units):
        xmax = stat[k].clamp(min=xmax_floor * float(stat[k].max()))
        wmax = torch.stack([qc.conv.weight.detach().abs().amax(dim=(0, 2, 3)) for _, qc in u]).amax(0).float()
        xmax, wmax = xmax.clamp(min=1e-5), wmax.clamp(min=1e-8)
        one = torch.ones_like(xmax)
        if rule == "smooth":
            s = xmax.pow(SMOOTH_ALPHA) / wmax.pow(1 - SMOOTH_ALPHA)
            best = (None, SMOOTH_ALPHA, s / s.log().mean().exp())
        elif rule == "awq":
            xmean = torch.stack([x.abs().float().mean(dim=(0, 2, 3)) for x in xs[k]]).mean(0).to(device).clamp(min=1e-4)
            best = (float("inf"), None, one)
            for a in AWQ_GRID:
                s = xmean.pow(a); s = s / (s.max() * s.min()).sqrt()
                e = sum(err(k, qc, s, True) for _, qc in u)
                if e < best[0]:
                    best = (e, a, s)
        else:
            best = (sum(err(k, qc, one, False) for _, qc in u), None, one)
            base = best[0]
            for a in alphas:
                s = xmax.pow(a) / wmax.pow(1 - a); s = s / s.log().mean().exp()
                e = sum(err(k, qc, s, False) for _, qc in u)
                if e < best[0]:
                    best = (e, a, s)
        if best[1] is not None and not (rule == "awq" and best[1] == 0.0):
            for _, qc in u:
                qc.apply_migration(best[2].to(qc.conv.weight))
            n_on += 1
        if verbose:
            short = u[0][0].replace("det.model.", "").replace("backbone.conv_encoder.model.encoder.", "swin.")
            print(f"  [{rule}] {short:45s} (소비 {len(u)}) a={'-' if best[1] is None else f'{best[1]:.2f}'}"
                  + (f" err x{best[0] / max(base, 1e-20):.3f}" if rule == "tied" else ""), flush=True)
    del xs
    return n_on, len(units)
