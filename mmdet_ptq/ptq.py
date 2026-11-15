"""11-15 InlierQ 논문 설정 재현: mmdet RetinaNet(RegNetX-3.2GF) / Faster R-CNN(R50) 위의 BRECQ와 InlierQ.
.venv_mmdet 전용. 양자화 모듈은 메인 실험과 같은 구현(quant/fake_quant.py, quant/adaround.py)을 그대로 쓴다.

InlierQ 논문(arXiv 2602.03472) 5절의 설정:
  - calibration: COCO train 64장, 데이터 증강 없음 -> mmdet test 파이프라인(1333x800 resize)으로 train2017 앞 64장.
  - head는 FP16 -> head(RetinaNet bbox_head, Faster R-CNN rpn_head/roi_head)는 양자화하지 않는다(FP32로 평가).
  - 첫 층 INT8 -> backbone stem conv(weight, 입력 activation) 8bit.
  - BRECQ는 공식 구현 -> 공식 BRECQ(main_imagenet.py)의 순차 2단계를 따른다.
      1단계: activation 양자화를 끄고 weight 반올림(AdaRound alpha)만 재구성. backbone은 block 단위, FPN과 stem은
             conv 단위. 손실 lp(p=2), 반올림 정규화 0.01, warmup 0.2, beta 20->2, Adam lr 1e-3, iters_w 20000.
      2단계: weight를 hard로 고정하고 activation step(LSQ)만 재구성. Adam lr 4e-4 + cosine, iters_a 5000, lp(p=2.4).
  - InlierQ는 2단계의 activation scale 최적화 손실만 inlier 위치의 FIM 가중 오차로 바꾼다(논문 식 12, 알고리즘 1).
    1단계는 BRECQ와 같으므로 1단계 결과를 캐시해 두 방법이 공유한다(비교 변수는 2단계 손실 하나).
논문에 없는 값과 정한 것:
  - weight: 채널별 비대칭, scale은 공식 BRECQ의 MSE(L2.4) 탐색. activation: 텐서별 비대칭, 양자화 지점은 양자화된 conv의 입력.
    head 입력(FPN 출력)은 FP다(head가 FP16).
  - BN은 conv에 접는다(공식 BRECQ와 같다).
  - batch 1(이미지 크기가 서로 달라서). 공식 BRECQ는 32.
  - heatmap: RetinaNet은 분류 logit의 sigmoid(anchor x 위치 x 레벨 전체에서 클래스마다 상위 K).
    Faster R-CNN은 RPN proposal(추론 설정 1000개)에 대한 RoI head 클래스 확률(softmax, 배경 제외)에서 클래스마다 상위 K.
  - K = 100(두 모델의 추론 max_per_img). tau = 0.5. GMM은 log h(h>0 위치)에 맞춘다(quant/inlierq.py와 같다).
"""
import os, sys, json, time, argparse, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common import build_runner, make_cfg, COCO                       # noqa: E402
from quant.fake_quant import QuantConv2d                               # noqa: E402
from quant.quant_model import wrap_convs, set_mode                     # noqa: E402
from quant.adaround import (convert_to_adaround, list_adaround_convs, lp_rec_loss, temp_decay,   # noqa: E402
                            free_cpu_mem, AdaRoundQuantConv2d)
from quant.inlierq import gmm2_posterior                               # noqa: E402

W_LR, REG_W, WARMUP, A_LR = 1e-3, 0.01, 0.2, 4e-4


# ------------------------------------------------------------------ 모델 준비
@torch.no_grad()
def _fold(conv, bn):
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight.mul_(s.view(-1, 1, 1, 1))
    b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    conv.bias = nn.Parameter((b - bn.running_mean) * s + bn.bias)


def fold_bn(backbone):
    """stem(conv1/bn1), Bottleneck(conv{i}/bn{i}), downsample(Sequential[conv, bn])을 접고 BN을 Identity로."""
    n = 0
    for mod in list(backbone.modules()):
        for i in ("1", "2", "3"):
            c, b = getattr(mod, "conv" + i, None), getattr(mod, "bn" + i, None)
            if isinstance(c, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                _fold(c, b); setattr(mod, "bn" + i, nn.Identity()); n += 1
        if isinstance(mod, nn.Sequential) and len(mod) == 2 and isinstance(mod[0], nn.Conv2d) \
                and isinstance(mod[1], nn.BatchNorm2d):
            _fold(mod[0], mod[1]); mod[1] = nn.Identity(); n += 1
    assert not any(isinstance(m, nn.BatchNorm2d) for m in backbone.modules()), "접히지 않은 BN이 남음"
    return n


def units_of(model):
    """재구성 단위 (이름, 모듈 경로). 순서 = forward 순서. stem/FPN은 conv 단위, backbone block은 block 단위."""
    u = [("stem", "backbone.conv1", False)]
    for li in range(1, 5):
        for bi in range(len(getattr(model.backbone, f"layer{li}"))):
            u.append((f"layer{li}.{bi}", f"backbone.layer{li}.{bi}", True))
    for k in range(len(model.neck.lateral_convs)):
        u.append((f"fpn.lat{k}", f"neck.lateral_convs.{k}.conv", False))
    for k in range(len(model.neck.fpn_convs)):
        u.append((f"fpn.out{k}", f"neck.fpn_convs.{k}.conv", False))
    return u


def load_calib(cfg, n):
    """train2017 앞 n장을 test 파이프라인으로 전처리해 (inputs[1,3,H,W] CPU, data_samples) 목록으로."""
    from mmdet.registry import DATASETS
    from mmengine.dataset import pseudo_collate
    dcfg = copy.deepcopy(cfg.test_dataloader.dataset)
    dcfg.ann_file = "annotations/instances_train2017.json"
    dcfg.data_prefix = dict(img="train2017/")
    dcfg.test_mode = True
    ds = DATASETS.build(dcfg)
    return ds, [ds[i] for i in range(n)]


def to_batch(model, item, device):
    from mmengine.dataset import pseudo_collate
    d = model.data_preprocessor(pseudo_collate([item]), False)
    return d["inputs"].to(device), d["data_samples"]


# ------------------------------------------------------------------ heatmap 손실(InlierQ 식 7)
def heat_logprob(model, feats, data_samples):
    """클래스별 heatmap log 값 [N, C] (N = 위치/후보). RetinaNet: log sigmoid(분류 logit). Faster R-CNN: log softmax 확률."""
    if hasattr(model, "roi_head"):
        from mmdet.structures.bbox import bbox2roi
        with torch.no_grad():
            props = model.rpn_head.predict([f.detach() for f in feats], data_samples, rescale=False)
        rois = bbox2roi([p.bboxes for p in props])
        cls = model.roi_head._bbox_forward(feats, rois)["cls_score"]   # [N, C+1], 배경이 마지막
        return F.log_softmax(cls.float(), dim=1)[:, :-1]
    cls_scores, _ = model.bbox_head(feats)
    C = model.bbox_head.cls_out_channels
    parts = []
    for s in cls_scores:                                               # [1, A*C, H, W]
        b, ac, h, w = s.shape
        parts.append(s.view(b, ac // C, C, h * w).permute(0, 2, 1, 3).reshape(C, -1))
    return F.logsigmoid(torch.cat(parts, dim=1).float()).t()          # [N, C]


def heat_loss(model, feats, data_samples, K):
    lp = heat_logprob(model, feats, data_samples)
    k = min(K, lp.size(0))
    return -lp.topk(k, dim=0).values.mean()


def iq_weights(fp_model, fp_mod, calib_dev, K, tau, label):
    """InlierQ 가중치(이미지별 [1, C, H, W] fp16 CPU, inlier 밖 0). quant/inlierq.InlierQContext.weights와 같은 규칙."""
    gs, hs, c, state = [], [], None, {}

    def hook(m, i, o):
        z = o.detach().float().requires_grad_(True)
        state["z"] = z
        return z + 0.0
    h = fp_mod.register_forward_hook(hook)
    try:
        for x, ds in calib_dev():
            state.clear()
            with torch.enable_grad():
                feats = fp_model.extract_feat(x)
                L = heat_loss(fp_model, feats, ds, K)
                g, = torch.autograd.grad(L, state["z"], allow_unused=True)
            if g is None:
                g = torch.zeros_like(state["z"])
            if c is None:
                c = 1.0 / float(g.pow(2).mean().sqrt().clamp(min=1e-30))
            g = g * c
            gs.append(g.half().cpu())
            hs.append(g.abs().sum(1).flatten().cpu())
    finally:
        h.remove()
    hv = torch.cat(hs)
    nz = hv > 0
    if int(nz.sum()) < 2:
        print(f"    [inlierq] {label} gradient 0 -> BRECQ 손실", flush=True)
        return None
    mask = torch.zeros_like(nz)
    post, (mu, sd, pi) = gmm2_posterior(torch.log(hv[nz]))
    mask[nz] = post >= tau
    out, off, tot, cnt = [], 0, 0.0, 0
    for g in gs:
        hw = g.shape[2] * g.shape[3]
        mk = mask[off:off + hw].view(1, 1, g.shape[2], g.shape[3]); off += hw
        w = g.float() * mk                                      # 제곱근 형태로 둔다(제곱은 손실에서)
        tot += float(w.pow(2).sum()); cnt += int(mk.sum()) * g.shape[1]
        out.append(w)
    norm = (tot / max(cnt, 1)) ** 0.5                          # 11-15: g^2를 fp16으로 두면 inf가 나서 g/rms로 저장
    print(f"    [inlierq] {label} inlier {float(mask.float().mean())*100:.1f}% (h=0 {float((~nz).float().mean())*100:.1f}%, "
          f"GMM mu {mu[0]:.2f}/{mu[1]:.2f}, pi {pi[0]:.2f}/{pi[1]:.2f})", flush=True)
    return [(w / max(norm, 1e-30)).half() for w in out]


def iq_loss(pred, target, r):
    """r = sqrt(FIM 대각) (inlier 밖 0). 손실 = inlier 위치 평균 sum_c r_c^2 (pred - target)_c^2."""
    d = ((pred - target) * r).pow(2)
    return d.sum() / (r != 0).any(1).sum().clamp(min=1)


# ------------------------------------------------------------------ 재구성
def capture(model, mod, calib_dev, what):
    buf = []

    def hook(m, i, o):
        buf.append((o if what == "out" else i[0]).detach().half().cpu())
    h = mod.register_forward_hook(hook)
    with torch.no_grad():
        for x, _ in calib_dev():
            model.extract_feat(x)
    h.remove()
    return buf


def recon(q_model, fp_model, units, calib_dev, device, stage, iters, method="brecq", K=100, tau=0.5, log=print):
    for ti, (label, path, is_block) in enumerate(units):
        qm, fm = q_model.get_submodule(path), fp_model.get_submodule(path)
        convs = list_adaround_convs(qm) if is_block else [qm]
        tgt = capture(fp_model, fm, calib_dev, "out")
        inp = capture(q_model, qm, calib_dev, "in")
        n = len(tgt)
        wbuf = None
        if stage == 2:
            for c in convs:
                c.soft = False
                c.use_lsq = c.a_obs.bits < 16
            params = [c.lsq_act.delta for c in convs if c.use_lsq]
            if not params:
                continue
            if method == "inlierq":
                wbuf = iq_weights(fp_model, fm, calib_dev, K, tau, label)
            opt = torch.optim.Adam(params, lr=A_LR)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=0.0)
        else:
            for c in convs:
                c.soft = True
            params = [c.alpha for c in convs]
            opt, sch = torch.optim.Adam(params, lr=W_LR), None
        t0 = time.time()
        for it in range(iters):
            j = random.randrange(n)
            x = inp[j].to(device).float()
            y = tgt[j].to(device).float()
            opt.zero_grad()
            out = qm(x)
            if stage == 2:
                loss = iq_loss(out, y, wbuf[j].to(device).float()) if wbuf is not None else lp_rec_loss(out, y, p=2.4)
            else:
                loss = lp_rec_loss(out, y)
                if it >= WARMUP * iters:
                    beta = temp_decay(it, iters, WARMUP)
                    loss = loss + REG_W * sum(c.reg_loss(beta) for c in convs)
            loss.backward()
            opt.step()
            if sch is not None:
                sch.step()
        for c in convs:
            c.soft = False
        log(f"  [stage{stage}] {ti+1}/{len(units)} {label} ({len(convs)} conv) {time.time()-t0:.0f}s loss {float(loss):.4g}")
        del tgt, inp, wbuf
        free_cpu_mem()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--w-bits", type=int, default=4)
    ap.add_argument("--a-bits", type=int, default=4)
    ap.add_argument("--method", choices=["brecq", "inlierq", "naive"], default="brecq")
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--iters-w", type=int, default=20000)
    ap.add_argument("--iters-a", type=int, default=5000)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--stage1-cache", default="")
    ap.add_argument("--eval-cap", type=int, default=0, help="0이면 val2017 전체")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda:0"
    os.makedirs(args.work_dir, exist_ok=True)
    print("[args]", vars(args), flush=True)

    runner = build_runner(args.model, args.work_dir)
    fp_model = runner.model.to(device).eval()
    q_model = copy.deepcopy(fp_model)
    for p in fp_model.parameters():
        p.requires_grad_(False)
    print(f"[bn] 접은 BN: fp {fold_bn(fp_model.backbone)}, q {fold_bn(q_model.backbone)}", flush=True)

    _, calib_items = load_calib(runner.cfg, args.calib)
    calib = [to_batch(fp_model, it, "cpu") for it in calib_items]

    def calib_dev():
        for x, ds in calib:
            yield x.to(device), ds

    nq = wrap_convs(q_model.backbone, args.w_bits, args.a_bits) + wrap_convs(q_model.neck, args.w_bits, args.a_bits)
    stem = q_model.backbone.conv1
    stem.w_bits, stem.a_obs.bits = 8, 8
    q_model.to(device).eval()
    # activation 초기 범위: 공식 BRECQ/QDrop처럼 MSE(L2.4) 탐색
    for m in q_model.modules():
        if isinstance(m, QuantConv2d):
            m.a_obs.method = "mse"
    set_mode(q_model, calibrating=True, quantized=False)
    with torch.no_grad():
        for x, _ in calib_dev():
            q_model.extract_feat(x)
    for m in q_model.modules():
        if isinstance(m, QuantConv2d):
            m.a_obs.freeze(); m.freeze_weight_quant()
    set_mode(q_model, calibrating=False, quantized=True)
    convert_to_adaround(q_model, channelwise_smult=False)
    convs = list_adaround_convs(q_model)
    print(f"[quant] conv {nq}개 양자화(backbone+FPN, head FP), stem W8A8, W{args.w_bits}A{args.a_bits}", flush=True)
    units = units_of(q_model)
    t0 = time.time()
    if args.method != "naive":
        if args.stage1_cache and os.path.exists(args.stage1_cache):
            sd = torch.load(args.stage1_cache, map_location=device, weights_only=True)
            for i, c in enumerate(convs):
                c.alpha.data.copy_(sd[f"{i}.alpha"])
                c.soft = False
            print(f"[stage1] 캐시 사용 {args.stage1_cache}", flush=True)
        else:
            for c in convs:
                c.act_quant_enabled = False
            recon(q_model, fp_model, units, calib_dev, device, 1, args.iters_w)
            for c in convs:
                c.act_quant_enabled = True
            if args.stage1_cache:
                torch.save({f"{i}.alpha": c.alpha.detach() for i, c in enumerate(convs)}, args.stage1_cache)
        for c in convs:
            c.act_quant_enabled = True
        recon(q_model, fp_model, units, calib_dev, device, 2, args.iters_a, args.method, args.K, args.tau)
    else:
        for c in convs:
            c.soft = False
    calib_s = time.time() - t0
    print(f"[build] {args.method} 완료 {calib_s:.0f}s", flush=True)

    runner.model = q_model                               # mmengine Runner.model은 일반 속성(테스트 루프가 이걸 쓴다)
    if args.eval_cap:
        # 시험용: val2017 앞 eval_cap장만. CocoMetric은 annotation 파일의 모든 이미지로 채점하므로, 그 이미지만 담은
        # annotation 파일을 만들어 데이터셋과 채점기에 같이 준다(indices만 자르면 나머지 이미지가 미검출로 채점된다).
        full = json.load(open(COCO + "annotations/instances_val2017.json"))
        keep = {im["id"] for im in full["images"][:args.eval_cap]}
        sub = dict(full, images=[im for im in full["images"] if im["id"] in keep],
                   annotations=[a for a in full["annotations"] if a["image_id"] in keep])
        sub_path = os.path.abspath(os.path.join(args.work_dir, f"val_cap{args.eval_cap}.json"))
        json.dump(sub, open(sub_path, "w"))
        runner._test_dataloader["dataset"]["ann_file"] = sub_path
        runner._test_evaluator["ann_file"] = sub_path
    m = runner.test()
    res = {k: round(float(v) * 100, 1) for k, v in m.items() if "bbox" in k}
    print("[result]", args.model, f"W{args.w_bits}A{args.a_bits}", args.method, json.dumps(res), flush=True)
    json.dump(dict(args=vars(args), result=res, calib_s=calib_s),
              open(os.path.join(args.work_dir, f"result_{args.model}_w{args.w_bits}a{args.a_bits}_{args.method}.json"), "w"))


if __name__ == "__main__":
    main()
