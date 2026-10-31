"""OWLv2 PTQ 실험 진입점 (10-17). run_comparison.py의 OWLv2판.

평가는 이미지마다 FP 모델과 조건별 모델을 함께 흘려 한 번에 잰다(FP logits를 저장하지 않는다).
  - COCO AP: COCO val2017 5000장, COCO 80 어휘(calibration과 같은 어휘). 이미지당 상위 100개(패치, 클래스) 쌍.
  - LVIS AP: LVIS minival과 겹치는 4809장, LVIS 1203 어휘(held-out). Fixed AP(run_comparison.run_lvis_eval과 같은
    클래스당 10000개 상한), 이미지당 상위 1000개 쌍.
  - LVIS_flip: FP가 확신(sigmoid > 0.25)하는 패치에서 LVIS top-1 클래스가 바뀐 비율(%). LVIS_lost: 그중 FP top-1이
    그 이미지의 GT 클래스에 있던 패치 수.
NMS는 쓰지 않는다(OWLv2 기본 후처리와 같음). TF32는 끈다(Linear를 1x1 conv로 바꿨기 때문에 cuDNN TF32가 켜져 있으면
원 모델과 logits가 최대 2.2 차이 났다. 끄면 0.002).
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))   # 10-27: 보조 모듈(model_zoo 등)은 core/
from owl_zoo import build_owl, preprocess, split_out
from owl_quant import apply_protocol, migrate, owl_units, linear_groups, set_bits_now
from quant.quant_model import calibrate, set_conv_wbits
from run_comparison import load_names, run_lvis_eval, PROTECT_BUDGET, save_qmodel, update_qmodel_meta
from diag_w4_sensitivity import select_protected
from model_zoo import prompt_names


def _dets(logits, boxes, side, topk):
    """(P,Q) logits, (P,4) cxcywh(0~1) -> 상위 topk (score, cls, xywh 원 이미지 픽셀)."""
    sc = torch.sigmoid(logits).flatten()
    v, i = sc.topk(min(topk, sc.numel()))
    p, c = i // logits.shape[1], i % logits.shape[1]
    b = boxes[p] * side
    xywh = torch.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 2], b[:, 3]], -1)
    return v.cpu().numpy(), c.cpu().numpy(), xywh.cpu().numpy()


@torch.no_grad()
def evaluate(models, names, paths, ids, cat_ids, device, topk, gt_cls=None, conf=0.25, swap=False):
    """models: {이름: OwlDet}, 'FP32'가 반드시 있어야 한다. 반환: {이름: dict(results, flip, lost, n_conf)}.
    swap(10-19, AP 성분 분리 진단): 양자화 모델마다 '<이름>|FP박스'(양자화 클래스 점수 + FP box: 분류 손상만)와
    '<이름>|FP클래스'(FP 클래스 점수 + 양자화 box: 위치 손상만) 결과를 더하고, FP가 확신하는 패치에서
    box 오차(cxcywh L1 평균, 정규화 좌표)를 box_l1로 돌려준다."""
    for m in models.values():
        m.set_classes(names)
    nq = len(names)
    out = {k: dict(results=[], flip=0, lost=0, box_l1=0.0) for k in models}
    if swap:
        out.update({f"{k}|{v}": dict(results=[], flip=0, lost=0, box_l1=0.0) for k in models if k != "FP32"
                    for v in ("FP박스", "FP클래스")})
    n_conf = 0
    for path, iid in zip(paths, ids):
        px, side = preprocess(path)
        px = px.to(device)
        top_fp = conf_mask = None
        add = lambda k, s, c, b: out[k]["results"].extend(
            {"image_id": int(iid), "category_id": int(cat_ids[ci]), "bbox": [float(x) for x in bb], "score": float(si)}
            for si, ci, bb in zip(s, c, b))
        for k, m in models.items():
            lg, bx, _ = split_out(m(px), nq)
            lg, bx = lg[0].float(), bx[0].float()
            add(k, *_dets(lg, bx, side, topk))
            if k == "FP32":
                lg_fp, bx_fp = lg, bx
            elif swap:
                add(f"{k}|FP박스", *_dets(lg, bx_fp, side, topk))
                add(f"{k}|FP클래스", *_dets(lg_fp, bx, side, topk))
            top = lg.argmax(-1)
            if k == "FP32":
                conf_mask = torch.sigmoid(lg).amax(-1) > conf
                top_fp = top
                n_conf += int(conf_mask.sum())
            else:
                ch = conf_mask & (top != top_fp)
                out[k]["flip"] += int(ch.sum())
                if conf_mask.any():
                    out[k]["box_l1"] += float((bx[conf_mask] - bx_fp[conf_mask]).abs().mean(-1).sum())
                if gt_cls is not None and iid in gt_cls:
                    g = torch.tensor(sorted(gt_cls[iid]), device=device)
                    out[k]["lost"] += int((ch & torch.isin(top_fp, g)).sum())
    for k in out:
        out[k]["n_conf"] = n_conf
        out[k]["flip_pct"] = 100.0 * out[k]["flip"] / max(n_conf, 1)
        out[k]["box_l1"] = out[k]["box_l1"] / max(n_conf, 1)
    return out


PROTECT_CACHE = "configs/protect_cache"
MIG_XMAX_FLOOR = 0.0     # +M max|x| 하한(--xmax-floor, 10-31 0.3% 확인용). 0 = 기존
def protect_file():
    import owl_zoo
    return os.path.join(PROTECT_CACHE, f"{owl_zoo.OWL_NAME.split('/')[-1]}_n200_c256_img960_la16_aqattn_cls_p16.json")


def build_q(cond, w_bits, a_bits, calib, device, prot, attn_bits=8, p_bits=16, qkv_attn=True):
    """조건 이름(run_comparison.py와 같은 형식: <기반>+<접미사>) -> 양자화 OwlDet.
    기반: naive | brecq | qdrop. 접미사: P(보호), M(우리 이전), S(SmoothQuant), W(AWQ). G는 해당 구조가 없다."""
    from run_comparison import MIG_ALPHAS, MIG_IMAGES, RECON_ITERS, QDROP_PROB
    from quant.adaround import convert_to_adaround
    from quant.brecq import optimize_brecq
    base, _, suf = cond.partition("+")
    flags = set(suf)
    assert base in ("naive", "brecq", "qdrop") and flags <= set("PMSW") and len(flags & set("MSW")) <= 1, cond
    m = build_owl(device)
    desc = apply_protocol(m, w_bits, a_bits, attn_bits=attn_bits, p_bits=p_bits, qkv_attn=qkv_attn)
    if "P" in flags and prot:
        set_conv_wbits(m, prot, 8)
    rule = next(({"M": "tied", "S": "smooth", "W": "awq"}[f] for f in "MSW" if f in flags), None)
    print(f"[build] {cond}: {', '.join(desc)}" + (f", W8 보호 {len(prot)}개" if "P" in flags else "")
          + (f", 이전={rule}" if rule else ""), flush=True)
    if rule:
        n = migrate(m, calib[:MIG_IMAGES], device, MIG_ALPHAS, rule=rule, verbose=False, xmax_floor=MIG_XMAX_FLOOR)
        print(f"  이전 적용 단위 {n}/{len(owl_units(m))}", flush=True)
    calibrate(m, calib, device=device, act_observer="mse")
    if base in ("brecq", "qdrop"):
        fpref = build_owl(device)
        convert_to_adaround(m, channelwise_smult=False)
        optimize_brecq(m, fpref, calib, device, iters=RECON_ITERS, verbose=False, learn_act_scale=True,
                       two_stage=False, batch=1, neck_layerwise=False,
                       qdrop_prob=QDROP_PROB if base == "qdrop" else 0.0)
        del fpref
        torch.cuda.empty_cache()
    return m.eval()


def qmodel_path(d, args, cond):
    """--save-model/--load-model 파일 이름(run_comparison.qmodel_path와 같은 형식, 모델 이름은 HF 체크포인트)."""
    import owl_zoo
    return os.path.join(d, f"{owl_zoo.OWL_NAME.split('/')[-1]}_w{args.w_bits}a{args.a_bits}_s{args.seed}_{cond.replace('+', '-')}.pt")


def load_q(path, device):
    """저장한 블록(OwlDet.model)을 새 FP OwlDet에 끼운다. OwlDet 자체는 텍스트 함수(클로저)를 들고 있어 저장하지 않는다."""
    m = build_owl(device)
    from qmodel_io import load_module
    m.model = load_module(path, map_location=device).to(device)   # 일반 텐서 속성은 .to()로 안 옮겨진다
    j = path[:-3] + ".json"
    meta = json.load(open(j)) if os.path.exists(j) else {}
    print(f"  [load] {path} (commit {meta.get('commit', '?')[:8]}{', dirty' if meta.get('dirty') else ''}, "
          f"저장 당시 COCO AP {meta.get('coco_ap', '?')})", flush=True)
    return m.eval()


def weight_mib(m):
    # AdaRound로 바뀐 층(BRECQ/QDrop)까지 세는 run_comparison의 계산을 그대로 쓴다.
    from run_comparison import quantized_weight_mib
    return quantized_weight_mib(m)


@torch.no_grad()
def diag_protect(fp, calib, coco_names, device, n_eval=200, conf=0.25):
    """누수 없는 층 단위 진단(diag_w4_sensitivity.rank_convs_leakfree의 OWLv2판). train2017 앞 n_eval장(calibration
    부분집합), COCO 어휘만. 기준 모델은 naive W8A8(확정 프로토콜). 층 하나씩 weight만 W4로 내려 COCO flip 증가를 잰다."""
    from quant.fake_quant import QuantConv2d
    rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    try:
        fp.set_classes(coco_names)
        nq = len(coco_names)
        evals = calib[:n_eval]
        refs = []
        for t in evals:
            lg = split_out(fp(t.to(device)), nq)[0][0]
            refs.append(((torch.sigmoid(lg).amax(-1) > conf).cpu(), lg.argmax(-1).cpu()))
        q = build_owl(device)
        apply_protocol(q, 8, 8)
        calibrate(q, calib, device=device, act_observer="mse")
        q.set_classes(coco_names)

        def flip():
            f = n = 0
            for t, (mk, top) in zip(evals, refs):
                lg = split_out(q(t.to(device)), nq)[0][0].cpu()
                f += int((mk & (lg.argmax(-1) != top)).sum()); n += int(mk.sum())
            return 100.0 * f / max(n, 1)

        base = flip()
        print(f"[diag] 기준(naive W8A8) COCO flip {base:.2f}%", flush=True)
        rows = []
        for name, qc in linear_groups(q):
            set_bits_now(qc, 4)
            d = flip() - base
            set_bits_now(qc, 8)
            rows.append(dict(name=name, mparam=qc.conv.weight.numel() / 1e6, coco=d, emb=0.0, box=0.0, cls=0.0))
            print(f"  {name:40s} {rows[-1]['mparam']:.3f}M  COCO flip +{d:.2f}", flush=True)
        del q
        return rows
    finally:
        torch.set_rng_state(rng[0]); torch.cuda.set_rng_state_all(rng[1])


def coco_ap(gt_ann, results, img_ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import contextlib, io
    gt = COCO(gt_ann) if isinstance(gt_ann, str) else gt_ann
    if not results:
        return 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(results)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = list(img_ids)
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--lvis-ann",
                    default="/data/taeho/lvis_datasets/labels_dl/extracted/lvis/annotations/lvis_v1_minival.json")
    ap.add_argument("--conditions", default="", help="쉼표로 구분(비우면 FP만 잰다)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--eval-cap", type=int, default=0, help=">0이면 평가 이미지를 앞에서 이만큼만(빠른 점검용)")
    ap.add_argument("--skip-coco", action="store_true")
    ap.add_argument("--w-bits", type=int, default=8)
    ap.add_argument("--a-bits", type=int, default=8)
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--diag", action="store_true", help="보호 진단 캐시만 만들고 끝낸다")
    ap.add_argument("--attn-bits", type=int, default=8, help="0이면 attention·region-텍스트 내적을 FP로(진단용)")
    ap.add_argument("--p-bits", type=int, default=16, help="softmax 출력 비트(확정 16, owl_quant.apply_protocol 참고)")
    ap.add_argument("--xmax-floor", type=float, default=0.0, help="+M max|x| 하한(단위 최댓값 대비). 0 = 기존")
    ap.add_argument("--no-qkv-quant", action="store_true", help="q/k/v 양자화 지점을 끈다(진단용)")
    ap.add_argument("--swap-eval", action="store_true", help="COCO AP를 분류·위치 성분으로 나눠 잰다(진단용)")
    ap.add_argument("--skip-lvis", action="store_true")
    ap.add_argument("--odinw", action="store_true", help="ODinW-13(GLIP 공식 split)도 평가한다(core/odinw.py)")
    ap.add_argument("--odinw-root", default="/data/taeho/odinw")
    ap.add_argument("--save-model", default="", help="빌드한 조건 모델을 이 폴더에 저장(+메타 .json)")
    ap.add_argument("--load-model", default="", help="빌드 대신 이 폴더의 저장 모델을 불러온다")
    ap.add_argument("--owl-name", default=None, help="HF 체크포인트(기본 owl_zoo.OWL_NAME = ST)")
    args = ap.parse_args()
    global MIG_XMAX_FLOOR
    MIG_XMAX_FLOOR = args.xmax_floor
    print(f"[args] {vars(args)}", flush=True)
    device = f"cuda:{args.device}"
    import owl_zoo
    if args.owl_name:
        owl_zoo.OWL_NAME = args.owl_name
    print(f"[owl] {owl_zoo.OWL_NAME}", flush=True)
    torch.backends.cudnn.allow_tf32 = False
    if os.environ.get("PTQ_MEM_FRACTION"):       # 10-18 점검용: 큰 GPU에서 작은 GPU 메모리 한도를 흉내 낸다
        torch.cuda.set_per_process_memory_fraction(float(os.environ["PTQ_MEM_FRACTION"]), int(args.device))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    _p = torch.cuda.get_device_properties(int(args.device))
    print(f"[gpu] --device {args.device} -> {_p.name} uuid={_p.uuid}", flush=True)

    coco_names = load_names("coco")
    lvis_names = load_names("lvis")
    from pycocotools.coco import COCO
    from lvis import LVIS
    coco_gt = COCO(os.path.join(args.coco_root, "annotations", "instances_val2017.json"))
    coco_cat_ids = sorted(coco_gt.getCatIds())
    lvis_gt = LVIS(args.lvis_ann)
    lvis_ids_all = set(lvis_gt.get_img_ids())
    probe = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))
    if args.eval_cap > 0:
        probe = probe[:args.eval_cap]
    coco_ids = [int(os.path.basename(p)[:-4]) for p in probe]
    lvis_pairs = [(p, i) for p, i in zip(probe, coco_ids) if i in lvis_ids_all]
    lvis_gt_cls = {}
    for a in lvis_gt.dataset["annotations"]:
        lvis_gt_cls.setdefault(a["image_id"], set()).add(a["category_id"] - 1)
    print(f"[data] COCO {len(probe)}장, LVIS {len(lvis_pairs)}장", flush=True)

    models = {"FP32": build_owl(device)}
    conditions = [c for c in args.conditions.split(",") if c]
    if conditions or args.diag:
        calib_paths = sorted(glob.glob(os.path.join(args.coco_root, "train2017", "*.jpg")))[:args.calib]
        calib = [preprocess(p)[0] for p in calib_paths]
        print(f"[data] calib {len(calib)}장 (train2017)", flush=True)
    if args.diag:
        rows = diag_protect(models["FP32"], calib, coco_names, device)
        os.makedirs(PROTECT_CACHE, exist_ok=True)
        json.dump(rows, open(protect_file(), "w"), indent=1)
        print(f"[diag] -> {protect_file()}")
        print("[diag] 보호 목록(예산 1.5%):", select_protected(rows, PROTECT_BUDGET, "coco")[0])
        return
    prot = []
    if any("P" in c.partition("+")[2] for c in conditions) and args.w_bits < 8:
        prot = select_protected(json.load(open(protect_file())), PROTECT_BUDGET, "coco")[0]
        print(f"[cond+P] 예산 {PROTECT_BUDGET:.1%} -> {prot}", flush=True)
    import random as _r
    snap = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state(), _r.getstate())
    sizes = {}
    for cond in conditions:
        torch.set_rng_state(snap[0]); torch.cuda.set_rng_state_all(snap[1])
        np.random.set_state(snap[2]); _r.setstate(snap[3])
        t0 = time.time()
        if args.load_model:
            models[cond] = load_q(qmodel_path(args.load_model, args, cond), device)
            sizes[cond] = weight_mib(models[cond])
            continue
        models[cond] = build_q(cond, args.w_bits, args.a_bits, calib, device, prot, args.attn_bits, args.p_bits or None,
                                 not args.no_qkv_quant)
        sizes[cond] = weight_mib(models[cond])
        print(f"  {cond} 빌드 {time.time() - t0:.0f}s  (이론적 크기 {sizes[cond]:.2f} MiB)", flush=True)
        if args.save_model:
            save_qmodel(models[cond], qmodel_path(args.save_model, args, cond),
                        dict(cond=cond, args=vars(args), calib_time=time.time() - t0, model_mib=sizes[cond],
                             protect=list(prot) if "P" in cond.partition("+")[2] else []))

    res = {}
    t0 = time.time()
    if not args.skip_coco:
        ev = evaluate(models, coco_names, probe, coco_ids, coco_cat_ids, device, topk=100, swap=args.swap_eval)
        for k, v in ev.items():
            res.setdefault(k, {})["coco"] = 100 * coco_ap(coco_gt, v["results"], coco_ids)
            for d in (args.save_model, args.load_model):
                if d and k in conditions and not args.eval_cap:
                    update_qmodel_meta(qmodel_path(d, args, k), coco_ap=round(res[k]["coco"], 4))
            if args.swap_eval and "|" not in k:
                res[k]["coco_flip"] = v["flip_pct"]; res[k]["box_l1"] = v["box_l1"]
        print(f"[eval] COCO {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    if not args.skip_lvis:
        ev = evaluate(models, prompt_names(lvis_names), [p for p, _ in lvis_pairs], [i for _, i in lvis_pairs],
                      list(range(1, len(lvis_names) + 1)), device, topk=1000, gt_cls=lvis_gt_cls)
        for k, v in ev.items():
            r = run_lvis_eval(lvis_gt, v["results"], [i for _, i in lvis_pairs])
            res.setdefault(k, {}).update(lvis=100 * r["AP"], apr=100 * r["APr"], apc=100 * r["APc"], apf=100 * r["APf"],
                                         lvis_flip=v["flip_pct"], lvis_lost=v["lost"], n_conf=v["n_conf"])
        print(f"[eval] LVIS {time.time() - t0:.0f}s", flush=True)
    if args.odinw:
        from odinw import load_odinw13, coco_ap as odinw_ap, report as odinw_report
        t0 = time.time()
        odinw_sets = load_odinw13(args.odinw_root, args.eval_cap)
        odinw_res = {k: {} for k in models}
        for ds in odinw_sets:
            ev = evaluate(models, ds["names"], ds["img_paths"], ds["img_ids"], ds["cat_ids"], device, topk=100)
            for k, v in ev.items():
                odinw_res[k][ds["name"]] = 100 * odinw_ap(ds["gt"], v["results"], ds["img_ids"])
            print(f"  [ODinW] {ds['name']} ({len(ds['img_ids'])}장) " +
                  " ".join(f"{k}={odinw_res[k][ds['name']]:.2f}" for k in odinw_res), flush=True)
        odinw_report(odinw_res, odinw_sets)
        for k, v in odinw_res.items():
            res.setdefault(k, {})["odinw"] = v["mean"]
        print(f"[eval] ODinW {time.time() - t0:.0f}s", flush=True)
    print("\n결과 | " + " | ".join(["COCO AP", "LVIS AP", "APr", "APc", "APf", "LVIS_flip", "LVIS_lost"]))
    for k, v in res.items():
        print(f"{k} | " + " | ".join(f"{v.get(c, float('nan')):.2f}" for c in ("coco", "lvis", "apr", "apc", "apf", "lvis_flip"))
              + f" | {v.get('lvis_lost', 0)}")
    if args.swap_eval:
        print("\n성분 분리(COCO) | 원래 AP | box를 FP로(분류 손상만) | 클래스를 FP로(위치 손상만) | COCO flip | box L1")
        for k in [k for k in res if "|" not in k and k != "FP32"]:
            print(f"{k} | {res[k]['coco']:.2f} | {res[k + '|FP박스']['coco']:.2f} | {res[k + '|FP클래스']['coco']:.2f} | "
                  f"{res[k]['coco_flip']:.2f}% | {res[k]['box_l1']:.4f}")
    for k, v in sizes.items():
        res[k]["size_mib"] = v
    print(f"[mem] 최대 GPU 메모리 {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB (예약 {torch.cuda.max_memory_reserved(device) / 2**30:.1f} GiB)")
    print(f"[json] {json.dumps(res)}")


if __name__ == "__main__":
    main()
