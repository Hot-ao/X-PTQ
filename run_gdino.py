"""Grounding DINO-T PTQ 실험 진입점 (10-30). run_owl.py의 Grounding DINO판.

양자화 구성은 core/gdino_quant.apply_protocol(첫 conv W8A8, box head 출력층 W8 A16, attention q/k/v·region-텍스트 8bit,
softmax 종류별 비트 -- 10-30 진단에서 종류 하나만 8bit로 내려도 COCO AP가 최대 0.5 떨어져 모두 16으로 둔다).
재구성(brecq/qdrop)은 quant/brecq_units.optimize_brecq_units, 단위는 gdino_quant.recon_units(41개, decoder 쪽 11개는
two-stage query 선택 뒤라 FP 입력).

평가는 이미지마다 FP 모델과 조건별 모델을 함께 흘려 한 번에 잰다.
  - COCO AP: COCO val2017 5000장, COCO 80 어휘(chunk 1개, calibration과 같은 어휘). 이미지당 상위 300개(query, 클래스)
    쌍(공식 PostProcessCocoGrounding과 같은 수).
  - COCO flip / lost: FP가 확신(최고 클래스 점수 > 0.3)하는 query마다 양자화 출력에서 box IoU 최대 query를 짝짓는다.
    짝의 IoU < 0.5면 lost, 짝의 top-1 클래스가 다르면 flip(%). DETR 계열은 query 순서가 FP와 같다는 보장이 없어서
    (two-stage 선택) OWLv2처럼 같은 위치끼리 비교할 수 없다.
  - LVIS AP(--lvis): LVIS minival과 겹치는 이미지, 1203 어휘를 chunk 32개로 나눠 chunk마다 forward. chunk별 상위 300개를
    모아 이미지당 상위 1000개, Fixed AP(run_comparison.run_lvis_eval). 이미지당 forward 32번이라 느리다
    (RTX 4000 Ada에서 FP만 4809장 약 15시간 추정) -- --lvis-cap으로 앞에서 일부만 쓸 수 있다.
"""
from __future__ import annotations
import argparse, copy, glob, json, os, sys, time
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import numpy as np
import torch
from torchvision.ops import box_iou, box_convert

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))
from gdino_zoo import build_gdino, preprocess, GDINO_NAME
from gdino_quant import apply_protocol, recon_units
from quant.quant_model import calibrate
from run_comparison import load_names, run_lvis_eval, update_qmodel_meta
from run_owl import coco_ap, weight_mib


def _dets(sc, bx, side, topk):
    """sc [Q, C] 클래스 점수, bx [Q, 4] cxcywh(0~1) -> 상위 topk (score, cls, xywh 원 이미지 픽셀)."""
    v, i = sc.flatten().topk(min(topk, sc.numel()))
    q, c = i // sc.shape[1], i % sc.shape[1]
    W, H = side
    b = bx[q] * torch.tensor([W, H, W, H], device=bx.device, dtype=bx.dtype)
    xywh = torch.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 2], b[:, 3]], -1)
    return v.cpu().numpy(), c.cpu().numpy(), xywh.cpu().numpy()


def _match(s0, b0, s1, b1, conf):
    """FP(s0, b0)가 확신하는 query -> (확신 수, flip 수, lost 수). 모듈 docstring의 box 짝짓기."""
    keep = s0.amax(-1) > conf
    if not keep.any():
        return 0, 0, 0
    iou = box_iou(box_convert(b0[keep], "cxcywh", "xyxy"), box_convert(b1, "cxcywh", "xyxy"))
    v, j = iou.max(-1)
    same = s1[j].argmax(-1) == s0[keep].argmax(-1)
    return int(keep.sum()), int(((v >= 0.5) & ~same).sum()), int((v < 0.5).sum())


@torch.no_grad()
def evaluate(models, names, paths, ids, cat_ids, device, topk, chunk_topk=300, conf=0.3):
    """models: {이름: GdinoDet}, 'FP32'가 반드시 있어야 한다. 어휘가 chunk 여러 개면 chunk마다 forward해서
    chunk별 상위 chunk_topk를 모은 뒤 이미지당 topk. flip/lost는 chunk마다 잰 것을 더한다.
    반환: {이름: dict(results, flip_pct, lost_pct, n_conf)}."""
    for m in models.values():
        m.set_classes(names)
    fp = models["FP32"]
    offs = np.cumsum([0] + [len(c) for c in fp.chunks])
    out = {k: dict(results=[], flip=0, lost=0) for k in models}
    n_conf, t0 = 0, time.time()
    for n, (path, iid) in enumerate(zip(paths, ids)):
        px, side = preprocess(path)
        px = px.to(device)
        per = {k: ([], [], []) for k in models}
        for ci in range(len(fp.chunks)):
            ref = None
            for k, m in models.items():
                m.chunk = ci
                o = m(px)[0].float()
                sc, bx = o[:, :-4], o[:, -4:]
                s, c, b = _dets(sc, bx, side, chunk_topk if len(fp.chunks) > 1 else topk)
                per[k][0].append(s); per[k][1].append(c + offs[ci]); per[k][2].append(b)
                if k == "FP32":
                    ref = (sc, bx)
                else:
                    nc, f, lo = _match(*ref, sc, bx, conf)
                    out[k]["flip"] += f; out[k]["lost"] += lo
            n_conf += _match(*ref, *ref, conf)[0]
        for k, (s, c, b) in per.items():
            s, c, b = np.concatenate(s), np.concatenate(c), np.concatenate(b)
            top = np.argsort(-s)[:topk]
            out[k]["results"].extend(
                {"image_id": int(iid), "category_id": int(cat_ids[c[j]]), "bbox": [float(x) for x in b[j]],
                 "score": float(s[j])} for j in top)
        if n % 500 == 0:
            print(f"    {n}/{len(paths)} {time.time() - t0:.0f}s", flush=True)
    for k in out:
        out[k]["n_conf"] = n_conf
        out[k]["flip_pct"] = 100.0 * out[k]["flip"] / max(n_conf, 1)
        out[k]["lost_pct"] = 100.0 * out[k]["lost"] / max(n_conf, 1)
    return out


PROTECT_CACHE = "configs/protect_cache"


def protect_file(args, n_eval):
    return os.path.join(PROTECT_CACHE, f"{GDINO_NAME.split('/')[-1]}_n{n_eval}_c{args.calib}_attn{args.attn_bits}"
                                       f"_p{args.p_bits}.json")


@torch.no_grad()
def fp_refs(fp, imgs, device):
    out = []
    for t in imgs:
        o = fp(t.to(device))[0].float()
        out.append((o[:, :-4], o[:, -4:]))
    return out


@torch.no_grad()
def judge(models, fp, imgs, device, conf=0.3, refs=None):
    """imgs(전처리된 텐서) 위에서 FP 대비 (flip %, lost %) -- 모델마다. 어휘는 지금 설정된 것(COCO 80, chunk 1개).
    refs: fp_refs로 미리 계산한 FP 출력(진단처럼 같은 이미지로 여러 번 잴 때)."""
    refs = refs or fp_refs(fp, imgs, device)
    out = []
    for q in models:
        n = f = lo = 0
        for t, ref in zip(imgs, refs):
            o = q(t.to(device))[0].float()
            a, b, c = _match(*ref, o[:, :-4], o[:, -4:], conf)
            n += a; f += b; lo += c
        out.append((100.0 * f / max(n, 1), 100.0 * lo / max(n, 1)))
    return out


@torch.no_grad()
def diag_protect(fp, calib, args, device, n_eval):
    """누수 없는 층 단위 진단(run_owl.diag_protect의 Grounding DINO판). 기준 모델은 naive W8A8(같은 attention 구성),
    calibration 앞 n_eval장, COCO 어휘. 층 하나씩 weight만 W4로 내려 판정 손상 증가를 잰다. DETR 계열은 query 순서가
    FP와 같지 않아 box를 짝지어 재므로, 판정 손상 = flip(짝의 클래스가 바뀜) + lost(짝이 없어짐), 단위 %p.
    반환: [dict(name, mparam, coco, flip, lost)], name은 m.det 기준 경로."""
    from gdino_quant import quant_layers
    from owl_quant import set_bits_now
    rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    try:
        q = copy.deepcopy(fp)
        apply_protocol(q, 8, 8, attn_bits=args.attn_bits, p_bits=args.p_bits)
        calibrate(q, calib, device=device, act_observer="mse")
        evals = calib[:n_eval]
        refs = fp_refs(fp, evals, device)
        f0, l0 = judge([q], fp, evals, device, refs=refs)[0]
        print(f"[diag] 기준(naive W8A8) flip {f0:.2f}% lost {l0:.2f}% ({n_eval}장)", flush=True)
        rows, t0 = [], time.time()
        layers = quant_layers(q)
        for k, (name, qc) in enumerate(layers):
            set_bits_now(qc, 4)
            f, l = judge([q], fp, evals, device, refs=refs)[0]
            set_bits_now(qc, 8)
            rows.append(dict(name=name, mparam=qc.conv.weight.numel() / 1e6, coco=(f - f0) + (l - l0),
                             flip=f - f0, lost=l - l0))
            print(f"  [{k + 1}/{len(layers)}] {name:70s} {rows[-1]['mparam']:.3f}M  +{rows[-1]['coco']:.2f} "
                  f"(flip {f - f0:+.2f}, lost {l - l0:+.2f}) {time.time() - t0:.0f}s", flush=True)
        del q
        return rows
    finally:
        torch.set_rng_state(rng[0]); torch.cuda.set_rng_state_all(rng[1])


def build_q(cond, fp, args, calib, device, prot=()):
    """조건 이름 <기반>[+<접미사>] -> 양자화 GdinoDet. 기반: naive | brecq | qdrop. 접미사: P(보호), M(우리 이전),
    S(SmoothQuant), W(AWQ). G는 해당 구조가 없다(텍스트 게이트 없음)."""
    from run_comparison import RECON_ITERS, QDROP_PROB, MIG_ALPHAS, MIG_IMAGES
    from quant.adaround import convert_to_adaround
    from quant.brecq_units import optimize_brecq_units
    from gdino_quant import migrate, qconv
    base, _, suf = cond.partition("+")
    flags = set(suf)
    assert base in ("naive", "brecq", "qdrop") and flags <= set("PMSW") and len(flags & set("MSW")) <= 1, \
        f"지원하지 않는 조건: {cond}"
    m = copy.deepcopy(fp)
    desc = apply_protocol(m, args.w_bits, args.a_bits, attn_bits=args.attn_bits, p_bits=args.p_bits)
    if "P" in flags:
        for n in prot:
            qconv(m.det.get_submodule(n)).w_bits = 8
    rule = next(({"M": "tied", "S": "smooth", "W": "awq"}[f] for f in "MSW" if f in flags), None)
    print(f"[build] {cond}: {', '.join(desc)}" + (f", W8 보호 {len(prot)}개" if "P" in flags else "")
          + (f", 이전={rule}" if rule else ""), flush=True)
    if rule:
        n_on, n_all = migrate(m, calib[:MIG_IMAGES], device, MIG_ALPHAS, rule=rule, verbose=True,
                              stat_images=calib if args.mig_stat == "all" else None, xmax_floor=args.xmax_floor)
        print(f"  이전 적용 단위 {n_on}/{n_all}", flush=True)
    calibrate(m, calib, device=device, act_observer="mse")
    if base in ("brecq", "qdrop"):
        n = convert_to_adaround(m, channelwise_smult=False)
        units = recon_units(m, calib[0], device)
        print(f"  AdaRound {n}개, 재구성 단위 {len(units)}개(FP 입력 {sum(u[3] for u in units)}), "
              f"iters {args.iters or RECON_ITERS}", flush=True)
        optimize_brecq_units(m, fp, units, calib, device, iters=args.iters or RECON_ITERS, learn_act_scale=True,
                             qdrop_prob=QDROP_PROB if base == "qdrop" else 0.0, verbose=True)
        torch.cuda.empty_cache()
    return m.eval()


def qmodel_path(d, args, cond):
    return os.path.join(d, f"{GDINO_NAME.split('/')[-1]}_w{args.w_bits}a{args.a_bits}_s{args.seed}_{cond.replace('+', '-')}.pt")


def save_q(m, path, meta):
    """m.det(HF 모델 전체, 양자화 모듈 포함)만 저장한다. GdinoDet의 토크나이저와 어휘 텐서는 다시 만든다."""
    import subprocess
    from qmodel_io import save_module
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_module(m.det, path + ".tmp")
    os.replace(path + ".tmp", path)
    root = os.path.dirname(os.path.abspath(__file__))
    git = lambda *a: subprocess.run(["git", "-C", root, *a], capture_output=True, text=True).stdout.strip()
    meta = dict(meta, commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
                saved=time.strftime("%Y-%m-%d %H:%M:%S"), torch=torch.__version__)
    json.dump(meta, open(path[:-3] + ".json", "w"), indent=1, ensure_ascii=False)
    print(f"  [save] {path} ({os.path.getsize(path) / 2**20:.1f} MiB)", flush=True)


def load_q(fp, path, device):
    from qmodel_io import load_module
    m = copy.deepcopy(fp)
    m.det = load_module(path, map_location=device).to(device)
    j = path[:-3] + ".json"
    meta = json.load(open(j)) if os.path.exists(j) else {}
    print(f"  [load] {path} (commit {meta.get('commit', '?')[:8]}, 저장 당시 COCO AP {meta.get('coco_ap', '?')})", flush=True)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--lvis-ann",
                    default="/data/taeho/lvis_datasets/labels_dl/extracted/lvis/annotations/lvis_v1_minival.json")
    ap.add_argument("--conditions", default="", help="쉼표로 구분(비우면 FP만 잰다)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--eval-cap", type=int, default=0, help=">0이면 COCO 평가 이미지를 앞에서 이만큼만")
    ap.add_argument("--w-bits", type=int, default=8)
    ap.add_argument("--a-bits", type=int, default=8)
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--iters", type=int, default=0, help="재구성 iters(0이면 run_comparison.RECON_ITERS)")
    ap.add_argument("--attn-bits", type=int, default=8, help="0이면 attention·region-텍스트 내적을 FP로")
    ap.add_argument("--p-bits", type=int, default=16, help="softmax 출력 비트(모든 종류)")
    ap.add_argument("--skip-coco", action="store_true")
    ap.add_argument("--lvis", action="store_true", help="LVIS minival도 잰다(chunk 32개, 느림)")
    ap.add_argument("--lvis-cap", type=int, default=0, help=">0이면 LVIS 이미지를 앞에서 이만큼만")
    ap.add_argument("--diag", action="store_true", help="P 진단을 돌려 캐시를 만든다(조건이 있으면 이어서 빌드)")
    ap.add_argument("--mig-stat", choices=["8", "all"], default="8", help="+M max|x| 통계: 8장(α 탐색 이미지) | calibration 전체")
    ap.add_argument("--xmax-floor", type=float, default=0.003, help="+M max|x| 하한(단위 최댓값 대비)")
    ap.add_argument("--diag-n", type=int, default=64, help="P 진단 이미지 수(calibration 앞에서)")
    ap.add_argument("--mcheck-n", type=int, default=200, help="M 사전 검사 이미지 수(calibration 다음 train2017)")
    ap.add_argument("--save-model", default="", help="빌드한 조건 모델을 이 폴더에 저장(+메타 .json)")
    ap.add_argument("--load-model", default="", help="빌드 대신 이 폴더의 저장 모델을 불러온다")
    args = ap.parse_args()
    print(f"[args] {vars(args)}", flush=True)
    device = f"cuda:{args.device}"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    _p = torch.cuda.get_device_properties(int(args.device))
    print(f"[gpu] --device {args.device} -> {_p.name} uuid={_p.uuid}", flush=True)

    coco_names = load_names("coco")
    from pycocotools.coco import COCO
    coco_gt = COCO(os.path.join(args.coco_root, "annotations", "instances_val2017.json"))
    coco_cat_ids = sorted(coco_gt.getCatIds())
    probe = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))
    if args.eval_cap > 0:
        probe = probe[:args.eval_cap]
    coco_ids = [int(os.path.basename(p)[:-4]) for p in probe]

    fp = build_gdino(device)
    fp.set_classes(coco_names)                     # calibration·재구성 어휘 = COCO 80(chunk 1개)
    models = {"FP32": fp}
    conditions = [c for c in args.conditions.split(",") if c]
    train = sorted(glob.glob(os.path.join(args.coco_root, "train2017", "*.jpg")))
    if (conditions and not args.load_model) or args.diag:
        calib = [preprocess(p)[0] for p in train[:args.calib]]
        print(f"[data] calib {len(calib)}장 (train2017)", flush=True)
    if args.diag:
        t0 = time.time()
        rows = diag_protect(fp, calib, args, device, args.diag_n)
        os.makedirs(PROTECT_CACHE, exist_ok=True)
        json.dump(rows, open(protect_file(args, args.diag_n), "w"), indent=1)
        from diag_w4_sensitivity import select_protected
        from run_comparison import PROTECT_BUDGET
        sel, tot, total = select_protected(rows, PROTECT_BUDGET, "coco")
        print(f"[diag] -> {protect_file(args, args.diag_n)} ({time.time() - t0:.0f}s)\n"
              f"[diag] 보호 목록(예산 {PROTECT_BUDGET:.1%}, {tot:.2f}/{total:.2f}M): {sel}", flush=True)
        if not conditions:
            return
    prot = []
    if any("P" in c.partition("+")[2] for c in conditions) and args.w_bits < 8:
        from diag_w4_sensitivity import select_protected
        from run_comparison import PROTECT_BUDGET
        prot, tot, total = select_protected(json.load(open(protect_file(args, args.diag_n))), PROTECT_BUDGET, "coco")
        print(f"[cond+P] 예산 {PROTECT_BUDGET:.1%} ({tot:.2f}/{total:.2f}M) -> {prot}", flush=True)
    mcheck = [preprocess(p)[0] for p in train[args.calib:args.calib + args.mcheck_n]] if conditions else []
    import random as _r
    snap = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state(), _r.getstate())
    sizes = {}
    for cond in conditions:
        torch.set_rng_state(snap[0]); torch.cuda.set_rng_state_all(snap[1])
        np.random.set_state(snap[2]); _r.setstate(snap[3])
        t0 = time.time()
        if args.load_model:
            models[cond] = load_q(fp, qmodel_path(args.load_model, args, cond), device)
        else:
            fp.set_classes(coco_names)
            models[cond] = build_q(cond, fp, args, calib, device, prot)
        sizes[cond] = weight_mib(models[cond])
        print(f"  {cond} 빌드 {time.time() - t0:.0f}s  (이론적 크기 {sizes[cond]:.2f} MiB)", flush=True)
        if mcheck:
            # +A(M 사전 검사)에 쓰는 값: calibration에 쓰지 않은 train2017 이미지에서 FP 대비 판정 손상. 같은 seed의
            # M 끔/켬 두 조건 로그를 비교해 run_comparison.MCHECK_RATIO 규칙으로 고른다(두 번 빌드하지 않으려고 분리).
            fp.set_classes(coco_names)
            f, l = judge([models[cond]], fp, mcheck, device)[0]
            print(f"  [mcheck] {cond}: calib 밖 {len(mcheck)}장 flip {f:.2f}% lost {l:.2f}% (합 {f + l:.2f})", flush=True)
        if args.save_model and not args.load_model:
            save_q(models[cond], qmodel_path(args.save_model, args, cond),
                   dict(cond=cond, args=vars(args), calib_time=time.time() - t0, model_mib=sizes[cond]))

    res = {}
    if not args.skip_coco:
        t0 = time.time()
        ev = evaluate(models, coco_names, probe, coco_ids, coco_cat_ids, device, topk=300)
        for k, v in ev.items():
            res.setdefault(k, {}).update(coco=100 * coco_ap(coco_gt, v["results"], coco_ids), coco_flip=v["flip_pct"],
                                         coco_lost=v["lost_pct"], n_conf=v["n_conf"])
            for d in (args.save_model, args.load_model):
                if d and k in conditions and not args.eval_cap:
                    update_qmodel_meta(qmodel_path(d, args, k), coco_ap=round(res[k]["coco"], 4))
        print(f"[eval] COCO {time.time() - t0:.0f}s", flush=True)
    if args.lvis:
        from lvis import LVIS
        lvis_gt = LVIS(args.lvis_ann)
        lv = set(lvis_gt.get_img_ids())
        allp = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))
        pairs = [(p, int(os.path.basename(p)[:-4])) for p in allp if int(os.path.basename(p)[:-4]) in lv]
        if args.lvis_cap > 0:
            pairs = pairs[:args.lvis_cap]
        lvis_names = load_names("lvis")
        t0 = time.time()
        ev = evaluate(models, lvis_names, [p for p, _ in pairs], [i for _, i in pairs],
                      list(range(1, len(lvis_names) + 1)), device, topk=1000)
        for k, v in ev.items():
            r = run_lvis_eval(lvis_gt, v["results"], [i for _, i in pairs])
            res.setdefault(k, {}).update(lvis=100 * r["AP"], apr=100 * r["APr"], apc=100 * r["APc"], apf=100 * r["APf"],
                                         lvis_flip=v["flip_pct"], lvis_lost=v["lost_pct"])
        print(f"[eval] LVIS {len(pairs)}장 {time.time() - t0:.0f}s", flush=True)
    cols = ("coco", "coco_flip", "coco_lost", "lvis", "apr", "apc", "apf", "lvis_flip")
    print("\n결과 | COCO AP | COCO_flip | COCO_lost | LVIS AP | APr | APc | APf | LVIS_flip")
    for k, v in res.items():
        print(f"{k} | " + " | ".join(f"{v.get(c, float('nan')):.2f}" for c in cols))
    for k, v in sizes.items():
        res[k]["size_mib"] = v
    print(f"[mem] 최대 GPU 메모리 {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB")
    print(f"[json] {json.dumps(res)}")


if __name__ == "__main__":
    main()
