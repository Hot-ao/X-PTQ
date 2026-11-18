"""어휘 크기별 LVIS AP (11-16). scripts/vocab_scale.py(flip)와 같은 어휘·같은 이미지 1,000장에서 AP를 잰다.

어휘와 시작 80 클래스는 flip 결과 JSON(runs/267_vocab_scale/w4a8_s0.json의 meta: base_lvis_idx, orders 앞 600)에서 그대로 읽는다.
예측·채점은 본문 LVIS AP와 같은 경로(run_comparison.predict_lvis_results: multi_label NMS, conf 0.001 /
run_lvis_eval: Fixed AP)를 쓰고, 예측 클래스 번호(V 안의 순서)를 LVIS category id로 바꾼다.
  AP_V  = 어휘 V의 클래스 전체로 채점(어휘마다 채점 클래스가 다르다)
  AP_80 = 시작 80 클래스만으로 채점(모든 어휘에서 같은 클래스: 경쟁 클래스가 늘어난 효과만 본다)
사용(저장소 루트):
  CUDA_DEVICE_ORDER=PCI_BUS_ID .venv/bin/python scripts/vocab_scale_ap.py --device 4 [--bits 4,8] [--limit 0]
출력: runs/267_vocab_scale/ap_w4a8_s0.json (어휘마다 저장, 다시 돌리면 이어서 한다)
"""
import argparse, glob, json, os, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "core"))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import contextlib, io
import numpy as np

OUT = os.path.join(ROOT, "runs", "267_vocab_scale")
LVIS_ANN = "/data/taeho/lvis_datasets/labels_dl/extracted/lvis/annotations/lvis_v1_minival.json"
COCO_ROOT = "/data/taeho/coco_datasets"
SIZES = [80, 150, 300, 600, 1203]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-world.pt")
    ap.add_argument("--bits", default="4,8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="4")
    ap.add_argument("--orders", default="rand0,rand1,rand2,sim,far")
    ap.add_argument("--limit", type=int, default=0, help="속도 확인용: 이미지 앞 N장, 어휘 80·150만")
    args = ap.parse_args()
    import run_comparison as rc
    from model_zoo import model_factory
    from lvis import LVIS
    device = "cpu" if args.device == "cpu" else f"cuda:{args.device}"
    w, a = map(int, args.bits.split(","))
    src = json.load(open(os.path.join(OUT, "w4a8_s0.json")))["meta"]          # 어휘 정의는 flip 실행과 같게
    base, orders = src["base_lvis_idx"], src["orders"]
    out_path = os.path.join(OUT, f"ap_w{w}a{a}_s{args.seed}" + (f"_lim{args.limit}" if args.limit else "") + ".json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {"meta": vars(args), "vocabs": {}}

    lv = LVIS(LVIS_ANN)
    ok = set(lv.get_img_ids())
    paths = [p for p in sorted(glob.glob(os.path.join(COCO_ROOT, "val2017", "*.jpg")))
             if int(os.path.splitext(os.path.basename(p))[0]) in ok]
    rng = np.random.RandomState(0)                                            # vocab_scale.py와 같은 표본
    paths = [paths[i] for i in sorted(rng.choice(len(paths), min(1000, len(paths)), replace=False))]
    if args.limit:
        paths = paths[:args.limit]
    assert len(paths) == src["n_images"] or args.limit
    ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in paths]
    lvis_names = rc.load_names("lvis")

    vocabs = {}
    for o in args.orders.split(","):
        for n in ([80, 150] if args.limit else SIZES):
            key = "all80" if n == 80 else ("all1203" if n == 1203 else f"{o}_{n}")
            # 11-16 수정: 저장된 순서는 앞 600개뿐이라 1,203은 순서와 무관하게 LVIS 전체로 둔다(처음 실행에서 680개로 잘못 채점)
            vocabs[key] = list(range(1203)) if n == 1203 else sorted(base + orders[o][:n - 80])
            assert len(vocabs[key]) == n

    Model = model_factory(args.model)
    stem = os.path.splitext(os.path.basename(args.model))[0]
    sys.path.insert(0, os.path.join(ROOT, "figures", "src"))
    from qual_dets import find_qmodel
    srcs = {"fp": None, "base": find_qmodel(stem, w, a, args.seed, "brecq"), "ours": find_qmodel(stem, w, a, args.seed, "brecq-GPM")}
    res["meta"].update(n_images=len(paths), sources=srcs)
    fp = Model(args.model); fp.set_classes(rc.load_names("coco")); fp.fuse(); fp.model.to(device).eval()
    models = {"fp": fp}
    models.update({k: rc.load_qmodel(Model, args.model, srcs[k], device)[0] for k in ("base", "ours")})

    def evaluate(preds, cat_ids):
        f = io.StringIO()
        with contextlib.redirect_stdout(f):                                   # LVISEval 표를 숨긴다
            from lvis import LVISEval, LVISResults
            from collections import defaultdict
            if not preds:
                return {"AP": 0.0}
            by = defaultdict(list)
            for r in preds:
                by[r["category_id"]].append(r)
            fixed = [r for rs in by.values() for r in sorted(rs, key=lambda x: -x["score"])[:10000]]
            ev = LVISEval(lv, LVISResults(lv, fixed, max_dets=-1), iou_type="bbox")
            ev.params.img_ids = ids; ev.params.cat_ids = cat_ids
            ev.run()
            return {k: float(v) for k, v in ev.get_results().items()}

    for key, V in vocabs.items():
        if key in res["vocabs"]:
            print(f"[skip] {key}"); continue
        t0 = time.time()
        names = [lvis_names[i] for i in V]
        r = {"size": len(V)}
        for k, m in models.items():
            rc.switch_vocab(m, names, device)
            preds = rc.predict_lvis_results(m, paths, ids, 640, device)
            for p in preds:                                                   # V 안의 순서 -> LVIS category id
                p["category_id"] = V[p["category_id"] - 1] + 1
            av = evaluate(preds, [i + 1 for i in V])
            a80 = evaluate([p for p in preds if p["category_id"] - 1 in set(base)], [i + 1 for i in base])
            r[f"APV_{k}"] = av["AP"] * 100; r[f"AP80_{k}"] = a80["AP"] * 100
            r[f"APrV_{k}"] = av.get("APr", 0) * 100
        r["sec"] = round(time.time() - t0)
        res["vocabs"][key] = r
        json.dump(res, open(out_path, "w"), indent=1)
        print(f"[{key}] |V|={len(V)} AP_V fp/base/ours {r['APV_fp']:.2f}/{r['APV_base']:.2f}/{r['APV_ours']:.2f}  "
              f"AP_80 {r['AP80_fp']:.2f}/{r['AP80_base']:.2f}/{r['AP80_ours']:.2f}  ({r['sec']}s)", flush=True)
    print("->", out_path)


if __name__ == "__main__":
    main()
