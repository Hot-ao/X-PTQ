"""어휘 크기를 80 -> 1,203으로 점진적으로 키우며 flip을 잰다 (11-16, 교수님 제안 ablation).

본문 3.2절의 COCO 3.9% / LVIS 17.5%는 어휘와 이미지가 같이 바뀐 비교다. 여기서는 이미지(LVIS minival에서 고정 표본)를
고정하고 어휘만 겹쳐 가며(nested) 키운다. 양자화 모델은 다시 만들지 않고 저장된 모델의 어휘만 바꾼다(switch_vocab,
LVIS 평가와 같은 경로). YOLO-World는 neck에도 텍스트가 들어가므로 1,203 결과에서 열만 고르지 않고 어휘마다 실제로 돌린다.

어휘: 시작 80 = COCO 80 클래스에 대응하는 LVIS 클래스(LVIS 이름 그대로; 그래서 끝점 1,203은 본문 LVIS 어휘와 같다).
  여기에 나머지 1,123 클래스를 순서대로 더한다. 순서:
  rand0/1/2 = 무작위 3개, sim = 시작 80과 텍스트 임베딩 cos가 가장 큰 클래스부터, far = 가장 작은 클래스부터.
지표(FP 대비, conf 0.25, 본문 flip 정의와 같다):
  flip      = 어휘 V에서 FP가 확신하는 영역(C_V)의 top-1이 바뀐 비율
  flip_fix  = 80 어휘에서 FP가 확신한 영역(C_80)을 고정하고, V 위의 top-1이 바뀐 비율(영역 변화 없이 경쟁 클래스만 늘어난 효과)
  margin    = C_V에서 FP 로짓 top-1 - top-2의 중앙값(어휘가 클수록 줄어드는지)
사용(저장소 루트):
  CUDA_DEVICE_ORDER=PCI_BUS_ID .venv/bin/python scripts/vocab_scale.py --device 4 [--bits 4,8] [--n-img 1000] [--limit 0]
출력: runs/267_vocab_scale/w4a8_s0.json (어휘마다 저장, 다시 돌리면 이어서 한다)
"""
import argparse, glob, json, os, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "core"))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import numpy as np
import torch

LVIS_ANN = "/data/taeho/lvis_datasets/labels_dl/extracted/lvis/annotations/lvis_v1_minival.json"
COCO_ROOT = "/data/taeho/coco_datasets"
OUT = os.path.join(ROOT, "runs", "267_vocab_scale")
SIZES = [80, 150, 300, 600, 1203]
CONF = 0.25
# COCO 이름이 LVIS 동의어와 바로 안 맞는 17개(나머지 63개는 동의어로 맞춘다)
MANUAL = {"car": "car_(automobile)", "bus": "bus_(vehicle)", "train": "train_(railroad_vehicle)", "tie": "necktie",
          "skis": "ski", "sports ball": "ball", "wine glass": "wineglass", "orange": "orange_(fruit)", "hot dog": "sausage",
          "potted plant": "flowerpot", "laptop": "laptop_computer", "mouse": "mouse_(computer_equipment)",
          "remote": "remote_control", "keyboard": "computer_keyboard", "cell phone": "cellular_telephone",
          "microwave": "microwave_oven", "hair drier": "hair_dryer"}


def coco_to_lvis_idx(coco_names, cats):
    norm = lambda s: s.lower().replace(" ", "_").replace("-", "_")
    by_name = {c["name"]: i for i, c in enumerate(cats)}
    idx = []
    for n in coco_names:
        if n in MANUAL:
            idx.append(by_name[MANUAL[n]]); continue
        hit = [i for i, c in enumerate(cats) if norm(n) == norm(c["name"]) or norm(n) in [norm(s) for s in c["synonyms"]]]
        assert len(hit) == 1, (n, hit)
        idx.append(hit[0])
    assert len(set(idx)) == 80
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-world.pt")
    ap.add_argument("--bits", default="4,8", help="w,a")
    ap.add_argument("--seed", type=int, default=0, help="양자화 모델 seed")
    ap.add_argument("--base", default="brecq")
    ap.add_argument("--ours", default="brecq-GPM")
    ap.add_argument("--device", default="4")
    ap.add_argument("--n-img", type=int, default=1000, help="LVIS minival에서 고정 표본(무작위, seed 0)")
    ap.add_argument("--orders", default="rand0,rand1,rand2,sim,far")
    ap.add_argument("--limit", type=int, default=0, help="속도 확인용: 이미지 앞 N장, 어휘 크기 80·1203만")
    args = ap.parse_args()
    import run_comparison as rc
    from model_zoo import model_factory
    from harness import SimilarityHarness
    sys.path.insert(0, os.path.join(ROOT, "figures", "src"))
    from qual_dets import find_qmodel
    device = "cpu" if args.device == "cpu" else f"cuda:{args.device}"
    w, a = map(int, args.bits.split(","))
    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, f"w{w}a{a}_s{args.seed}" + (f"_lim{args.limit}" if args.limit else "") + ".json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {"meta": vars(args), "vocabs": {}}

    # 이미지: LVIS minival 중 val2017에 있는 4,809장에서 고정 표본
    from lvis import LVIS
    lv = LVIS(LVIS_ANN)
    cats = sorted(lv.dataset["categories"], key=lambda c: c["id"])
    ok = set(lv.get_img_ids())
    paths = [p for p in sorted(glob.glob(os.path.join(COCO_ROOT, "val2017", "*.jpg")))
             if int(os.path.splitext(os.path.basename(p))[0]) in ok]
    rng = np.random.RandomState(0)
    paths = [paths[i] for i in sorted(rng.choice(len(paths), min(args.n_img, len(paths)), replace=False))]
    if args.limit:
        paths = paths[:args.limit]
    lvis_names, coco_names = rc.load_names("lvis"), rc.load_names("coco")
    assert len(lvis_names) == len(cats) == 1203
    base = coco_to_lvis_idx(coco_names, cats)
    rest = [i for i in range(1203) if i not in set(base)]

    Model = model_factory(args.model)
    stem = os.path.splitext(os.path.basename(args.model))[0]
    fp = Model(args.model); fp.set_classes(coco_names); fp.fuse(); fp.model.to(device).eval()
    # 텍스트 임베딩(FP, LVIS 1,203) -> sim/far 순서
    rc.switch_vocab(fp, lvis_names, device)
    tf = fp.model.txt_feats.reshape(-1, fp.model.txt_feats.shape[-1]).float().cpu()
    tf = torch.nn.functional.normalize(tf, dim=-1)
    near = (tf[rest] @ tf[base].T).max(1).values.numpy()
    orders = {}
    for o in args.orders.split(","):
        if o.startswith("rand"):
            orders[o] = [rest[i] for i in np.random.RandomState(int(o[4:]) + 100).permutation(len(rest))]
        elif o == "sim":
            orders[o] = [rest[i] for i in np.argsort(-near, kind="stable")]
        elif o == "far":
            orders[o] = [rest[i] for i in np.argsort(near, kind="stable")]
    sizes = [80, 1203] if args.limit else SIZES
    vocabs = {}                                          # 키 -> LVIS 인덱스(정렬). 80과 1203은 모든 순서가 공유
    for o, ordr in orders.items():
        for n in sizes:
            key = "all80" if n == 80 else ("all1203" if n == 1203 else f"{o}_{n}")
            vocabs[key] = sorted(base + ordr[:n - 80])
    res["meta"].update(n_images=len(paths), base_lvis_idx=base,
                       orders={o: v[:600] for o, v in orders.items()},
                       sources={k: find_qmodel(stem, w, a, args.seed, getattr(args, k)) for k in ("base", "ours")})
    qm = {k: rc.load_qmodel(Model, args.model, res["meta"]["sources"][k], device)[0] for k in ("base", "ours")}

    # C_80(80 어휘에서 FP가 확신한 영역)을 먼저 만든다
    keys = ["all80"] + [k for k in vocabs if k != "all80"]
    fix = {}
    for key in keys:
        V = vocabs[key]
        if key in res["vocabs"] and key != "all80":
            print(f"[skip] {key}"); continue
        t0 = time.time()
        names = [lvis_names[i] for i in V]
        pos80 = torch.tensor([V.index(i) for i in base])      # V 안에서 시작 80 클래스의 열
        for m in [fp] + list(qm.values()):
            rc.switch_vocab(m, names, device)
        hs = {"fp": SimilarityHarness(fp.model, device=device)}
        hs.update({k: SimilarityHarness(m.model, device=device) for k, m in qm.items()})
        acc = {k: dict(tot=0, fl=0, tot_fix=0, fl_fix=0) for k in qm}
        margins = []
        for ii, p in enumerate(paths):
            t = rc.preprocess(p, 640)
            sf = hs["fp"].run_image(t, ii).sim.float()
            prob = sf.sigmoid(); mp, cf = prob.max(-1)
            conf = (mp > CONF).nonzero(as_tuple=True)[0]
            if key == "all80":
                fix[ii] = conf.cpu()
            fx = fix[ii].to(sf.device)
            if len(conf) and sf.shape[1] > 1:
                top2 = sf[conf].topk(2, dim=-1).values
                margins.append((top2[:, 0] - top2[:, 1]).cpu())
            for k in qm:
                sq = hs[k].run_image(t, ii).sim.float()
                cq = sq.argmax(-1)
                acc[k]["tot"] += len(conf); acc[k]["fl"] += int((cf[conf] != cq[conf]).sum())
                acc[k]["tot_fix"] += len(fx); acc[k]["fl_fix"] += int((cf[fx] != cq[fx]).sum())
        for h in hs.values():
            h.close()
        mg = torch.cat(margins) if margins else torch.zeros(1)
        r = {"size": len(V), "n_conf": acc["base"]["tot"], "n_fix": acc["base"]["tot_fix"],
             "margin_med": float(mg.median()), "sec": round(time.time() - t0)}
        for k in qm:
            r[f"flip_{k}"] = 100 * acc[k]["fl"] / max(acc[k]["tot"], 1)
            r[f"flipfix_{k}"] = 100 * acc[k]["fl_fix"] / max(acc[k]["tot_fix"], 1)
        res["vocabs"][key] = r
        json.dump(res, open(out_path, "w"), indent=1)
        print(f"[{key}] |V|={len(V)} n={r['n_conf']} margin={r['margin_med']:.3f} "
              f"flip base/ours {r['flip_base']:.2f}/{r['flip_ours']:.2f}  "
              f"fix {r['flipfix_base']:.2f}/{r['flipfix_ours']:.2f}  ({r['sec']}s)", flush=True)
    print("->", out_path)


if __name__ == "__main__":
    main()
