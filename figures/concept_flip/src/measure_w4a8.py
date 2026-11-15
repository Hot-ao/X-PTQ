"""그림 1 (Voronoi) 실측 보강 (11-13). YOLO-World-S W4A8 seed 0, 저장 모델(BRECQ, Ours = brecq+GPM 새 규칙).

1) COCO 어휘 점수: LVIS minival 450559에서 FP의 helmet 앵커(LVIS 어휘 기준, dump_scores.py와 같은 앵커)를 COCO 80 어휘로
   다시 채점해 FP / BRECQ / Ours의 클래스별 점수를 저장한다(그림 (a)의 칸을 실제 COCO 클래스로 그리기 위해).
2) 영역 임베딩 오차와 flip: core/diag_w4_sensitivity.measure(12.cv2 개입 실험 runs/257과 같은 지표)를
   LVIS minival(val2017) 앞 N장에 적용. emb = L2 정규화 영역 임베딩의 FP 대비 제곱 거리(FP 확신 위치 가중 평균).
사용(저장소 루트): CUDA_DEVICE_ORDER=PCI_BUS_ID .venv/bin/python figures/concept_flip/src/measure_w4a8.py --device 7 --n 500
출력: figures/concept_flip/data/coco_scores_450559.json, data/w4a8_emb_flip_n<N>.json
"""
from __future__ import annotations
import argparse, glob, json, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "core"))
sys.path.insert(0, os.path.join(ROOT, "figures", "src"))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
DATA = os.path.join(ROOT, "figures", "concept_flip", "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="7")
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    import torch
    import run_comparison as rc
    import diag_w4_sensitivity as dg
    from model_zoo import model_factory
    from qual_dets import find_qmodel, COCO_ROOT, LVIS_ANN
    from quant.pdquant import _find_head
    from quant.vocab_metric import VocabMetric, encode_text_bank
    from ultralytics import YOLOWorld
    from ultralytics.utils import nms as nms_mod
    from lvis import LVIS
    torch.manual_seed(0); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    dev = f"cuda:{args.device}"
    W = os.path.join(ROOT, "yolov8s-world.pt")
    coco, lvis = rc.load_names("coco"), rc.load_names("lvis")
    Model = model_factory("yolov8s-world.pt")
    srcs = {"fp": None, "brecq": find_qmodel("yolov8s-world", 4, 8, 0, "brecq"),
            "ours": find_qmodel("yolov8s-world", 4, 8, 0, "brecq-GPM")}

    def load(key):
        if srcs[key] is None:
            m = Model("yolov8s-world.pt"); m.set_classes(coco); m.fuse(); m.model.to(dev).eval(); return m
        m, _ = rc.load_qmodel(Model, "yolov8s-world.pt", srcs[key], dev); return m

    # ---- 1) COCO 어휘 점수(앵커는 dump_scores.py의 LVIS helmet 앵커)
    anchor = json.load(open(os.path.join(DATA, "scores_450559_helmet.json")))["anchor"]
    cap = {}
    orig = nms_mod.non_max_suppression

    def hook(prediction, *a, **k):
        p = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
        cap["pred"] = p.detach().float().cpu(); return orig(prediction, *a, **k)
    nms_mod.non_max_suppression = hook
    img = os.path.join(COCO_ROOT, "val2017", "000000450559.jpg")
    sc = {}
    for key in srcs:
        m = load(key); rc.switch_vocab(m, coco, dev)
        m.predict(source=img, imgsz=640, device=dev, conf=0.1, verbose=False)
        sc[key] = cap["pred"][0][4:, anchor].tolist()
        del m; torch.cuda.empty_cache()
    nms_mod.non_max_suppression = orig
    order = sorted(range(len(coco)), key=lambda i: -max(sc[k][i] for k in sc))[:8]
    out1 = dict(image=450559, anchor=anchor, vocab="coco", classes=[coco[i] for i in order],
                scores={k: [round(v[i], 4) for i in order] for k, v in sc.items()},
                top1={k: coco[max(range(len(coco)), key=lambda i: v[i])] for k, v in sc.items()}, sources=srcs)
    json.dump(out1, open(os.path.join(DATA, "coco_scores_450559.json"), "w"), indent=1)
    print("[coco]", json.dumps({k: out1[k] for k in ("classes", "scores", "top1")}), flush=True)

    # ---- 2) 영역 임베딩 오차와 flip (val2017, LVIS minival 이미지 정렬 앞 N장)
    ids = set(LVIS(LVIS_ANN).get_img_ids())
    paths = [p for p in sorted(glob.glob(os.path.join(COCO_ROOT, "val2017", "*.jpg")))
             if int(os.path.basename(p)[:-4]) in ids][:args.n]
    evals = [rc.preprocess(p, 640) for p in paths]
    fp = YOLOWorld(W); fp.set_classes(coco); fp.fuse(); fp.model.to(dev).eval()
    dg.HS = dg.HeadSim(_find_head(fp.model))
    vl = VocabMetric(encode_text_bank(YOLOWorld, W, lvis, dev))
    vc = VocabMetric(encode_text_bank(YOLOWorld, W, coco, dev))
    refs = dg.reference(fp, evals, dev, vl, vc, 0.25, "coco")
    res = {}
    for key in ("brecq", "ours"):
        q = load(key)
        r = dg.measure(q, evals, refs, dev, vl, vc)
        res[key] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
        print(f"[{key}] emb {r['emb']:.4f}  COCO flip {r['coco']:.2f}%  LVIS flip {r['lvis']:.2f}%", flush=True)
        del q; torch.cuda.empty_cache()
    out2 = dict(n_images=len(paths), images="LVIS minival (val2017) sorted, first N", metric="core/diag_w4_sensitivity.measure",
                tau=0.25, sources=srcs, results=res)
    f = os.path.join(DATA, f"w4a8_emb_flip_n{len(paths)}.json")
    json.dump(out2, open(f, "w"), indent=1); print("->", f)


if __name__ == "__main__":
    main()
