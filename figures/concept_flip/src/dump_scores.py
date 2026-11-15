"""개념 그림 B/C용 (11-13): 한 영역(앵커)의 1203개 LVIS 클래스 점수를 FP / BRECQ / Ours에서 뽑는다(GPU, 이미지 1장).

NMS 입력(raw 예측, (1, 4+nc, N))을 가로채서, FP에서 대상 클래스(helmet) 점수가 가장 높은 앵커를 고르고
세 모델 모두 같은 앵커의 클래스별 점수(sigmoid 뒤)를 저장한다. 입력 이미지·앵커 격자가 같으므로 앵커 번호로 맞출 수 있다.
모델: YOLO-World-S W4A8 seed 0, BRECQ = qmodels/yolov8s-world_w4a8_s0_brecq.pt, Ours = r1102/..._brecq-GPM.pt
(figures/src/qual_dets.py와 같은 모델·같은 어휘 전환).
사용(저장소 루트): CUDA_DEVICE_ORDER=PCI_BUS_ID .venv/bin/python figures/concept_flip/src/dump_scores.py --device 7
출력: figures/concept_flip/data/scores_450559_helmet.json
"""
from __future__ import annotations
import argparse, json, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "core"))
sys.path.insert(0, os.path.join(ROOT, "figures", "src"))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="7")
    ap.add_argument("--image", type=int, default=450559)
    ap.add_argument("--target", default="helmet")
    ap.add_argument("--topk", type=int, default=12)
    args = ap.parse_args()
    import torch
    import run_comparison as rc
    from model_zoo import model_factory
    from qual_dets import find_qmodel, COCO_ROOT
    from ultralytics.utils import nms as nms_mod
    device = f"cuda:{args.device}"
    cap = {}
    orig = nms_mod.non_max_suppression

    def hook(prediction, *a, **k):
        p = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
        cap["pred"] = p.detach().float().cpu()
        return orig(prediction, *a, **k)
    nms_mod.non_max_suppression = hook

    names = rc.load_names("lvis"); coco = rc.load_names("coco")
    tgt = names.index(args.target) if args.target in names else [i for i, n in enumerate(names) if n.startswith(args.target)][0]
    img = os.path.join(COCO_ROOT, "val2017", f"{args.image:012d}.jpg")
    Model = model_factory("yolov8s-world.pt")
    srcs = {"fp": None, "brecq": find_qmodel("yolov8s-world", 4, 8, 0, "brecq"),
            "ours": find_qmodel("yolov8s-world", 4, 8, 0, "brecq-GPM")}
    scores, anchor = {}, None
    for key, path in srcs.items():
        if path is None:
            m = Model("yolov8s-world.pt"); m.set_classes(coco); m.fuse(); m.model.to(device).eval()
        else:
            m, _ = rc.load_qmodel(Model, "yolov8s-world.pt", path, device)
        rc.switch_vocab(m, names, device)
        m.predict(source=img, imgsz=640, device=device, conf=0.1, verbose=False)
        p = cap["pred"][0]                    # (4+nc, N)
        cls = p[4:]
        if anchor is None:                    # FP에서 대상 클래스 점수가 가장 높은 앵커
            anchor = int(cls[tgt].argmax())
        scores[key] = cls[:, anchor].tolist()
        print(key, path, "anchor", anchor, "target", round(scores[key][tgt], 3), flush=True)
        del m; torch.cuda.empty_cache()
    fp = scores["fp"]
    order = sorted(range(len(names)), key=lambda i: -max(fp[i], scores["brecq"][i], scores["ours"][i]))[:args.topk]
    out = dict(image=args.image, target=names[tgt], anchor=anchor, sources={k: v for k, v in srcs.items()},
               classes=[names[i] for i in order],
               scores={k: [round(v[i], 4) for i in order] for k, v in scores.items()},
               top1={k: names[max(range(len(names)), key=lambda i: v[i])] for k, v in scores.items()},
               n_vocab=len(names))
    os.makedirs(os.path.join(ROOT, "figures", "concept_flip", "data"), exist_ok=True)
    f = os.path.join(ROOT, "figures", "concept_flip", "data", f"scores_{args.image}_{args.target}.json")
    json.dump(out, open(f, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("classes", "scores", "top1")}, indent=0))
    print("->", f)


if __name__ == "__main__":
    main()
