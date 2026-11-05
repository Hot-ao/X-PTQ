"""정성 비교 1단계 (11-03): 저장된 양자화 모델로 LVIS minival 이미지의 검출을 뽑아 둔다(GPU).

FP / BRECQ / Ours를 LVIS-1203 어휘로 바꿔 같은 이미지에 돌리고, 시각화용 검출(일반 NMS, conf >= 0.1, 이미지당 최대 100개)을
모델별 JSON으로 저장한다. AP 평가 경로(predict_lvis_results, multi_label NMS, conf 0.001)와 다르다 -- 그림용이다.
사례 고르기와 그리기는 figures/src/qual_figure.py(CPU)가 한다.
사용(저장소 루트):
  CUDA_DEVICE_ORDER=PCI_BUS_ID .venv/bin/python figures/src/qual_dets.py --device 3 \\
      --tag s_w4a8 --w-bits 4 --a-bits 8 --seed 0 [--ours brecq-GPM] [--limit 0]
  QDrop만 추가: ... --keys qdrop --qdrop-path <모델.pt>
출력: /data/taeho/cvpr2027_dets/qual_<tag>/{fp,brecq,ours,qdrop}.json  ({image_id: [[x1,y1,x2,y2,score,cls], ...]})
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "core"))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

QM = "/data/taeho/cvpr2027_qmodels"
LVIS_ANN = "/data/taeho/lvis_datasets/labels_dl/extracted/lvis/annotations/lvis_v1_minival.json"
COCO_ROOT = "/data/taeho/coco_datasets"


def find_qmodel(stem, w, a, seed, cond):
    """새 규칙 폴더(r1102)를 먼저, 없으면 기존 폴더. BRECQ처럼 규칙과 무관한 조건은 기존 폴더에만 있다."""
    name = f"{stem}_w{w}a{a}_s{seed}_{cond}.pt"
    for d in (os.path.join(QM, "r1102"), QM):
        if os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    raise FileNotFoundError(name)


def dump(model, paths, ids, imgsz, device, conf):
    out = {}
    for p, iid in zip(paths, ids):
        r = model.predict(source=p, imgsz=imgsz, device=device, conf=conf, max_det=100, verbose=False, agnostic_nms=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            out[iid] = []
            continue
        xyxy = r.boxes.xyxy.cpu().numpy(); sc = r.boxes.conf.cpu().numpy(); cl = r.boxes.cls.cpu().numpy().astype(int)
        out[iid] = [[round(float(v), 1) for v in b] + [round(float(s), 4), int(c)] for b, s, c in zip(xyxy, sc, cl)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-world.pt")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--w-bits", type=int, required=True)
    ap.add_argument("--a-bits", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default="brecq", help="비교 기준선 조건(저장 모델 이름의 조건 부분)")
    ap.add_argument("--ours", default="brecq-GPM", help="우리 조건(저장 모델 이름의 조건 부분)")
    ap.add_argument("--base-path", default="", help="기준선 모델 파일을 직접 지정(이름 규칙 대신)")
    ap.add_argument("--ours-path", default="", help="우리 모델 파일을 직접 지정(이름 규칙 대신)")
    ap.add_argument("--qdrop-path", default="", help="QDrop 모델 파일. 주면 qdrop.json도 만든다")
    ap.add_argument("--keys", default="fp,brecq,ours", help="이번에 뽑을 모델(쉼표). 예: qdrop만 추가할 때 --keys qdrop")
    ap.add_argument("--device", default="3")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0, help="앞 N장만(0 = 전체 4809장)")
    args = ap.parse_args()
    import run_comparison as rc
    from model_zoo import model_factory
    from lvis import LVIS
    device = f"cuda:{args.device}"
    lv = LVIS(LVIS_ANN)
    ids_ok = set(lv.get_img_ids())
    paths, ids = [], []
    for p in sorted(glob.glob(os.path.join(COCO_ROOT, "val2017", "*.jpg"))):
        iid = int(os.path.splitext(os.path.basename(p))[0])
        if iid in ids_ok:
            paths.append(p); ids.append(iid)
    if args.limit:
        paths, ids = paths[:args.limit], ids[:args.limit]
    stem = os.path.splitext(os.path.basename(args.model))[0]
    Model = model_factory(args.model)
    lvis_names, coco = rc.load_names("lvis"), rc.load_names("coco")
    out_dir = f"/data/taeho/cvpr2027_dets/qual_{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    srcs = {"fp": None, "brecq": args.base_path or find_qmodel(stem, args.w_bits, args.a_bits, args.seed, args.base),
            "ours": args.ours_path or find_qmodel(stem, args.w_bits, args.a_bits, args.seed, args.ours)}
    if args.qdrop_path:
        srcs["qdrop"] = args.qdrop_path
    keys = args.keys.split(",")
    mp = os.path.join(out_dir, "meta.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else {"runs": []}
    meta.setdefault("runs", []).append({"args": vars(args), "sources": {k: srcs[k] for k in keys}, "n_images": len(ids)})
    for key in keys:
        path = srcs[key]
        t = time.time()
        if path is None:
            m = Model(args.model); m.set_classes(coco); m.fuse(); m.model.to(device).eval()
        else:
            m, _ = rc.load_qmodel(Model, args.model, path, device)
        rc.switch_vocab(m, lvis_names, device)
        d = dump(m, paths, ids, 640, device, args.conf)
        json.dump(d, open(os.path.join(out_dir, f"{key}.json"), "w"))
        print(f"[{key}] {len(d)}장, 검출 {sum(len(v) for v in d.values())}개, {time.time() - t:.0f}s", flush=True)
        del m
    json.dump(meta, open(mp, "w"), indent=1)
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
