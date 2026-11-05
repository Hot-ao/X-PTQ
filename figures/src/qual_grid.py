"""정성 비교 한 장 요약 (11-03): 같은 이미지에서 FP32 / BRECQ / Ours를 W4A8 · W4A5 · W4A4로 한 번에 보인다(CPU).
(QDrop 행은 그림이 복잡해져 넣지 않기로 했다(11-03). METHODS에 ("qdrop", "QDrop")을 더하면 들어간다 -- qdrop.json 필요.)

qual_dets.py가 만든 세 태그(s_w4a8, s_w4a5, s_w4a4; 각 fp, brecq, ours)의 검출을 쓴다. 판정 규칙은 qual_figure.py와 같다
(위치마다 최고 점수 박스가 모델의 판정, FP 판정 중 GT와 맞는 것만 대상, IoU >= 0.5).
사례 규칙(손으로 고르지 않는다):
  - 대상마다 세 비트 모두에서 Ours는 FP와 같은 클래스, 기준선(BRECQ)은 다른 클래스이거나 놓침 = strict win
  - 후보 이미지 = strict win >= 1, 그리고 어느 비트·어느 대상에서도 기준선만 맞히고 Ours가 틀린 경우 없음
  - 희귀(r) > 흔함(c) > 잦음(f), 그다음 FP 판정 수가 적은 순(그림이 덜 복잡한 것)
배치: 왼쪽 열 = FP32(두 행에 걸쳐), 그 오른쪽 = 행 BRECQ / Ours, 열 W4A8 / W4A5 / W4A4.
사용(저장소 루트):
  .venv/bin/python figures/src/qual_grid.py                 # 후보 목록 figures/qual_grid_candidates.md + 1순위 그림
  .venv/bin/python figures/src/qual_grid.py --pick 446651,509824,187055    # 그 이미지들로(위아래로 쌓음)
출력: figures/qual_grid.{pdf,png}
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qual_figure import LVIS_ANN, COCO_ROOT, FREQ_RANK, iou, match, top_per_location, draw

BITS = [("s_w4a8", "W4A8"), ("s_w4a5", "W4A5"), ("s_w4a4", "W4A4")]
METHODS = [("brecq", "BRECQ"), ("ours", "Ours")]
BASES = [k for k, _ in METHODS if k != "ours"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-fp", type=float, default=0.3)
    ap.add_argument("--t-show", type=float, default=0.15)
    ap.add_argument("--pick", default="", help="image_id(쉼표로 여러 장이면 위아래로 쌓는다)")
    ap.add_argument("--max-boxes", type=int, default=8)
    args = ap.parse_args()
    from lvis import LVIS
    dets = {}
    seeds = {}
    for tag, _ in BITS:
        d = f"/data/taeho/cvpr2027_dets/qual_{tag}"
        dets[tag] = {k: json.load(open(os.path.join(d, f"{k}.json"))) for k in ["fp"] + [m for m, _ in METHODS]}
        m = json.load(open(os.path.join(d, "meta.json")))
        seeds[tag] = (m["runs"][0] if "runs" in m else m)["args"]["seed"]      # 11-03 이전 meta는 runs 목록이 없다
    fp = dets["s_w4a8"]["fp"]                     # FP 모델은 태그와 무관하게 같다
    lv = LVIS(LVIS_ANN)
    gt = defaultdict(list)
    for a in lv.dataset["annotations"]:
        x, y, w, h = a["bbox"]
        gt[a["image_id"]].append(([x, y, x + w, y + h], a["category_id"] - 1))
    freq = {c["id"] - 1: c["frequency"] for c in lv.dataset["categories"]}
    names = {c["id"] - 1: c["name"].replace("_", " ") for c in lv.dataset["categories"]}
    cand = []
    for iid, fdets in fp.items():
        g = gt.get(int(iid), [])
        targets = [f for f in top_per_location(fdets, args.t_fp) if any(c == f[5] and iou(f[:4], b) >= 0.5 for b, c in g)]
        wins, bad = [], False
        for f in targets:
            lab = {(tag, k): match(f[:4], dets[tag][k][iid], args.t_show) for tag, _ in BITS for k, _ in METHODS}
            ours_ok = all(lab[(tag, "ours")] == f[5] for tag, _ in BITS)
            base_bad = all(lab[(tag, k)] != f[5] for tag, _ in BITS for k in BASES)
            if any(lab[(tag, k)] == f[5] and lab[(tag, "ours")] != f[5] for tag, _ in BITS for k in BASES):
                bad = True
            if ours_ok and base_bad:
                wins.append((f, lab))
        if wins and not bad:
            cand.append((int(iid), wins, len(top_per_location(fdets, args.t_fp))))
    cand.sort(key=lambda x: (min(FREQ_RANK[freq[f[5]]] for f, _ in x[1]), x[2], -len(x[1])))
    nm = lambda c: "놓침" if c is None else names[c]
    lines = ["# 한 장 요약 후보 (FP32 / BRECQ / Ours x W4A8 · W4A5 · W4A4)", "",
             f"규칙: FP 판정(점수 >= {args.t_fp}) 중 GT와 맞는 대상에서, 세 비트 모두 Ours는 FP와 같은 클래스이고 BRECQ는 "
             f"다른 클래스이거나 놓침. 기준선만 맞힌 경우가 하나라도 있는 이미지는 뺀다. seed: " +
             ", ".join(f"{b} s{seeds[t]}" for t, b in BITS) + ".", "",
             f"후보 {len(cand)}장.", "", "| image_id | FP 판정 수 | 대상: FP·Ours 클래스 [빈도] (BRECQ W4A8/A5/A4) |",
             "|---|---|---|"]
    for iid, wins, n in cand[:60]:
        desc = "; ".join(f"{names[f[5]]} [{freq[f[5]]}] (" + " · ".join("/".join(nm(lab[(t, k)]) for t, _ in BITS) for k in BASES)
                         + ")" for f, lab in wins)
        lines.append(f"| {iid} | {n} | {desc} |")
    open(os.path.join(ROOT, "figures", "qual_grid_candidates.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[4:20]))
    if not cand:
        return
    picks = [int(x) for x in args.pick.split(",")] if args.pick else [cand[0][0]]
    wins_of = {i: w for i, w, _ in cand}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    # 이미지마다 한 블록(FP32 | 행 이름 | W4A8 | W4A5 | W4A4, 행 = METHODS)을 위아래로 쌓는다. 열 제목은 맨 위 블록만.
    ims = [Image.open(os.path.join(COCO_ROOT, "val2017", f"{iid:012d}.jpg")).convert("RGB") for iid in picks]
    ars = [im.height / im.width for im in ims]
    W, nr, gap = 7.0, len(METHODS), 0.12             # gap: 블록 사이 간격(in)
    pw = W / 4.09                                    # 패널 폭(행 이름 열 0.09 포함)
    bh = [nr * pw * ar for ar in ars]                # 블록 높이(in)
    H = sum(bh) + gap * (len(picks) - 1) + 0.25
    fig = plt.figure(figsize=(W, H))
    outer = fig.add_gridspec(2 * len(picks) - 1, 1, left=0.005, right=0.998, top=1 - 0.22 / H, bottom=0.005, hspace=0,
                             height_ratios=sum([[b, gap] for b in bh], [])[:-1])
    for b, (iid, img) in enumerate(zip(picks, ims)):
        focus = [f for f, _ in wins_of[iid]] if iid in wins_of else top_per_location(fp[str(iid)], args.t_fp)
        gs = outer[2 * b].subgridspec(nr, 5, wspace=0.02, hspace=0.03, width_ratios=[1, 0.09, 1, 1, 1])
        # FP32는 왼쪽 열에서 블록 행들의 가운데 높이에 한 칸 크기로 둔다
        a0 = fig.add_subplot(gs[:, 0]); a0.axis("off")
        pos = a0.get_position(); h = pos.height / nr
        fa = fig.add_axes([pos.x0, pos.y0 + (pos.height - h) / 2, pos.width, h])
        draw(fa, img, fp[str(iid)], focus, names, args.t_show, "FP32", args.max_boxes)
        for r, (k, mname) in enumerate(METHODS):
            for c, (tag, bname) in enumerate(BITS):
                ax = fig.add_subplot(gs[r, c + 2])
                draw(ax, img, dets[tag][k][str(iid)], focus, names, args.t_show, bname if (b == 0 and r == 0) else "",
                     args.max_boxes)
                if c == 0:
                    ax.set_ylabel(mname, fontsize=8, rotation=90, labelpad=3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, "figures", f"qual_grid.{ext}"), dpi=250)
    print(f"그림({','.join(map(str, picks))}) -> figures/qual_grid.{{pdf,png}}")

if __name__ == "__main__":
    main()
