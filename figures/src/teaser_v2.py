"""그림 1 후보 v2 (11-09): 기존 PTQ의 문제(조용한 라벨 바뀜)와 우리 방법의 장점(W4A4까지 버팀)을 한 장에.

(a) 정성 예시, YOLO-World-S W4A8 seed 0: FP32 / BRECQ / Ours(brecq+GPM, 새 규칙 r1102).
    검출은 qual_dets.py가 만든 /data/taeho/cvpr2027_dets/qual_s_w4a8 (fp, brecq, ours)를 쓴다. 그리기 규칙은 qual_figure.draw와 같다.
    BRECQ는 같은 설정에서 COCO AP를 FP의 92%로 유지한다 -- 그런데 held-out 어휘 라벨을 확신을 갖고 바꾼다.
(b) YOLO-World-S 비트별 LVIS AP. 11-11부터 seed 0(documents/RESULTS_FINAL_1seed.md §2와 같다). PTQ_DRAFT_SEEDS=0,1,2면 3 seed 평균(점 = seed별 값).
사용(저장소 루트): .venv/bin/python figures/src/teaser_v2.py [--image 450559]
출력: figures/teaser_v2.{pdf,png}
"""
from __future__ import annotations
import argparse, json, os, statistics as st, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qual_figure import LVIS_ANN, COCO_ROOT, iou, top_per_location, draw

DETS = "/data/taeho/cvpr2027_dets/qual_s_w4a8"
FP_LVIS = 25.89
SEEDS = tuple(int(x) for x in os.environ.get("PTQ_DRAFT_SEEDS", "0").split(","))   # 11-11: 논문 결과 seed 0 통일. 3 seed판은 PTQ_DRAFT_SEEDS=0,1,2
BITS = ["W8A8", "W4A8", "W4A5", "W4A4"]
# LVIS AP, seed 0/1/2 (확정 프로토콜, 새 규칙). Ours = brecq+GPM(seed 0 검사가 고른 구성).
VALS = {
    "BRECQ": {"W8A8": [25.54, 25.49, 25.45], "W4A8": [22.75, 22.77, 22.61],
              "W4A5": [11.29, 20.01, 19.61], "W4A4": [0.45, 0.52, 0.33]},
    "QDrop": {"W8A8": [25.37, 25.41, 25.37], "W4A8": [22.63, 22.58, 22.13],
              "W4A5": [5.74, 16.01, 16.31], "W4A4": [0.27, 0.25, 0.17]},
    "Ours":  {"W8A8": [25.78, 25.74, 25.61], "W4A8": [23.97, 24.22, 24.42],
              "W4A5": [22.94, 23.11, 22.89], "W4A4": [18.60, 17.95, 17.72]},
}
# validate_palette.js 통과(light, 인접 CVD ΔE >= 20). Ours는 정성 그림의 "맞음" 파랑과 같은 색.
COL = {"Ours": "#2a78d6", "BRECQ": "#eb6834", "QDrop": "#9a6bc9"}
INK, MUTED, GRID, AXIS = "#0b0b0b", "#6b6a64", "#e1e0d9", "#c3c2b7"


def crop_box(boxes, W, H, pad=0.30, aspect=0.82):
    x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
    w, h = (x2 - x1) * (1 + 2 * pad), (y2 - y1) * (1 + 2 * pad)
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    w, h = min(w, W), min(h, H)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    l = min(max(0, cx - w / 2), W - w); t = min(max(0, cy - h / 2), H - h)
    return l, t, l + w, t + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="450559")
    ap.add_argument("--t-fp", type=float, default=0.5)
    ap.add_argument("--t-show", type=float, default=0.15)
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "teaser_v2"))
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from PIL import Image
    from lvis import LVIS

    dets = {k: json.load(open(os.path.join(DETS, f"{k}.json"))) for k in ("fp", "brecq", "ours")}
    lv = LVIS(LVIS_ANN)
    names = {c["id"] - 1: c["name"].replace("_", " ") for c in lv.dataset["categories"]}
    iid = args.image
    gt = [([a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]], a["category_id"] - 1)
          for a in lv.dataset["annotations"] if a["image_id"] == int(iid)]
    focus = [f for f in top_per_location(dets["fp"][iid], args.t_fp)
             if any(c == f[5] and iou(f[:4], b) >= 0.5 for b, c in gt)]
    print("focus:", [(names[f[5]], round(f[4], 2)) for f in focus])
    img = Image.open(os.path.join(COCO_ROOT, "val2017", f"{int(iid):012d}.jpg")).convert("RGB")
    l, t, r, b = crop_box([f[:4] for f in focus], img.width, img.height)
    img = img.crop((int(l), int(t), int(r), int(b)))
    shift = lambda ds: [[d[0] - l, d[1] - t, d[2] - l, d[3] - t] + list(d[4:]) for d in ds]

    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.edgecolor": AXIS, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(6.9, 2.05))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.55], wspace=0.06, left=0.005, right=0.995, top=0.86, bottom=0.14)
    titles = [("fp", "FP32"), ("brecq", "BRECQ, W4A8"), ("ours", "Ours, W4A8")]
    for i, (k, ttl) in enumerate(titles):
        ax = fig.add_subplot(gs[0, i])
        draw(ax, img, shift(dets[k][iid]), shift(focus), names, args.t_show, ttl, max_boxes=len(focus))
        ax.title.set_fontsize(7.5)
    fig.text(0.005, 0.045, "(a) BRECQ keeps 92% of FP COCO AP at W4A8, yet relabels held-out (LVIS) objects with high confidence.",
             fontsize=6.6, color=INK, ha="left")

    ax = fig.add_subplot(gs[0, 3])
    pos = ax.get_position(); ax.set_position([pos.x0 + 0.06, pos.y0, pos.width - 0.06, pos.height])
    x = list(range(len(BITS)))
    ax.axhline(FP_LVIS, color=MUTED, linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
    ax.text(-0.25, FP_LVIS + 0.6, f"FP32 {FP_LVIS:.1f}", fontsize=6.3, color=MUTED, va="bottom")
    for m in ("BRECQ", "QDrop", "Ours"):
        means = [st.mean(VALS[m][bt][i] for i in SEEDS) for bt in BITS]
        for i, bt in enumerate(BITS):
            if len(SEEDS) > 1:
                ax.scatter([i] * len(SEEDS), [VALS[m][bt][k] for k in SEEDS], s=5, color=COL[m], alpha=0.35, linewidths=0, zorder=2)
        ax.plot(x, means, color=COL[m], linewidth=1.6, marker="o", markersize=4, markeredgecolor="white",
                markeredgewidth=0.8, zorder=3)
    o, bq = st.mean(VALS["Ours"]["W4A4"][k] for k in SEEDS), st.mean(VALS["BRECQ"]["W4A4"][k] for k in SEEDS)
    ax.annotate(f"{o:.1f}", (3, o), xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=6.6, color=INK)
    ax.annotate("< 0.5", (3, bq), xytext=(5, 1), textcoords="offset points",
                ha="left", va="bottom", fontsize=6.6, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(BITS, fontsize=6.5)
    ax.set_xlim(-0.3, len(BITS) - 0.1); ax.set_ylim(0, 29)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    ax.set_title("(b) held-out LVIS AP, YOLO-World-S", loc="left", fontsize=7.5, color=INK)
    h = [Line2D([], [], color=COL[m], linewidth=1.6, marker="o", markersize=4, markeredgecolor="white", label=lab)
         for m, lab in (("Ours", "Ours"), ("BRECQ", "BRECQ"), ("QDrop", "QDrop"))]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=6.2, frameon=False, handlelength=1.6, borderaxespad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300)
    print("->", args.out)


if __name__ == "__main__":
    main()
