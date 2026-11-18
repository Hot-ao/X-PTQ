"""그림 1 후보 v3 (11-12): 현상 -> 원인 -> 효과를 한 줄에.

(a) 현상: YOLO-World-S W4A8 seed 0, FP32 / BRECQ / Ours 정성 비교(teaser_v2와 같은 검출·같은 이미지).
(b) 원인: YOLO-World-S 층별 누수 없는 진단(Δ flip, configs/protect_cache). diag_concentration.py와 같은 데이터·같은 규칙.
(c) 효과: W4A4, YOLO 계열 6개 모델 LVIS AP 3 seed 평균(documents/RESULTS_2026-11-04.md §1).
사용(저장소 루트): .venv/bin/python figures/src/teaser_v3.py [--image 450559]
출력: figures/teaser/teaser_v3.{pdf,png}
"""
from __future__ import annotations
import argparse, json, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qual_figure import LVIS_ANN, COCO_ROOT, iou, top_per_location, draw
from teaser_v2 import DETS, crop_box, COL, INK, MUTED, GRID, AXIS
from diag_concentration import MODELS as DIAG_MODELS, PC, metrics, BUDGET

# W4A4 LVIS AP, 3 seed 평균 (RESULTS §1). FP32는 모델별.
W4A4 = [  # (이름, FP32, BRECQ, QDrop, Ours)
    ("S", 25.89, 0.43, 0.23, 18.09),
    ("S-v2", 22.19, 0.21, 0.01, 10.90),
    ("M", 29.80, 0.65, 0.03, 14.95),
    ("L", 33.43, 0.04, 0.00, 23.43),
    ("E-v8s", 25.94, 3.44, 0.13, 20.15),
    ("E-11s", 26.81, 0.47, 0.10, 10.97),
]
GRAY = "#c3c2b7"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="450559")
    ap.add_argument("--t-fp", type=float, default=0.5)
    ap.add_argument("--t-show", type=float, default=0.15)
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "teaser", "teaser_v3"))
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from PIL import Image
    from lvis import LVIS

    # ---- (a) 데이터
    dets = {k: json.load(open(os.path.join(DETS, f"{k}.json"))) for k in ("fp", "brecq", "ours")}
    lv = LVIS(LVIS_ANN)
    names = {c["id"] - 1: c["name"].replace("_", " ") for c in lv.dataset["categories"]}
    iid = args.image
    gt = [([a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]], a["category_id"] - 1)
          for a in lv.dataset["annotations"] if a["image_id"] == int(iid)]
    focus = [f for f in top_per_location(dets["fp"][iid], args.t_fp)
             if any(c == f[5] and iou(f[:4], b) >= 0.5 for b, c in gt)]
    img = Image.open(os.path.join(COCO_ROOT, "val2017", f"{int(iid):012d}.jpg")).convert("RGB")
    l, t, r, b = crop_box([f[:4] for f in focus], img.width, img.height, pad=0.22, aspect=0.95)
    img = img.crop((int(l), int(t), int(r), int(b)))
    shift = lambda ds: [[d[0] - l, d[1] - t, d[2] - l, d[3] - t] + list(d[4:]) for d in ds]

    # ---- (b) 데이터
    name, f, kind, gset = DIAG_MODELS[0]
    assert name == "YOLO-World-S"
    rows = json.load(open(os.path.join(PC, f)))
    mt = metrics(rows, gset)

    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.edgecolor": AXIS, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(6.9, 2.15))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.35, 1.75], wspace=0.05,
                          left=0.004, right=0.995, top=0.81, bottom=0.23)
    helmet = [f for f in focus if names[f[5]] == "helmet"]       # 판정이 바뀐 대상만 색으로(나머지는 회색)
    fp_h = helmet[0]
    for i, (k, ttl) in enumerate([("brecq", "BRECQ"), ("ours", "Ours")]):
        ax = fig.add_subplot(gs[0, i])
        draw(ax, img, shift(dets[k][iid]), shift(helmet), names, args.t_show, ttl, max_boxes=1)
        ax.title.set_fontsize(7)
        fig.canvas.draw()                  # 라벨이 이미지 밖으로 나가면 이미지 안으로 민다
        rnd = fig.canvas.get_renderer(); inv = ax.transData.inverted()
        for tx in ax.texts:
            bb = tx.get_window_extent(rnd); (xa, _), (xb, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
            wdt = xb - xa
            tx.set_ha("left"); tx.set_x(min(max(0, xa), img.width - wdt))
    fig.text(0.004, 0.955, "(a) What breaks (W4A8)", fontsize=7.5, color=INK, weight="bold")
    fig.text(0.004, 0.07, f"FP32: helmet {fp_h[4]:.2f}.  BRECQ keeps 92% of FP COCO AP,\nyet confidently calls the held-out helmet a cap.",
             fontsize=6.2, color=MUTED, va="center")

    def place(ax, dx=0.045):
        p = ax.get_position(); ax.set_position([p.x0 + dx, p.y0, p.width - dx, p.height]); return p.x0

    # (b) 층별 진단
    ax = fig.add_subplot(gs[0, 2]); x0 = place(ax)
    x = list(range(len(rows)))
    hs = [r["coco"] for r in rows]
    bars = ax.bar(x, hs, width=0.9, color=[COL["Ours"] if r["name"] in mt["psel"] else GRAY for r in rows], linewidth=0)
    for bb, r in zip(bars, rows):
        if r["name"] in gset:
            bb.set_facecolor("none"); bb.set_edgecolor(COL["Ours"]); bb.set_hatch("////"); bb.set_linewidth(0.7)
    ti = max(x, key=lambda i: hs[i])
    ax.annotate("text-fusion conv", (ti, hs[ti]), xytext=(4, -1), textcoords="offset points",
                fontsize=6.2, color=INK, va="top", ha="left")
    ax.text(0.98, 0.55, f"{BUDGET*100:.1f}% of weights\nhold {mt['capture']*100:.0f}%\nof the damage", transform=ax.transAxes,
            ha="right", va="center", fontsize=6.5, color=INK)
    ax.set_xlim(-1, len(rows)); ax.set_xticks([])
    ax.set_xlabel("layers of YOLO-World-S", fontsize=6.3, color=MUTED, labelpad=2)
    ax.set_ylabel("Δ flip (pp)", fontsize=6.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor="none", edgecolor=COL["Ours"], hatch="////", label="G"),
                       Patch(facecolor=COL["Ours"], label="P (8 bit)"), Patch(facecolor=GRAY, label="other")],
              loc="upper left", fontsize=5.8, frameon=False, handlelength=1.2,
              borderaxespad=0.1)
    fig.text(x0, 0.955, "(b) Where", fontsize=7.5, color=INK, weight="bold")

    # (c) W4A4, 6개 모델
    ax = fig.add_subplot(gs[0, 3]); x0 = place(ax)
    w = 0.26
    for j, (m, off) in enumerate((("BRECQ", -w), ("QDrop", 0.0), ("Ours", w))):
        vals = [row[2 + j] for row in W4A4]
        ax.bar([i + off for i in range(len(W4A4))], [max(v, 0.25) for v in vals], width=w * 0.92, color=COL[m],
               linewidth=0, label=m)
    for i, row in enumerate(W4A4):
        ax.plot([i - 1.5 * w, i + 1.5 * w], [row[1]] * 2, color=MUTED, linewidth=0.9, linestyle=(0, (2, 1.5)))
        ax.text(i + w, row[4] + 0.6, f"{row[4]:.0f}", ha="center", va="bottom", fontsize=5.8, color=INK)
    ax.set_xticks(range(len(W4A4))); ax.set_xticklabels([r[0] for r in W4A4], fontsize=6)
    ax.set_ylim(0, 35); ax.set_ylabel("held-out LVIS AP", fontsize=6.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    h, lab = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color=MUTED, linewidth=0.9, linestyle=(0, (2, 1.5)))); lab.append("FP32")
    ax.legend(h, lab, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, fontsize=5.8, frameon=False, handlelength=1.2,
              columnspacing=0.8, borderaxespad=0.1)
    fig.text(x0, 0.955, "(c) Fix (W4A4, 3 seeds)", fontsize=7.5, color=INK, weight="bold")
    fig.text(x0, 0.07, "BRECQ, QDrop ≤ 3.4 AP; Ours 11–23 AP,\ntraining-free, +1.6% weight memory.", fontsize=6.2,
             color=MUTED, va="center")
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300)
    print("->", args.out)


if __name__ == "__main__":
    main()
