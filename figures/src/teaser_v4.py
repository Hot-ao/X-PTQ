"""그림 1 후보 v4 (11-13): 무엇이 깨지나 -> 어디서·왜 -> 해결, 세 칸 모두 실측.

(a) 무엇이 깨지나: YOLO-World-S W4A8 seed 0, LVIS minival 450559. BRECQ는 helmet을 cap으로, Ours는 helmet 유지
    (teaser_v2/v3와 같은 검출, /data/taeho/cvpr2027_dets/qual_s_w4a8).
(b) 어디서·왜: 12.cv2 개입 실험(runs/257, n500.log). YOLO-World-S naive, 12.cv2만 W4 나머지 W8A8,
    평가 = train2017 정렬 1000번부터 500장(calibration과 겹치지 않음), LVIS 어휘 flip.
    3.30 (W8A8) -> 69.73 (12.cv2 W4) -> 17.74 (분기별 scale 분리) -> 9.75 (게이트를 conv 뒤로 = G).
(c) 해결: YOLO-World-S 비트별 LVIS AP, 논문 기준 seed 0(bits_curve.py와 같은 리더보드 집계).
사용(저장소 루트): .venv/bin/python figures/src/teaser_v4.py   (리더보드 로그를 읽어 1-2분)
출력: figures/teaser_v4.{pdf,png}
"""
from __future__ import annotations
import json, os, sys, statistics as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qual_figure import LVIS_ANN, COCO_ROOT, iou, top_per_location, draw
from teaser_v2 import DETS, crop_box, COL, INK, MUTED, GRID, AXIS

INTERV = [("W8A8", 3.30), ("12.cv2\nW4", 69.73), ("+split\nscales", 17.74), ("+G", 9.75)]
SEC = "#7f7f7f"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from PIL import Image
    from lvis import LVIS
    import bits_curve as bc

    # (c) 데이터: bits_curve와 같은 집계(seed 0)
    per, fp_lvis = bc.load()
    BITS = bc.BITS
    curve = {k: [st.mean(per[b][k].values()) if per[b][k] else float("nan") for b in BITS] for k in ("brecq", "qdrop", "ours")}
    print("FP", fp_lvis, {k: [round(v, 2) for v in vs] for k, vs in curve.items()}, [per[b]["_cfg"] for b in BITS])

    # (a) 데이터
    dets = {k: json.load(open(os.path.join(DETS, f"{k}.json"))) for k in ("fp", "brecq", "ours")}
    lv = LVIS(LVIS_ANN)
    names = {c["id"] - 1: c["name"].replace("_", " ") for c in lv.dataset["categories"]}
    iid = "450559"
    gt = [([a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]], a["category_id"] - 1)
          for a in lv.dataset["annotations"] if a["image_id"] == int(iid)]
    focus = [f for f in top_per_location(dets["fp"][iid], 0.5) if any(c == f[5] and iou(f[:4], b) >= 0.5 for b, c in gt)]
    helmet = [f for f in focus if names[f[5]] == "helmet"]
    img = Image.open(os.path.join(COCO_ROOT, "val2017", f"{int(iid):012d}.jpg")).convert("RGB")
    l, t, r, b = crop_box([f[:4] for f in focus], img.width, img.height, pad=0.12, aspect=0.8)
    img = img.crop((int(l), int(t), int(r), int(b)))
    shift = lambda ds: [[d[0] - l, d[1] - t, d[2] - l, d[3] - t] + list(d[4:]) for d in ds]

    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.edgecolor": AXIS, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(6.9, 1.95))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.85, 0.85, 1.45, 1.6], wspace=0.05, left=0.004, right=0.995, top=0.82,
                          bottom=0.19)

    def head(x, label, sec):
        t = fig.text(x, 0.955, label, fontsize=7.5, color=INK, weight="bold", va="center")
        fig.canvas.draw()
        bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        fig.text(bb.x1 + 0.006, 0.955, sec, fontsize=6.5, color=SEC, va="center")

    # (a)
    for i, (k, ttl) in enumerate([("brecq", "BRECQ"), ("ours", "Ours")]):
        ax = fig.add_subplot(gs[0, i])
        draw(ax, img, shift(dets[k][iid]), shift(helmet), names, 0.15, ttl, max_boxes=1)
        ax.title.set_fontsize(7)
        fig.canvas.draw(); rnd = fig.canvas.get_renderer(); inv = ax.transData.inverted()
        for tx in ax.texts:                      # 라벨이 이미지 밖으로 나가면 안으로
            bb = tx.get_window_extent(rnd); (xa, _), (xb, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
            tx.set_ha("left"); tx.set_x(min(max(0, xa), img.width - (xb - xa)))
    head(0.004, "(a) What breaks", "§3.1")
    fig.text(0.004, 0.06, "YOLO-World-S, W4A8, held-out LVIS words", fontsize=6.2, color=MUTED, va="center")

    def place(ax, dx):
        p = ax.get_position(); ax.set_position([p.x0 + dx, p.y0, p.width - dx, p.height]); return p.x0

    # (b) 개입 실험
    ax = fig.add_subplot(gs[0, 2]); x0 = place(ax, 0.05)
    vals = [v for _, v in INTERV]
    cols = [AXIS, "#d03b3b", "#9fb9dc", COL["Ours"]]
    ax.bar(range(4), vals, width=0.62, color=cols, linewidth=0)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=6.4, color=INK)
    ax.set_xticks(range(4)); ax.set_xticklabels([n for n, _ in INTERV], fontsize=6.0)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 80); ax.set_yticks([0, 25, 50, 75]); ax.set_ylabel("LVIS flip (%)", fontsize=6.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    head(x0, "(b) Where & why", "§3.4")

    # (c) 비트별 곡선
    ax = fig.add_subplot(gs[0, 3]); x0 = place(ax, 0.05)
    xs = list(range(len(BITS)))
    ax.axhline(fp_lvis, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    ax.text(len(BITS) - 1, fp_lvis + 0.7, "FP32", fontsize=6, color=MUTED, ha="right", va="bottom")
    for k, lab, lw, z in (("qdrop", "QDrop", 1.1, 2), ("brecq", "BRECQ", 1.1, 2), ("ours", "Ours", 2.0, 3)):
        ax.plot(xs, curve[k], color=COL["QDrop" if k == "qdrop" else "BRECQ" if k == "brecq" else "Ours"], linewidth=lw,
                marker="o", markersize=3 if k != "ours" else 3.8, markeredgecolor="white", markeredgewidth=0.6, zorder=z,
                label=lab)
    o, bq = curve["ours"][-1], curve["brecq"][-1]
    ax.annotate(f"{o:.1f}", (xs[-1], o), xytext=(-4, -4), textcoords="offset points", ha="right", va="top", fontsize=6.4, color=INK)
    ax.annotate("", (xs[-1] + 0.12, o), (xs[-1] + 0.12, bq + 0.3),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7, shrinkA=0, shrinkB=0))
    ax.text(xs[-1] + 0.2, (o + bq) / 2, f"+{o - bq:.0f}", fontsize=6.6, color=INK, va="center", weight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(BITS, fontsize=5.9)
    ax.set_xlim(-0.25, len(BITS) - 0.2); ax.set_ylim(0, 29)
    ax.set_ylabel("held-out LVIS AP", fontsize=6.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=6, frameon=False, handlelength=1.4, borderaxespad=0.1)
    head(x0, "(c) Fix", "§5.2")

    out = os.path.join(ROOT, "figures", "teaser_v4")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300)
    print("->", out)


if __name__ == "__main__":
    main()
