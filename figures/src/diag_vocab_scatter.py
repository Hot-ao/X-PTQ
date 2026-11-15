"""11-13 3.2 검증 그림: 층별 진단 Δ_ℓ을 COCO 어휘(진단에 쓰는 것)와 LVIS 어휘(검증 전용)로 잰 값의 산점도.

입력: runs/260_diag_vocab_check/{yolov8s-world,yoloe-v8s-seg}.json (scripts/diag_vocab_check.py)
점 하나 = conv 그룹 하나. 진한 점 = COCO 진단이 1.5% 예산으로 고른 보호 층(LVIS로 골라도 같다).
축은 symlog(0 근처는 선형): 12.cv2 하나가 다른 층보다 10배 커서 선형 축이면 나머지가 원점에 뭉친다.
출력: figures/diag_vocab_scatter.{pdf,png}
"""
import json, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "core"))
import diag_w4_sensitivity as dg

RUN = os.path.join(ROOT, "runs", "260_diag_vocab_check")
MODELS = [("yolov8s-world", "YOLO-World-S (text gate)"), ("yoloe-v8s-seg", "YOLOE-v8s (no gate)")]
BLUE, GRAY, INK, MUTED, GRID, AXIS = "#2a78d6", "#a8a79f", "#0b0b0b", "#6b6a64", "#e1e0d9", "#c3c2b7"


def spearman(a, b):
    ra = np.argsort(np.argsort(-np.asarray(a))); rb = np.argsort(np.argsort(-np.asarray(b)))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.edgecolor": AXIS, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6))
    for ax, (stem, title) in zip(axes, MODELS):
        rows = json.load(open(os.path.join(RUN, f"{stem}.json")))["rows"]
        sel = set(dg.select_protected(rows, 0.015, "coco")[0])
        x = np.array([r["coco"] for r in rows]); y = np.array([r["lvis"] for r in rows])
        prot = np.array([r["name"] in sel for r in rows])
        lim = (min(x.min(), y.min(), 0) - 0.3, max(x.max(), y.max()) * 1.6)
        ax.set_xscale("symlog", linthresh=1.0); ax.set_yscale("symlog", linthresh=1.0)
        ax.plot([0, lim[1]], [0, lim[1]], color=GRID, linewidth=0.8, zorder=1)          # y = x (같은 크기)
        ax.scatter(x[~prot], y[~prot], s=14, color=GRAY, edgecolors="white", linewidths=0.6, zorder=2)
        ax.scatter(x[prot], y[prot], s=22, color=BLUE, edgecolors="white", linewidths=0.8, zorder=3)
        top = sorted(rows, key=lambda r: r["coco"], reverse=True)[:2]
        for r in top:
            ax.annotate((r["name"][:-5] if r["name"].count(".") > 1 else r["name"]), (r["coco"], r["lvis"]), xytext=(-4, 4),
                        textcoords="offset points", ha="right", fontsize=6.3, color=INK)
        rho = spearman(x, y)
        ax.text(0.97, 0.04, f"Spearman ρ = {rho:.2f}\nprotected set: identical\nin both vocabularies",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4, color=INK, linespacing=1.3)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(title, loc="left", fontsize=7.5, color=INK)
        ax.set_xlabel("Δflip with COCO vocab. (%p, used for diagnosis)")
        ax.set_ylabel("Δflip with LVIS vocab. (%p, check only)")
        ax.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
        print(f"{stem}: rho={rho:.3f}, n={len(rows)}, protected={sorted(sel)}")
    h = [Line2D([], [], marker="o", linestyle="none", color=BLUE, markeredgecolor="white", markersize=5,
                label="protected by P (1.5% weight budget)"),
         Line2D([], [], marker="o", linestyle="none", color=GRAY, markeredgecolor="white", markersize=4.5, label="other conv layers"),
         Line2D([], [], color=GRID, linewidth=0.8, label="y = x")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=6.4, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=2.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, "figures", f"diag_vocab_scatter.{ext}"), dpi=300)
    print("-> figures/diag_vocab_scatter.{pdf,png}")


if __name__ == "__main__":
    main()
