"""어휘 크기별 flip (11-16 초안). runs/267_vocab_scale/w4a{8,5}_s0.json (scripts/vocab_scale.py) -> figures/experiments/vocab_scale.{pdf,png}
선 = 무작위 순서 3개 평균, 띠 = 최소~최대. 점선 = 비슷한 클래스부터(sim) / 먼 클래스부터(far) 더한 어휘.
사용(저장소 루트): .venv/bin/python figures/src/vocab_scale_fig.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COL = {"base": "#eb6834", "ours": "#2a78d6"}          # teaser_v2와 같은 색(BRECQ 주황, Ours 파랑)
NAME = {"base": "BRECQ", "ours": "BRECQ + Ours"}
INK, MUTED, GRID = "#0b0b0b", "#6b6a64", "#e1e0d9"
SIZES = [80, 150, 300, 600, 1203]


def val(V, key, n, m):
    k = "all80" if n == 80 else ("all1203" if n == 1203 else f"{key}_{n}")
    return V[k][f"flip_{m}"]


def main():
    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#c3c2b7", "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6))
    for ax, bits in zip(axes, ("w4a8", "w4a5")):
        V = json.load(open(os.path.join(ROOT, "runs", "267_vocab_scale", f"{bits}_s0.json")))["vocabs"]
        for m in ("base", "ours"):
            r = np.array([[val(V, f"rand{s}", n, m) for n in SIZES] for s in range(3)])
            ax.fill_between(SIZES, r.min(0), r.max(0), color=COL[m], alpha=0.18, linewidth=0)
            ax.plot(SIZES, r.mean(0), color=COL[m], lw=1.6, marker="o", ms=3.5, mec="white", mew=0.7, label=NAME[m] + " (random)")
            for o, ls, mk in (("sim", (0, (3, 1.5)), "^"), ("far", (0, (1, 1.5)), "v")):
                ax.plot(SIZES, [val(V, o, n, m) for n in SIZES], color=COL[m], lw=0.9, ls=ls, marker=mk, ms=3,
                        label=f"{NAME[m]} ({'similar' if o == 'sim' else 'dissimilar'} first)")
        ax.set_xscale("log"); ax.set_xticks(SIZES); ax.set_xticklabels([str(n) for n in SIZES])
        ax.minorticks_off(); ax.set_xlabel("vocabulary size", color=INK)
        ax.set_ylabel("flip (%)", color=INK); ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_ylim(0, None)
        ax.set_title(f"YOLO-World-S {bits.upper()}", loc="left", fontsize=7.5, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, "figures", "experiments", f"vocab_scale.{ext}"), dpi=250)
    print("-> figures/experiments/vocab_scale.{pdf,png}")


if __name__ == "__main__":
    main()
