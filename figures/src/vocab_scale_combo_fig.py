"""어휘 크기별 flip(막대) + AP_80(선) 한 그래프 (11-16 시안). runs/267_vocab_scale/{,ap_}w4a{8,5}_s0.json -> figures/experiments/vocab_scale_combo.{pdf,png}
막대 = flip(왼쪽 축), 선 = 처음 80 클래스로 채점한 LVIS AP(오른쪽 축). 비트마다 패널 하나, 두 패널의 축 범위는 같다.
막대를 쓰려고 가로축은 어휘 크기를 같은 간격(범주)으로 둔다. 값 = 무작위 순서 3개 평균.
사용(저장소 루트): .venv/bin/python figures/src/vocab_scale_combo_fig.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COL = {"fp": "#6b6a64", "base": "#eb6834", "ours": "#2a78d6"}
NAME = {"fp": "FP32", "base": "BRECQ", "ours": "BRECQ + Ours"}
INK, MUTED, GRID = "#0b0b0b", "#6b6a64", "#e1e0d9"
SIZES = [80, 150, 300, 600, 1203]


def mean(V, f):
    ks = lambda n: ["all80"] if n == 80 else (["all1203"] if n == 1203 else [f"rand{s}_{n}" for s in range(3)])
    return np.array([np.mean([V[k][f] for k in ks(n)]) for n in SIZES])


def main():
    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.edgecolor": "#c3c2b7", "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.7))
    x = np.arange(len(SIZES)); bw = 0.34
    for j, (ax, bits) in enumerate(zip(axes, ("w4a8", "w4a5"))):
        F = json.load(open(os.path.join(ROOT, "runs", "267_vocab_scale", f"{bits}_s0.json")))["vocabs"]
        A = json.load(open(os.path.join(ROOT, "runs", "267_vocab_scale", f"ap_{bits}_s0.json")))["vocabs"]
        for k, (m, off) in enumerate((("base", -bw / 2), ("ours", bw / 2))):
            v = mean(F, f"flip_{m}")
            ax.bar(x + off, v, bw * 0.92, color=COL[m], alpha=0.35, edgecolor=COL[m], lw=0.6, label=f"{NAME[m]} flip (bar)")
            for xi, vi in zip(x + off, v):
                ax.text(xi, vi + 1, f"{vi:.0f}", ha="center", va="bottom", fontsize=5.5, color=MUTED)
        ax.set_ylim(0, 80); ax.set_xticks(x); ax.set_xticklabels([str(n) for n in SIZES])
        ax.set_xlabel("vocabulary size", color=INK); ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_title(f"YOLO-World-S {bits.upper()}", loc="left", fontsize=8, color=INK, fontweight="bold")
        ax2 = ax.twinx()
        for m in ("fp", "ours", "base"):
            ax2.plot(x, mean(A, f"AP80_{m}"), color=COL[m], lw=1.6, marker="o", ms=3.5, mec="white", mew=0.7,
                     label=f"{NAME[m]} AP$_{{80}}$ (line)")
        ax2.set_ylim(0, 55); ax2.spines["top"].set_visible(False)
        if j == 0:
            ax.set_ylabel("flip (%)  ↓  [bars]", color=INK)
        else:
            ax2.set_ylabel("LVIS AP$_{80}$  ↑  [lines]", color=INK)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, fontsize=6.5, frameon=False, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, "figures", "experiments", f"vocab_scale_combo.{ext}"), dpi=250, bbox_inches="tight", pad_inches=1 / 72)
    print("-> figures/experiments/vocab_scale_combo.{pdf,png}")


if __name__ == "__main__":
    main()
