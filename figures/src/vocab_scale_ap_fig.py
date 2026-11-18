"""어휘 크기별 flip과 AP (11-16 v2). runs/267_vocab_scale/{,ap_}w4a{8,5}_s0.json -> figures/experiments/vocab_scale_ap.{pdf,png}
주장: 같은 비트 안에서 어휘가 커지면 flip은 늘지만 FP와의 AP 차이는 거의 그대로다(AP로는 안 보인다). 저비트(W4A5)에서는
flip이 훨씬 크고 BRECQ AP도 무너지는데, Ours는 FP 가까이를 지킨다.
위 행 = flip, 아래 행 = AP_80(어휘가 커져도 처음 80 클래스로 채점하는 LVIS AP). 행마다 두 비트의 세로축을 같게 둔다.
선 = 무작위 순서 3개 평균, 띠 = 최소~최대. 가로축은 실제 간격(선형). 화살표 = FP와 BRECQ의 AP 차이(80, 1,203).
(v1: AP_V 점선 포함, scratchpad/vocab_scale_ap_fig.v1.py)
사용(저장소 루트): .venv/bin/python figures/src/vocab_scale_ap_fig.py
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


def series(V, f):
    ks = lambda n: ["all80"] if n == 80 else (["all1203"] if n == 1203 else [f"rand{s}_{n}" for s in range(3)])
    r = [[V[k][f] for k in ks(n)] for n in SIZES]
    return np.array([np.mean(x) for x in r]), np.array([min(x) for x in r]), np.array([max(x) for x in r])


def line(ax, mu, lo, hi, m):
    ax.fill_between(SIZES, lo, hi, color=COL[m], alpha=0.18, lw=0)
    ax.plot(SIZES, mu, color=COL[m], lw=1.6, marker="o", ms=3.5, mec="white", mew=0.7, label=NAME[m])


def gap(ax, x, top, bot, dx):
    ax.annotate("", (x + dx, bot), (x + dx, top), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7, shrinkA=0, shrinkB=0))
    ax.text(x + dx + (12 if dx > 0 else -12), (top + bot) / 2, f"{top - bot:.1f}", fontsize=6.5, color=INK,
            ha="left" if dx > 0 else "right", va="center")


def main():
    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#c3c2b7", "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.4), sharex=True, sharey="row")
    for j, bits in enumerate(("w4a8", "w4a5")):
        F = json.load(open(os.path.join(ROOT, "runs", "267_vocab_scale", f"{bits}_s0.json")))["vocabs"]
        A = json.load(open(os.path.join(ROOT, "runs", "267_vocab_scale", f"ap_{bits}_s0.json")))["vocabs"]
        # 위: flip
        ax = axes[0, j]
        for m in ("base", "ours"):
            mu, lo, hi = series(F, f"flip_{m}")
            line(ax, mu, lo, hi, m)
            for i, ha in ((0, "right"), (-1, "left")):
                ax.text(SIZES[i] + (-25 if i == 0 else 25), mu[i], f"{mu[i]:.1f}", fontsize=6.5, color=INK, ha=ha, va="center")
        ax.set_title(f"YOLO-World-S {bits.upper()}", loc="left", fontsize=8, color=INK, fontweight="bold")
        # 아래: AP_80, FP와의 차이 표시
        ax = axes[1, j]
        mus = {}
        for m in ("fp", "ours", "base"):
            mu, lo, hi = series(A, f"AP80_{m}")
            line(ax, mu, lo, hi, m); mus[m] = mu
        for i, dx in ((0, -40), (-1, 40)):
            gap(ax, SIZES[i], mus["fp"][i], mus["base"][i], dx)
    axes[0, 0].set_ylabel("flip (%)  ↓", color=INK)
    axes[1, 0].set_ylabel("LVIS AP$_{80}$  ↑\n(same 80 classes scored)", color=INK)
    axes[0, 0].set_ylim(0, 70); axes[1, 0].set_ylim(20, 52)
    for ax in axes.flat:
        ax.set_xticks(SIZES); ax.set_xlim(-60, 1350)
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xticklabels([str(n) for n in SIZES]); ax.set_xlabel("vocabulary size", color=INK)
        tl = ax.get_xticklabels(); tl[0].set_ha("right"); tl[1].set_ha("left")
    h, l = axes[1, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, "figures", "experiments", f"vocab_scale_ap.{ext}"), dpi=250, bbox_inches="tight", pad_inches=1 / 72)
    print("-> figures/experiments/vocab_scale_ap.{pdf,png}")


if __name__ == "__main__":
    main()
