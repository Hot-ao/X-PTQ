"""어휘 크기별 flip(막대) + FP 대비 AP 하락(선), 두 비트를 한 그래프에 (11-16 시안).
runs/267_vocab_scale/{,ap_}w4a{8,5}_s0.json -> figures/experiments/vocab_scale_one.{pdf,png}
색 = 방법(BRECQ 주황, Ours 파랑), 비트 = 막대 채움(W4A8 옅게 / W4A5 진하게)과 선 모양(W4A8 점선·빈 점 / W4A5 실선).
선 = ΔAP_80 = FP AP_80 - 양자화 AP_80(처음 80 클래스로 채점). 둘 다 낮을수록 좋다. 값 = 무작위 순서 3개 평균.
사용(저장소 루트): .venv/bin/python figures/src/vocab_scale_one_fig.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COL = {"base": "#eb6834", "ours": "#2a78d6"}
NAME = {"base": "BRECQ", "ours": "BRECQ + Ours"}
INK, MUTED, GRID = "#0b0b0b", "#6b6a64", "#e1e0d9"
SIZES = [80, 150, 300, 600, 1203]


def mean(V, f):
    ks = lambda n: ["all80"] if n == 80 else (["all1203"] if n == 1203 else [f"rand{s}_{n}" for s in range(3)])
    return np.array([np.mean([V[k][f] for k in ks(n)]) for n in SIZES])


def main():
    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.edgecolor": "#c3c2b7", "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax2 = ax.twinx(); ax2.spines["top"].set_visible(False)
    ax.set_zorder(ax2.get_zorder() - 1); ax.patch.set_visible(False)   # 선이 막대 위에
    x = np.arange(len(SIZES)); bw = 0.34
    # 11-16 v2: 방법마다 막대 하나. 아래 진한 부분 = W4A8 flip, 막대 끝 = W4A5 flip(합이 아니라 겹친 것: W4A5 >= W4A8)
    F = {b: json.load(open(os.path.join(ROOT, "runs", "267_vocab_scale", f"{b}_s0.json")))["vocabs"] for b in ("w4a8", "w4a5")}
    A = {b: json.load(open(os.path.join(ROOT, "runs", "267_vocab_scale", f"ap_{b}_s0.json")))["vocabs"] for b in ("w4a8", "w4a5")}
    hb, lb, hl, ll = [], [], [], []
    for m, off in (("base", -0.5), ("ours", 0.5)):
        f8, f5 = mean(F["w4a8"], f"flip_{m}"), mean(F["w4a5"], f"flip_{m}")
        assert (f5 >= f8).all()
        b5 = ax.bar(x + off * bw, f5, bw * 0.92, color=COL[m], alpha=0.3, edgecolor=COL[m], lw=0.6)
        b8 = ax.bar(x + off * bw, f8, bw * 0.92, color=COL[m], alpha=0.95, edgecolor=COL[m], lw=0.6)
        for xi, v8, v5 in zip(x + off * bw, f8, f5):
            ax.text(xi, v5 + 0.8, f"{v5:.0f}", ha="center", va="bottom", fontsize=5.5, color=MUTED)
            if v8 >= 4:
                ax.text(xi, v8 - 0.6, f"{v8:.0f}", ha="center", va="top", fontsize=5.5, color="white")
        hb += [b8, b5]; lb += [f"{NAME[m]} W4A8", f"{NAME[m]} W4A5"]
        for bits, ls, mfc in (("w4a8", (0, (2.5, 2.5)), "white"), ("w4a5", "-", None)):
            d = mean(A[bits], "AP80_fp") - mean(A[bits], f"AP80_{m}")
            (l,) = ax2.plot(x, d, color=COL[m], lw=1.4, ls=ls, marker="o", ms=3.5, mfc=mfc or COL[m],
                            mec=COL[m] if mfc else "white", mew=0.8, zorder=3 if bits == "w4a8" else 2)
            hl.append(l); ll.append(f"{NAME[m]} {bits.upper()}")
    ax.set_ylim(0, 70); ax2.set_ylim(-14, 25)   # 선을 막대 아래쪽 숫자 위로 올린다(오른쪽 축은 0~25만 표시)
    ax2.set_yticks(range(0, 26, 5)); ax2.spines["right"].set_bounds(0, 25)
    ax.set_xticks(x); ax.set_xticklabels([str(n) for n in SIZES]); ax.set_xlabel("vocabulary size", color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_ylabel("flip (%) [bars]  \u2193", color=INK)
    ax2.set_ylabel("AP$_{80}$ drop from FP32 (pt) [lines]  \u2193", color=INK)
    fig.legend(hb, lb, title="flip (bar: dark = W4A8, full = W4A5)", title_fontsize=6.3, fontsize=6.2, frameon=False,
               loc="lower left", ncol=1, bbox_to_anchor=(0.0, 0.0))
    fig.legend(hl, ll, title="AP$_{80}$ drop (line)", title_fontsize=6.3, fontsize=6.2, frameon=False,
               loc="lower right", ncol=1, bbox_to_anchor=(1.0, 0.0), handlelength=3)
    fig.tight_layout(rect=(0, 0.25, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, "figures", "experiments", f"vocab_scale_one.{ext}"), dpi=250, bbox_inches="tight", pad_inches=1 / 72)
    print("-> figures/experiments/vocab_scale_one.{pdf,png}")


if __name__ == "__main__":
    main()
