"""개념 그림 세 가지 (11-13): 점수 오차를 줄이는 PTQ와 판정(top-1)을 지키는 우리 방법의 차이.

A. 결정 경계(개념, 합성 좌표): 텍스트 임베딩 칸(Voronoi). (a) 작은 어휘, (b) 큰 어휘에서 같은 오차 e,
   (c) 오차를 층별 성분으로 나누면 한 성분(진단이 찾는 층)이 크다 -> 그것을 제거하면 칸 안에 남는다.
   좌표는 설명용 합성값이다(실측 아님).
B. 실측 점수: YOLO-World-S W4A8 seed 0, LVIS minival 450559의 helmet 앵커에서 FP / BRECQ / Ours의 클래스별 점수
   (data/scores_450559_helmet.json, dump_scores.py).
C. A의 (a)(b) + B를 한 줄로.
사용(저장소 루트): .venv/bin/python figures/concept_flip/src/make_concept.py
출력: figures/concept_flip/concept_{A,B,C}.{pdf,png}
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
INK, MUTED, GRID, AXIS = "#0b0b0b", "#6b6a64", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#d03b3b"
CELL_OK, CELL_BAD, CELL = "#dbe8f8", "#f6d9d6", "#f3f2ee"
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.edgecolor": AXIS, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "mathtext.fontset": "dejavusans"})

# ---- A: 합성 좌표
FEAT = np.array([0.12, 0.02])                 # FP 영역 특징
E_DOM = np.array([0.46, 0.33])                # 진단이 찾는 층(텍스트 융합 conv)의 오차 성분
E_REST = [np.array(v) for v in ([0.05, 0.03], [0.04, -0.03], [0.03, 0.04], [-0.02, 0.03])]   # 나머지 층들
E = E_DOM + sum(E_REST)
SMALL = {"helmet": (0, 0), "person": (2.3, 1.2), "skateboard": (-1.9, 1.5), "dog": (-1.4, -1.9), "car": (2.0, -1.7)}
LARGE = dict(SMALL, **{"cap": (1.0, 0.72), "football helmet": (-0.85, 0.55), "hat": (1.35, -0.35),
                       "baseball cap": (0.45, 1.45), "gas mask": (-0.45, -0.95), "visor": (1.6, 0.9)})
LIM = (-1.6, 1.9, -1.25, 1.75)


def cells(ax, anchors, ok="helmet", bad=None):
    names = list(anchors); P = np.array([anchors[n] for n in names])
    xs = np.linspace(LIM[0], LIM[1], 700); ys = np.linspace(LIM[2], LIM[3], 600)
    X, Y = np.meshgrid(xs, ys)
    d = ((X[..., None] - P[:, 0]) ** 2 + (Y[..., None] - P[:, 1]) ** 2)
    lab = d.argmin(-1)
    col = np.zeros(lab.shape + (3,))
    for i, n in enumerate(names):
        c = CELL_OK if n == ok else (CELL_BAD if n == bad else CELL)
        col[lab == i] = matplotlib.colors.to_rgb(c)
    ax.imshow(col, extent=LIM, origin="lower", interpolation="nearest", zorder=0)
    ax.contour(X, Y, lab, levels=np.arange(len(names)) + 0.5, colors=[AXIS], linewidths=0.6, zorder=1)
    for n, (x, y) in anchors.items():
        if LIM[0] < x < LIM[1] and LIM[2] < y < LIM[3]:
            ax.plot(x, y, marker="^", ms=3.5, color=MUTED, zorder=2)
            ax.text(x, y - 0.12, n, ha="center", va="top", fontsize=5.8, color=INK if n in (ok, bad) else MUTED, zorder=2)
    ax.set_xlim(LIM[0], LIM[1]); ax.set_ylim(LIM[2], LIM[3]); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(AXIS); s.set_linewidth(0.6)


def nearest(anchors, p):
    return min(anchors, key=lambda n: np.sum((np.array(anchors[n]) - p) ** 2))


def feat_and_err(ax, err, col, ball=True):
    q = FEAT + err
    if ball:
        ax.add_patch(Circle(FEAT, np.linalg.norm(E), fill=False, ls=(0, (2, 1.5)), ec=MUTED, lw=0.7, zorder=3))
    ax.add_patch(FancyArrowPatch(FEAT, q, arrowstyle="-|>", mutation_scale=6, color=col, lw=1.0, zorder=4))
    ax.plot(*FEAT, "o", ms=4.5, color=INK, zorder=5)
    ax.plot(*q, "o", ms=4.5, mfc="white", mec=col, mew=1.2, zorder=5)
    return q


def panel_a(ax):
    cells(ax, SMALL)
    q = feat_and_err(ax, E, BLUE)
    assert nearest(SMALL, q) == "helmet"
    ax.set_title("(a) calibration-size vocabulary", loc="left", fontsize=7, color=INK)
    ax.set_xlabel("top-1 kept", fontsize=6.4, color=BLUE, labelpad=3)


def panel_b(ax):
    cells(ax, LARGE, bad="cap")
    q = feat_and_err(ax, E, RED)
    assert nearest(LARGE, q) == "cap", nearest(LARGE, q)
    ax.set_title("(b) open vocabulary (LVIS)", loc="left", fontsize=7, color=INK)
    ax.set_xlabel(r"same $\|e\|$ $\rightarrow$ top-1 flips", fontsize=6.4, color=RED, labelpad=3)


def panel_c(ax):
    cells(ax, LARGE, bad="cap")
    # BRECQ: 성분을 이어 그린다(작은 회색 = 나머지 층, 큰 화살표 = 진단된 층)
    p = FEAT.copy()
    for e in E_REST:
        ax.add_patch(FancyArrowPatch(p, p + e, arrowstyle="-|>", mutation_scale=4, color=MUTED, lw=0.7, zorder=4)); p = p + e
    ax.add_patch(FancyArrowPatch(p, p + E_DOM, arrowstyle="-|>", mutation_scale=6, color=RED, lw=1.0, ls=(0, (3, 1.5)),
                                 zorder=4))
    ax.plot(*(p + E_DOM), "x", ms=5, color=RED, mew=1.2, zorder=5)
    ax.text(*(p + E_DOM / 2 + np.array([-0.12, 0.13])), r"$e_{\ell^\star}$", fontsize=7.5, color=RED, ha="center")
    q = FEAT + sum(E_REST)
    assert nearest(LARGE, q) == "helmet"
    ax.plot(*FEAT, "o", ms=4.5, color=INK, zorder=5)
    ax.plot(*q, "o", ms=4.5, mfc="white", mec=BLUE, mew=1.2, zorder=6)
    ax.set_title(r"(c) remove the diagnosed component", loc="left", fontsize=7, color=INK)
    ax.set_xlabel(r"$\ell^\star$ = layer with the largest $\Delta_\ell$  $\rightarrow$  G / P", fontsize=6.4, color=BLUE,
                  labelpad=3)


def legend_a(fig, y):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", ms=4.5, color=INK, label="FP region feature"),
         Line2D([], [], marker="o", ls="", ms=4.5, mfc="white", mec=BLUE, mew=1.2, label="quantized (top-1 kept)"),
         Line2D([], [], marker="o", ls="", ms=4.5, mfc="white", mec=RED, mew=1.2, label="quantized (top-1 flipped)"),
         Line2D([], [], marker="^", ls="", ms=3.5, color=MUTED, label="text embedding"),
         Line2D([], [], ls=(0, (2, 1.5)), color=MUTED, lw=0.7, label=r"error ball $\|e\|$")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, y), ncol=5, fontsize=6, frameon=False, handlelength=1.4,
               columnspacing=1.2)


# ---- B: 실측 점수
def load_scores(n=6):
    d = json.load(open(os.path.join(OUT, "data", "scores_450559_helmet.json")))
    short = {"cap/cap headwear": "cap (headwear)", "baseball cap/jockey cap/golf cap": "baseball cap",
             "gasmask/respirator/gas helmet": "gas mask"}
    return d, [short.get(c, c) for c in d["classes"][:n]], {k: v[:n] for k, v in d["scores"].items()}


def panel_scores(axes, titles=("FP32", "BRECQ", "Ours")):
    d, cls, sc = load_scores()
    y = np.arange(len(cls))[::-1]
    for ax, key, ttl in zip(axes, ("fp", "brecq", "ours"), titles):
        v = np.array(sc[key]); top = int(v.argmax())
        flipped = cls[top] != "helmet"
        cols = [(RED if flipped else BLUE) if i == top else ("#9fb9dc" if cls[i] == "helmet" else AXIS) for i in range(len(cls))]
        ax.barh(y, v, height=0.62, color=cols, linewidth=0)
        for i in range(len(cls)):
            ax.text(v[i] + 0.02, y[i], f"{v[i]:.2f}", va="center", fontsize=5.8, color=INK if i == top else MUTED)
        ax.set_xlim(0, 1.22); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", ".5", "1"], fontsize=5.8)
        ax.set_yticks(y); ax.set_yticklabels(cls if ax is axes[0] else [""] * len(cls), fontsize=6.2)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
        ax.set_title(ttl, loc="left", fontsize=7, color=INK, pad=11)
        ax.text(0, 1.015, cls[top].split(" (")[0] + ("  ✗" if flipped else "  ✓"), transform=ax.transAxes, fontsize=6.0,
                color=RED if flipped else BLUE, va="bottom")
    return d


def fig_A():
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.2, wspace=0.05)
    panel_a(axs[0]); panel_b(axs[1]); panel_c(axs[2])
    legend_a(fig, 0.0)
    save(fig, "concept_A")


def fig_B():
    fig, axs = plt.subplots(1, 3, figsize=(5.0, 1.9))
    fig.subplots_adjust(left=0.2, right=0.98, top=0.8, bottom=0.2, wspace=0.12)
    d = panel_scores(axs)
    fig.text(0.2, 0.02, f"YOLO-World-S W4A8, LVIS vocabulary ({d['n_vocab']} classes), scores at the FP helmet anchor",
             fontsize=5.8, color=MUTED)
    save(fig, "concept_B")


def fig_C():
    fig = plt.figure(figsize=(6.9, 2.35))
    gl = fig.add_gridspec(1, 2, left=0.005, right=0.53, top=0.86, bottom=0.2, wspace=0.05)
    gr = fig.add_gridspec(1, 3, left=0.665, right=0.99, top=0.8, bottom=0.2, wspace=0.12)
    a1, a2 = fig.add_subplot(gl[0, 0]), fig.add_subplot(gl[0, 1])
    panel_a(a1); panel_b(a2)
    bs = [fig.add_subplot(gr[0, i]) for i in range(3)]
    panel_scores(bs)
    fig.text(0.55, 0.955, "(c) measured: helmet region, W4A8", fontsize=7, color=INK)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", ms=4.5, color=INK, label="FP feature"),
         Line2D([], [], marker="o", ls="", ms=4.5, mfc="white", mec=BLUE, mew=1.2, label="quantized, kept"),
         Line2D([], [], marker="o", ls="", ms=4.5, mfc="white", mec=RED, mew=1.2, label="quantized, flipped"),
         Line2D([], [], marker="^", ls="", ms=3.5, color=MUTED, label="text embedding")]
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=4, fontsize=6, frameon=False,
               handlelength=1.2, columnspacing=1.0)
    save(fig, "concept_C")


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), dpi=300)
    plt.close(fig)
    print("->", os.path.join(OUT, stem))


if __name__ == "__main__":
    fig_A(); fig_B(); fig_C()
