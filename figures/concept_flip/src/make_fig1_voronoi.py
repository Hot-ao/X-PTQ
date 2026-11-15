"""그림 1 후보 (11-13 수정): 문제(어휘가 커지면 같은 모델·같은 오차로 판정이 더 뒤집힌다) -> 해결(오차를 줄인다, 텍스트 임베딩은 그대로).

모두 YOLO-World-S W4A8 seed 0 전체 모델(BRECQ, Ours = brecq+GPM) 기준. 서론·초록의 수치와 같은 설정이다.
  칸 모양(텍스트 임베딩 위치)과 (a)의 클래스 이름은 설명용 합성값이다(실측 아님).
  flip 숫자(실측, 리더보드 seed 0, LVIS minival 전체): BRECQ COCO 어휘 3.94%, LVIS 어휘 17.49%; Ours LVIS 어휘 10.69%.
  원 반경 비율(실측, data/w4a8_emb_flip_n500.json, measure_w4a8.py): 영역 임베딩 오차 ||e||^2 BRECQ 0.0702, Ours 0.0373
    (L2 정규화 영역 임베딩의 FP 대비 제곱 거리, FP 확신 위치 가중 평균, LVIS minival 앞 500장) -> 반경 x sqrt(0.0373/0.0702) = 0.73.
  텍스트 임베딩은 head에서 8bit 상수(quant/attn_quant.py, 데이터 불필요)로 BRECQ와 Ours가 똑같다 -> (b)(c)의 칸은 같다.
  참고: 이 helmet 영역을 COCO 80 어휘로 채점하면 모든 클래스 점수가 0.002 이하(data/coco_scores_450559.json)라
  COCO 어휘에서는 배경이다. 그래서 (a)는 "small vocabulary" 개념도로 둔다.
(d) YOLO-World-S 비트별 LVIS AP, 논문 기준 seed 0(figures/src/bits_curve.py의 리더보드 집계 값).
사용(저장소 루트): .venv/bin/python figures/concept_flip/src/make_fig1_voronoi.py
출력: figures/concept_flip/fig1_voronoi.{pdf,png}(3칸), fig1_voronoi_ap.{pdf,png}(+ (d))
"""
from __future__ import annotations
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D
import make_concept as mc
from make_concept import INK, MUTED, GRID, AXIS, BLUE, RED, ORANGE, cells, nearest

EMB = {"brecq": 0.0702, "ours": 0.0373}
FLIP = {"coco": 3.94, "lvis": 17.49, "ours": 10.69}
FEAT = np.array([0.12, 0.02])
E = np.array([0.56, 0.40])                          # 방향은 cap 쪽(실측 사례: helmet -> cap)
R = float(np.linalg.norm(E))
SCALE_G = float(np.sqrt(EMB["ours"] / EMB["brecq"]))  # 0.73
SMALL = {"helmet": (0, 0), "person": (0.85, -0.85), "skateboard": (-1.9, 1.5), "dog": (-1.5, -1.9), "car": (2.2, 1.4)}
LARGE = dict(SMALL, **{"cap": (1.05, 0.75), "football helmet": (-0.72, 0.6), "hat": (1.45, -0.3),
                       "baseball cap": (0.45, 1.45), "gas mask": (-0.55, -0.85), "visor": (1.6, 0.95)})
BITS = ["W8A8", "W4A8", "W4A6", "W4A5", "W4A4"]
AP = {"BRECQ": [25.54, 22.75, 21.33, 11.29, 0.45], "QDrop": [25.37, 22.63, 20.10, 5.74, 0.27],
      "Ours": [25.78, 23.97, 23.66, 22.94, 18.60]}
FP_AP = 25.89
COLS = {"Ours": BLUE, "BRECQ": ORANGE, "QDrop": "#9a6bc9"}


def ball_arrow(ax, err, col, kept):
    q = FEAT + err
    ax.add_patch(Circle(FEAT, np.linalg.norm(err), fill=False, ls=(0, (2, 1.5)), ec=MUTED, lw=0.7, zorder=3))
    ax.add_patch(FancyArrowPatch(FEAT, q, arrowstyle="-|>", mutation_scale=6, color=col, lw=1.0, zorder=4))
    ax.plot(*FEAT, "o", ms=4.5, color=INK, zorder=5)
    ax.plot(*q, "o", ms=4.5, mfc="white", mec=col, mew=1.2, zorder=5)
    return q


def foot(ax, s, col):
    ax.set_xlabel(s, fontsize=6.6, color=col, labelpad=3)


def panels(axs):
    a, b, c = axs
    cells(a, SMALL)
    q = ball_arrow(a, E, BLUE, True); assert nearest(SMALL, q) == "helmet"
    a.set_title("(a) small vocabulary", loc="left", fontsize=7, color=INK)
    foot(a, f"80 classes: flip {FLIP['coco']:.1f}%", MUTED)
    cells(b, LARGE, bad="cap")
    q = ball_arrow(b, E, RED, False); assert nearest(LARGE, q) == "cap", nearest(LARGE, q)
    b.set_title("(b) open vocabulary", loc="left", fontsize=7, color=INK)
    foot(b, f"1,203 classes: flip {FLIP['lvis']:.1f}%", RED)
    cells(c, LARGE, bad="cap")
    q = ball_arrow(c, E * SCALE_G, BLUE, True); assert nearest(LARGE, q) == "helmet"
    c.add_patch(Circle(FEAT, R, fill=False, ls=(0, (1, 2)), ec=AXIS, lw=0.6, zorder=2))      # (b)의 원(비교용)
    c.set_title("(c) Ours: same cells, smaller error", loc="left", fontsize=7, color=INK)
    foot(c, f"‖e‖² ×{EMB['ours'] / EMB['brecq']:.2f}  →  flip {FLIP['ours']:.1f}%", BLUE)


def legend(fig, y, x=0.5, loc="lower center", n=5):
    h = [Line2D([], [], marker="o", ls="", ms=4.5, color=INK, label="FP region feature"),
         Line2D([], [], marker="o", ls="", ms=4.5, mfc="white", mec=BLUE, mew=1.2, label="quantized, top-1 kept"),
         Line2D([], [], marker="o", ls="", ms=4.5, mfc="white", mec=RED, mew=1.2, label="quantized, top-1 flipped"),
         Line2D([], [], marker="^", ls="", ms=3.5, color=MUTED, label="text embedding"),
         Line2D([], [], ls=(0, (2, 1.5)), color=MUTED, lw=0.7, label="quantization error ‖e‖")][:n]
    fig.legend(handles=h, loc=loc, bbox_to_anchor=(x, y), ncol=len(h), fontsize=6, frameon=False, handlelength=1.4,
               columnspacing=1.1)


def ap_panel(ax):
    xs = list(range(len(BITS)))
    ax.axhline(FP_AP, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(0, FP_AP + 0.6, "FP32", fontsize=6, color=MUTED, va="bottom")
    for k, lw, ms in (("QDrop", 1.1, 3), ("BRECQ", 1.1, 3), ("Ours", 2.0, 3.8)):
        ax.plot(xs, AP[k], color=COLS[k], lw=lw, marker="o", ms=ms, mec="white", mew=0.6, zorder=3 if k == "Ours" else 2,
                label=k)
    o, b = AP["Ours"][-1], AP["BRECQ"][-1]
    ax.annotate("", (xs[-1] + 0.15, o), (xs[-1] + 0.15, b + 0.3), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7,
                                                                                    shrinkA=0, shrinkB=0))
    ax.text(xs[-1] + 0.24, (o + b) / 2, f"+{o - b:.0f}", fontsize=6.6, color=INK, va="center", weight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(BITS, fontsize=5.8)
    ax.set_xlim(-0.25, len(BITS) + 0.15); ax.set_ylim(0, 29)
    ax.set_ylabel("held-out LVIS AP", fontsize=6.4)
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=5.9, frameon=False, handlelength=1.3, borderaxespad=0.1)
    ax.set_title("(d) result (YOLO-World-S)", loc="left", fontsize=7, color=INK)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(True)


# (d) W4A8 seed 0, LVIS flip(%): (기법 단독, 기법 + Ours). 구성(GP/GPM)은 seed 0 BRECQ 검사가 고른 것을 세 기법에 같이 쓴다.
# 출처: 리더보드(make_leaderboard collect -> best / pick_ours, seeds=(0,)), 11-13 조회. YOLOE의 GP는 실제로 P(G 해당 없음).
D_MODELS = ["S", "S-v2", "M", "L", "E-v8s", "E-11s"]
D_FLIP = {
    "BRECQ":    [(17.5, 10.7), (45.6, 27.1), (30.6, 22.4), (22.4, 9.9), (14.2, 10.4), (20.1, 18.0)],
    "QDrop":    [(18.6, 11.1), (42.8, 25.5), (31.7, 22.5), (32.9, 9.5), (14.6, 10.6), (20.1, 17.9)],
    "PD-Quant": [(24.0, 15.9), (45.7, 28.2), (38.2, 29.0), (36.7, 17.5), (19.9, 15.7), (27.0, 23.9)],
}


def flip_panel(fig, cell):
    sub = cell.subgridspec(1, 3, wspace=0.12)
    axs = [fig.add_subplot(sub[0, i]) for i in range(3)]
    y = list(range(len(D_MODELS)))[::-1]
    for ax, (base, vals) in zip(axs, D_FLIP.items()):
        for yy, (a, o) in zip(y, vals):
            ax.plot([a, o], [yy, yy], color=AXIS, lw=1.2, zorder=1)
            ax.plot(a, yy, "o", ms=3.6, mfc="white", mec=MUTED, mew=0.9, zorder=2)
            ax.plot(o, yy, "o", ms=3.8, color=BLUE, zorder=3)
        ax.set_xlim(0, 50); ax.set_xticks([0, 25, 50])
        ax.set_xticklabels(["0", "25", "50" if ax is axs[-1] else ""], fontsize=5.6)
        ax.set_ylim(-0.6, len(D_MODELS) - 0.4)
        ax.set_yticks(y); ax.set_yticklabels(D_MODELS if ax is axs[0] else [], fontsize=5.8)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_title(base, fontsize=6.4, color=INK, pad=2)
    axs[1].set_xlabel("LVIS flip (%), W4A8", fontsize=6.2, labelpad=2)
    return axs


def legend_d(fig, x, y):
    h = [Line2D([], [], marker="o", ls="", ms=3.6, mfc="white", mec=MUTED, mew=0.9, label="PTQ alone"),
         Line2D([], [], marker="o", ls="", ms=3.8, color=BLUE, label="+ Ours")]
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(x, y), ncol=2, fontsize=6, frameon=False, handlelength=1.0,
               columnspacing=1.0)



def cells_min(ax, anchors, show=("helmet", "cap"), bad=None, fs=5.6):
    """1단판: 칸은 모두 그리고 이름은 show만."""
    import matplotlib.colors as mcol
    names = list(anchors); P = np.array([anchors[n] for n in names])
    L = mc.LIM
    xs = np.linspace(L[0], L[1], 500); ys = np.linspace(L[2], L[3], 430)
    X, Y = np.meshgrid(xs, ys)
    lab = (((X[..., None] - P[:, 0]) ** 2 + (Y[..., None] - P[:, 1]) ** 2)).argmin(-1)
    col = np.zeros(lab.shape + (3,))
    for i, n in enumerate(names):
        col[lab == i] = mcol.to_rgb(mc.CELL_OK if n == "helmet" else (mc.CELL_BAD if n == bad else mc.CELL))
    ax.imshow(col, extent=L, origin="lower", interpolation="nearest", zorder=0)
    ax.contour(X, Y, lab, levels=np.arange(len(names)) + 0.5, colors=[AXIS], linewidths=0.5, zorder=1)
    for n, (x, y) in anchors.items():
        if L[0] < x < L[1] and L[2] < y < L[3]:
            ax.plot(x, y, marker="^", ms=2.6, color=MUTED, zorder=2)
            if n in show:
                ax.text(x, y - 0.14, n, ha="center", va="top", fontsize=fs, color=INK, zorder=2)
    ax.set_xlim(L[0], L[1]); ax.set_ylim(L[2], L[3]); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(AXIS); sp.set_linewidth(0.5)


def ball_arrow_min(ax, err, col):
    q = FEAT + err
    ax.add_patch(Circle(FEAT, np.linalg.norm(err), fill=False, ls=(0, (2, 1.5)), ec=MUTED, lw=0.6, zorder=3))
    ax.add_patch(FancyArrowPatch(FEAT, q, arrowstyle="-|>", mutation_scale=5, color=col, lw=0.9, zorder=4))
    ax.plot(*FEAT, "o", ms=3.4, color=INK, zorder=5)
    ax.plot(*q, "o", ms=3.4, mfc="white", mec=col, mew=1.0, zorder=5)
    return q


def one_column():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(3.3, 1.55))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.27, wspace=0.06)
    cells_min(a, SMALL); q = ball_arrow_min(a, E, BLUE); assert nearest(SMALL, q) == "helmet"
    cells_min(b, LARGE, bad="cap"); q = ball_arrow_min(b, E, RED); assert nearest(LARGE, q) == "cap"
    cells_min(c, LARGE, bad="cap")
    c.add_patch(Circle(FEAT, R, fill=False, ls=(0, (1, 2)), ec=AXIS, lw=0.5, zorder=2))
    q = ball_arrow_min(c, E * SCALE_G, BLUE); assert nearest(LARGE, q) == "helmet"
    for ax, t in ((a, "(a) 80 classes"), (b, "(b) 1,203 classes"), (c, "(c) + Ours")):
        ax.set_title(t, loc="left", fontsize=6.4, color=INK, pad=2)
    for ax, t, col in ((a, f"flip {FLIP['coco']:.1f}%", MUTED), (b, f"flip {FLIP['lvis']:.1f}%", RED),
                       (c, f"flip {FLIP['ours']:.1f}%", BLUE)):
        ax.set_xlabel(t, fontsize=6.2, color=col, labelpad=2)
    h = [Line2D([], [], marker="o", ls="", ms=3.4, color=INK, label="FP feature"),
         Line2D([], [], marker="o", ls="", ms=3.4, mfc="white", mec=BLUE, mew=1.0, label="quantized, kept"),
         Line2D([], [], marker="o", ls="", ms=3.4, mfc="white", mec=RED, mew=1.0, label="quantized, flipped"),
         Line2D([], [], marker="^", ls="", ms=2.6, color=MUTED, label="text embedding"),
         Line2D([], [], ls=(0, (2, 1.5)), color=MUTED, lw=0.6, label="error ‖e‖")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=5.4, frameon=False,
               handlelength=1.2, columnspacing=0.8, handletextpad=0.3, labelspacing=0.25)
    save(fig, "fig1_voronoi_1col")


def save(fig, stem):
    out = os.path.join(mc.OUT, stem)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300)
    plt.close(fig); print("->", out)


def main():
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.2, wspace=0.05)
    panels(axs); legend(fig, 0.0)
    save(fig, "fig1_voronoi")

    fig = plt.figure(figsize=(6.9, 2.15))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15], left=0.005, right=0.995, top=0.88, bottom=0.25, wspace=0.06)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    panels(axs)
    ax = fig.add_subplot(gs[0, 3])
    p = ax.get_position(); ax.set_position([p.x0 + 0.045, p.y0, p.width - 0.045, p.height])
    ap_panel(ax)
    legend(fig, 0.0, x=0.005, loc="lower left", n=5)
    save(fig, "fig1_voronoi_ap")

    fig = plt.figure(figsize=(6.9, 2.1))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.2, 1.5], left=0.005, right=0.99, top=0.86, bottom=0.27, wspace=0.06)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    panels(axs)
    axs[2].set_title("(c) Ours: smaller error", loc="left", fontsize=7, color=INK)
    dl = flip_panel(fig, gs[0, 4])
    x0 = axs[2].get_position().x1 + 0.012
    fig.text(x0, 0.955, "(d) 6 detectors, each PTQ + Ours", fontsize=7, color=INK, va="center")
    h = [Line2D([], [], marker="o", ls="", ms=4.5, color=INK, label="FP feature"),
         Line2D([], [], marker="o", ls="", ms=4.5, mfc="white", mec=BLUE, mew=1.2, label="quantized, kept"),
         Line2D([], [], marker="o", ls="", ms=4.5, mfc="white", mec=RED, mew=1.2, label="quantized, flipped"),
         Line2D([], [], marker="^", ls="", ms=3.5, color=MUTED, label="text embedding"),
         Line2D([], [], ls=(0, (2, 1.5)), color=MUTED, lw=0.7, label="error ‖e‖")]
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.005, 0.0), ncol=5, fontsize=6, frameon=False,
               handlelength=1.3, columnspacing=0.9)
    legend_d(fig, x0 + 0.02, 0.0)
    save(fig, "fig1_voronoi_flip")


if __name__ == "__main__":
    main()
    one_column()
