"""비트별 성능 곡선 (11-03, 논문 그림 5 후보): activation 비트를 내릴 때 기존 PTQ는 무너지고 우리 방법은 버틴다.

YOLO-World-S, LVIS AP. 값은 리더보드의 주요 표 A(results/LEADERBOARD.html에 박힌 데이터의 fixed.t1)를 그대로 읽는다.
그래서 리더보드가 갱신되면(seed가 쌓이면) 다시 실행만 하면 된다. 칸의 seed 수는 출력과 그림 제목에 적는다.
  - Ours (BRECQ) = 표 A의 "Ours" 행(seed마다 calibration 밖 flip으로 GP/GPM 선택, rho = 1.0)
  - LVIS AP < 15인 칸은 속이 빈 점으로 표시한다(기준선 seed 붕괴. 11-03 기준 BRECQ W4A5 seed 0)
그림: figures/bits_curve.{pdf,png}
사용(저장소 루트): .venv/bin/python figures/src/bits_curve.py
"""
from __future__ import annotations
import json, os, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BITS = ["W8A8", "W4A8", "W4A6", "W4A5", "W4A4"]
COLLAPSE = 15.0
INK, MUTED, GRID, AXIS = "#0b0b0b", "#6b6a64", "#e1e0d9", "#c3c2b7"
SERIES = [  # (표 A의 cond, 표시 이름, 색, 선 모양, 굵기)
    ("brecq", "BRECQ", "#eb6834", "-", 1.3),
    ("qdrop", "QDrop", "#1baf7a", "-", 1.3),
    ("brecq+S", "BRECQ + SmoothQuant", "#eda100", "-", 1.3),
    ("brecq+W", "BRECQ + AWQ", "#e87ba4", "-", 1.3),
    ("qdrop+GPM", "Ours (on QDrop)", "#2a78d6", (0, (4, 2)), 1.6),
    ("ours", "Ours (on BRECQ)", "#2a78d6", "-", 2.0),
]


def load():
    h = open(os.path.join(ROOT, "results", "LEADERBOARD.html"), encoding="utf-8").read()
    D = json.loads(re.search(r"const D = (\{.*?\});\n", h, re.S).group(1))
    rows = {r["cond"]: r["cells"] for r in D["fixed"]["t1"]}
    return D["generated"], rows


def main():
    gen, rows = load()
    fp = rows["FP32"]["W8A8"]["lvis"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.edgecolor": AXIS, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.axhline(fp, color=MUTED, linewidth=1.0, linestyle=(0, (3, 2)), zorder=1)
    ax.text(len(BITS) - 1 + 0.25, fp + 0.4, f"FP32 {fp:.1f}", fontsize=6.3, color=MUTED, ha="right", va="bottom")
    ns = set()
    print(f"리더보드 {gen}")
    for cond, name, col, ls, lw in SERIES:
        cells = rows.get(cond, {})
        xs, ys = [], []
        for i, b in enumerate(BITS):
            c = cells.get(b)
            if c and c.get("lvis") is not None:
                xs.append(i); ys.append(c["lvis"]); ns.add(c["n"])
        print(f"{name:22s} " + "  ".join(f"{BITS[x]} {y:5.2f}" for x, y in zip(xs, ys)))
        ax.plot(xs, ys, color=col, linestyle=ls, linewidth=lw, zorder=4 if cond == "ours" else 3)
        for x, y in zip(xs, ys):      # 기준선이 무너진 칸은 속 빈 점(우리 방법 행은 표시하지 않는다)
            hollow = y < COLLAPSE and cond not in ("ours", "qdrop+GPM")
            ax.plot([x], [y], "o", markersize=4.2, zorder=5, color=col, markerfacecolor="white" if hollow else col,
                    markeredgecolor=col if hollow else "white", markeredgewidth=0.9)
        if cond == "ours":            # 직접 표기: 우리 방법만
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=6.2, color=INK)
    ax.set_xticks(range(len(BITS))); ax.set_xticklabels(BITS, fontsize=6.8)
    ax.set_xlim(-0.3, len(BITS) - 0.7); ax.set_ylim(0, 29)
    ax.set_ylabel("LVIS AP (held-out vocabulary)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    seeds = f"{min(ns)} seed" if len(ns) == 1 else f"{min(ns)}–{max(ns)} seeds"
    ax.set_title(f"YOLO-World-S ({seeds})", loc="left", fontsize=7.5, color=INK)
    h = [Line2D([], [], color=c, linestyle=ls, linewidth=lw, label=n) for _, n, c, ls, lw in SERIES[::-1]]
    h.append(Line2D([], [], color=MUTED, marker="o", markerfacecolor="white", linestyle="none", markersize=4.2,
                    label=f"baseline collapsed (AP < {COLLAPSE:g})"))
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=6, frameon=False, handlelength=2.4, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, "figures", f"bits_curve.{ext}"), dpi=300)
    print("그림 -> figures/bits_curve.{pdf,png}")


if __name__ == "__main__":
    main()
