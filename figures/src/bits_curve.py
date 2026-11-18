"""비트별 성능 곡선 (11-03 초판, 11-10 3 seed판, 논문 그림 후보): activation 비트를 내릴 때 기존 PTQ는 무너지고 우리 방법은 버틴다.

YOLO-World-S, LVIS AP. 11-10부터 리더보드 집계 코드(scripts/leaderboard/make_leaderboard.py의 collect -> build_results,
best, pick_ours)를 직접 불러 seed 0-2를 쓴다. 메인 표 초안 G와 같은 규칙이다.
  - 11-11부터 seed 0 한 번 실행 값(논문 결과 통일). PTQ_DRAFT_SEEDS=0,1,2면 점 = 평균, 막대 = seed 최솟값-최댓값
  - Ours (on BRECQ) = BRECQ seed 0의 calibration 밖 flip으로 GP/GPM을 고르고 모든 seed에 고정(pick_ours)
  - Ours (on QDrop) = QDrop 위에 같은 구성
  - 11-13: PD-Quant, Ours (on PD-Quant) 추가(seed 0만 있다)
  - SmoothQuant / AWQ는 seed 0만 있어서 이 그림에서 뺐다(11-03판에는 있었음, 표 1에 남는다)
  - 기준선 seed 중 하나라도 LVIS AP < 15면 속 빈 점(BRECQ·QDrop W4A5 seed 0 붕괴)
그림: figures/experiments/bits_curve.{pdf,png}
사용(저장소 루트): .venv/bin/python figures/src/bits_curve.py   (로그 1,200여 개를 읽어 1-2분 걸린다)
"""
from __future__ import annotations
import os, sys, statistics as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "leaderboard"))
BITS = ["W8A8", "W4A8", "W4A6", "W4A5", "W4A4"]
MODEL = "yolov8s-world"
SEEDS = tuple(int(x) for x in os.environ.get("PTQ_DRAFT_SEEDS", "0").split(","))   # 11-11: 논문 결과 seed 0 통일. 3 seed판은 PTQ_DRAFT_SEEDS=0,1,2
COLLAPSE = 15.0
INK, MUTED, GRID, AXIS = "#0b0b0b", "#6b6a64", "#e1e0d9", "#c3c2b7"
SERIES = [  # (키, 표시 이름, 색, 선 모양, 굵기)
    ("brecq", "BRECQ", "#eb6834", "-", 1.3),
    ("qdrop", "QDrop", "#1baf7a", "-", 1.3),
    ("pdquant", "PD-Quant", "#9a6bc9", "-", 1.3),
    ("qdrop+ours", "Ours (on QDrop)", "#2a78d6", (0, (4, 2)), 1.6),
    ("pdquant+ours", "Ours (on PD-Quant)", "#2a78d6", (0, (1, 1.5)), 1.6),   # 11-13 추가
    ("ours", "Ours (on BRECQ)", "#2a78d6", "-", 2.0),
]


def load():
    import make_leaderboard as L
    recs, allrecs, _, _ = L.collect()
    allrecs += L.collect_json_logs()
    res = L.build_results(allrecs)
    per = {}
    for b in BITS:
        row = {}
        for c in ("brecq", "qdrop", "pdquant"):
            x = L.best(res, MODEL, b, c, seeds=SEEDS)
            row[c] = {k: v["lvis"] for k, v in x["per"].items()} if x else {}
        x, note = L.pick_ours(res, MODEL, b, SEEDS)
        row["ours"] = {k: v["lvis"] for k, v in x["per"].items()} if x else {}
        _, n0 = L.pick_ours(res, MODEL, b)
        qc = "qdrop+" + (n0.split(":")[-1] if n0 else "GPM")
        x = L.best(res, MODEL, b, qc, seeds=SEEDS)
        row["qdrop+ours"] = {k: v["lvis"] for k, v in x["per"].items()} if x else {}
        x = L.best(res, MODEL, b, "pdquant+" + qc.split("+", 1)[1], seeds=SEEDS)
        row["pdquant+ours"] = {k: v["lvis"] for k, v in x["per"].items()} if x else {}
        row["_cfg"] = n0.split(":")[-1] if n0 else "?"
        per[b] = row
    fp = L.best(res, MODEL, "W8A8", "FP32")
    return per, fp["stats"]["lvis"][0]


def main():
    per, fp = load()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.edgecolor": AXIS, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.4, 2.55))   # 11-15: 범례를 축 안 왼쪽 아래로 옮겨 높이를 줄였다
    ax.axhline(fp, color=MUTED, linewidth=1.0, linestyle=(0, (3, 2)), zorder=1)
    ax.text(len(BITS) - 1 + 0.25, fp + 0.4, f"FP32 {fp:.1f}", fontsize=6.3, color=MUTED, ha="right", va="bottom")
    ns = set()
    for key, name, col, ls, lw in SERIES:
        xs, ys, lo, hi, hollow = [], [], [], [], []
        for i, b in enumerate(BITS):
            v = [per[b][key][s] for s in SEEDS if per[b][key].get(s) is not None]
            if not v:
                continue
            ns.add(len(v))
            m = st.mean(v)
            xs.append(i); ys.append(m); lo.append(m - min(v)); hi.append(max(v) - m)
            hollow.append(key in ("brecq", "qdrop", "pdquant") and min(v) < COLLAPSE)
        print(f"{name:16s} " + "  ".join(f"{BITS[x]} {y:5.2f} [{y - l:.2f}-{y + h:.2f}]" for x, y, l, h in zip(xs, ys, lo, hi)))
        ax.plot(xs, ys, color=col, linestyle=ls, linewidth=lw, zorder=4 if key == "ours" else 3)
        if len(SEEDS) > 1:
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=col, elinewidth=0.9, capsize=1.8, capthick=0.9, zorder=3)
        for x, y, h in zip(xs, ys, hollow):
            ax.plot([x], [y], "o", markersize=4.2, zorder=5, color=col, markerfacecolor="white" if h else col,
                    markeredgecolor=col if h else "white", markeredgewidth=0.9)
        if key == "ours":            # 직접 표기: 우리 방법만
            for x, y, h in zip(xs, ys, hi):
                if fp - (y + h) < 2.0 and x > 0:   # FP 점선과 겹치면 오른쪽 아래로(W4A8)
                    ax.annotate(f"{y:.1f}", (x, y), xytext=(6, -7), textcoords="offset points", ha="left", fontsize=6.2, color=INK)
                else:
                    ax.annotate(f"{y:.1f}", (x, y + h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=6.2, color=INK)
    print("구성(BRECQ seed 0 선택):", {b: per[b]["_cfg"] for b in BITS})
    ax.set_xticks(range(len(BITS))); ax.set_xticklabels(BITS, fontsize=6.8)
    ax.set_xlim(-0.3, len(BITS) - 0.7); ax.set_ylim(0, 29)
    ax.set_ylabel("LVIS AP (proxy for deployment vocabulary)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    seeds = f"{min(ns)} seeds" if len(ns) == 1 else f"{min(ns)}–{max(ns)} seeds"
    ax.set_title("YOLO-World-S (seed 0)" if len(SEEDS) == 1 else f"YOLO-World-S (mean of {seeds}, bars = seed range)", loc="left", fontsize=7.2, color=INK)
    # 범례: 왼쪽 열 = Ours 세 개, 오른쪽 열 = 기준선 세 개 + 붕괴 표시(ncol=2는 열 순서로 채운다)
    L_ = {k: Line2D([], [], color=c, linestyle=ls, linewidth=lw, label=n) for k, n, c, ls, lw in SERIES}
    h = [L_[k] for k in ("ours", "qdrop+ours", "pdquant+ours")]
    h += [L_[k] for k in ("brecq", "qdrop", "pdquant")]
    h.append(Line2D([], [], color=MUTED, marker="o", markerfacecolor="white", linestyle="none", markersize=4.2,
                    label=f"baseline collapsed (AP < {COLLAPSE:g})" if len(SEEDS) == 1 else f"a baseline seed collapsed (AP < {COLLAPSE:g})"))
    # 11-15: 축 안 왼쪽 아래(W8A8--W4A6, AP 0--17)가 비어 있어 범례를 그 안에 둔다(그림 밖 범례보다 높이가 준다).
    ax.legend(handles=h, loc="lower left", ncol=1, fontsize=5.6, frameon=True, framealpha=0.92, edgecolor="none",
              handlelength=2.4, borderaxespad=0.3, labelspacing=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, "figures", "experiments", f"bits_curve.{ext}"), dpi=300, bbox_inches="tight", pad_inches=1 / 72)   # 11-16: 그래프 밖 공백을 1pt만 남기고 자른다
    print("그림 -> figures/experiments/bits_curve.{pdf,png}")


if __name__ == "__main__":
    main()
