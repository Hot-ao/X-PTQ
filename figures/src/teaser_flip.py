"""동기 그림 (11-03, 논문 그림 1 후보): AP는 버티는데 held-out 어휘의 판정이 무너진다.

YOLO-World-S, BRECQ, 확정 프로토콜(runs/158_final, 6 seed). 리더보드 수집기(make_leaderboard.collect)로 seed별 값을 읽는다.
  (a) FP 대비 AP 유지율: COCO AP(calibration 어휘), LVIS AP(held-out 어휘)
  (b) top-1 flip: COCO 어휘(Top1_flip), LVIS 어휘(LVIS_flip). FP가 확신하는 예측 중 top-1 클래스가 바뀐 비율
BRECQ가 무너진 seed(LVIS AP < 15, W4A5 seed 0)는 평균에서 뺀다. W4A4는 BRECQ가 전부 무너져(LVIS AP < 1) 뺀다.
그림: figures/teaser_flip.{pdf,png}
사용(저장소 루트): .venv/bin/python figures/src/teaser_flip.py
"""
from __future__ import annotations
import os, statistics as st, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "leaderboard"))

MODEL, PROTO, COND = "yolov8s-world", "확정 (attn8+A16)", "brecq"
BITS = ["W8A8", "W4A8", "W4A6", "W4A5"]
COLLAPSE_LVIS = 15.0
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, AXIS = "#0b0b0b", "#6b6a64", "#e1e0d9", "#c3c2b7"


def load():
    os.environ["PTQ_EXTRA_RUNS"] = ""
    import make_leaderboard as ml
    recs = ml.collect()[0]
    fp, per = {}, {b: [] for b in BITS}
    for (m, p, s, c, seed), r in recs.items():
        if m != MODEL or p != PROTO:
            continue
        mt = r["metrics"]
        if c == "FP32" and mt.get("lvis") is not None:
            fp = {"coco": mt["coco"], "lvis": mt["lvis"]}
        if c == COND and s in per and mt.get("lvis") is not None and mt["lvis"] >= COLLAPSE_LVIS:
            per[s].append(mt)
    out = {}
    for b in BITS:
        ms = per[b]
        out[b] = {k: st.mean(x[k] for x in ms) for k in ("coco", "lvis", "top1", "lvis_flip")} | {"n": len(ms)}
    return fp, out


def main():
    fp, d = load()
    print("FP", fp)
    for b in BITS:
        v = d[b]
        print(f"{b}: n={v['n']} COCO {v['coco']:.2f} ({v['coco']/fp['coco']*100:.0f}%) LVIS {v['lvis']:.2f} ({v['lvis']/fp['lvis']*100:.0f}%)"
              f" | flip COCO {v['top1']:.2f} LVIS {v['lvis_flip']:.2f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.edgecolor": AXIS, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.4, 1.85))
    x = range(len(BITS))
    series = [  # (축, 값, 색, 표시용 형식, 표기 위치: 위 +1 / 아래 -1)
        (a1, [d[b]["coco"] / fp["coco"] * 100 for b in BITS], BLUE, "{:.0f}%", 1),
        (a1, [d[b]["lvis"] / fp["lvis"] * 100 for b in BITS], ORANGE, "{:.0f}%", -1),
        (a2, [d[b]["top1"] for b in BITS], BLUE, "{:.1f}", 1),
        (a2, [d[b]["lvis_flip"] for b in BITS], ORANGE, "{:.1f}", 1),
    ]
    for ax, ys, col, fmt, side in series:
        ax.plot(x, ys, color=col, linewidth=1.6, marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        # 선택적 직접 표기: W4A8(본문이 인용하는 칸)과 마지막 칸만
        for i in (1, len(BITS) - 1):
            ax.annotate(fmt.format(ys[i]), (i, ys[i]), xytext=(0, 5 * side), textcoords="offset points", ha="center",
                        va="bottom" if side > 0 else "top",
                        fontsize=6.3, color=INK)
    a1.set_ylim(60, 104); a1.set_yticks([60, 70, 80, 90, 100])
    a1.set_title("(a) AP retained vs. FP (%)", loc="left", fontsize=7.5, color=INK)
    a2.set_ylim(0, 33)
    a2.set_title("(b) top-1 flip (%)", loc="left", fontsize=7.5, color=INK)
    for ax in (a1, a2):
        ax.set_xticks(list(x)); ax.set_xticklabels(BITS, fontsize=6.5)
        ax.set_xlim(-0.35, len(BITS) - 0.65)
        ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    h = [Line2D([], [], color=BLUE, linewidth=1.6, marker="o", markersize=4, markeredgecolor="white", label="COCO (calibration vocabulary)"),
         Line2D([], [], color=ORANGE, linewidth=1.6, marker="o", markersize=4, markeredgecolor="white", label="LVIS (held-out vocabulary)")]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=6.3, frameon=False, handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, "figures", f"teaser_flip.{ext}"), dpi=300)
    print("그림 -> figures/teaser_flip.{pdf,png}")


if __name__ == "__main__":
    main()
