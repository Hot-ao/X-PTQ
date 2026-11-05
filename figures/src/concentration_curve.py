"""손상 집중도 곡선 (11-03, 논문 그림 4 후보).

보호 진단 캐시(configs/protect_cache, 층 하나만 W4로 내렸을 때의 calibration 쪽 COCO flip 증가)로 모델마다
층을 손상이 큰 순으로 쌓아 x = 누적 weight 비율(양자화 weight 전체 대비), y = 누적 손상 비율(양수 증가 합 대비)을 그린다.
손상이 weight에 고르게 퍼져 있다면 y = x(회색 점선)다. 점은 P 규칙(select_protected, 예산 1.5%)이 실제로 고른 층의 몫이다.
G가 처리하는 C2fAttn cv2도 포함한 값이다(집중도 자체를 보이는 그림이라 처방과 무관하게 센다).
색은 구조 계열(텍스트 게이트 CNN / 게이트 없는 CNN / transformer), 선 모양은 계열 안의 모델을 구분한다.
그림: figures/concentration_curve.{pdf,png}
사용(저장소 루트): .venv/bin/python figures/src/concentration_curve.py
"""
from __future__ import annotations
import json, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "core"))
from diag_w4_sensitivity import select_protected

PC = os.path.join(ROOT, "configs", "protect_cache")
BUDGET = 0.015
XMAX = 10.0                                   # x축 상한(%)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, AXIS = "#0b0b0b", "#6b6a64", "#e1e0d9", "#c3c2b7"
MODELS = [  # (표시 이름, 캐시, 색, 선 모양)
    ("YOLO-World-S", "yolov8s-world_n200_c256_img640_la16_aqattn_cls.json", BLUE, "-"),
    ("YOLO-World-S v2", "yolov8s-worldv2_n200_c256_img640_la16_aqattn_cls.json", BLUE, (0, (5, 2))),
    ("YOLO-World-M", "yolov8m-world_n200_c256_img640_la16_aqattn_cls.json", BLUE, (0, (1.5, 1.5))),
    ("YOLO-World-L", "yolov8l-world_n200_c256_img640_la16_aqattn_cls.json", BLUE, (0, (5, 1.5, 1.5, 1.5))),
    ("YOLOE-v8s", "yoloe-v8s-seg_n200_c256_img640_la16_aqattn_cls.json", AQUA, "-"),
    ("YOLOE-11s", "yoloe-11s-seg_n200_c256_img640_la16_aqattn_cls.json", AQUA, (0, (5, 2))),
    ("OWLv2-B/16", "owlv2-base-patch16_n200_c256_img960_la16_aqattn_cls_p16.json", ORANGE, "-"),
    ("Grounding DINO-T", "grounding-dino-tiny_n64_c256_attn8_p16.json", ORANGE, (0, (5, 2))),
]


def fill(rows, b, W):
    """예산 b(비율) 안에 손상 큰 순으로 들어가는 층을 담는다(안 들어가는 층은 건너뜀). (쓴 weight, 담은 손상)."""
    used = cap = 0.0
    for r in sorted(rows, key=lambda r: r["coco"], reverse=True):
        if used + r["mparam"] <= b * W:
            used += r["mparam"]; cap += max(r["coco"], 0.0)
    return used, cap


def curve(rows):
    """예산별 최대 포착 곡선 (x = 예산 %, y = 담은 손상 %), 그리고 P 규칙 선택의 (x %, y %)."""
    W = sum(r["mparam"] for r in rows)
    tot = sum(max(r["coco"], 0.0) for r in rows)
    xs = [i * 0.02 for i in range(int(XMAX / 0.02) + 1)]
    ys = [fill(rows, x / 100, W)[1] / tot * 100 for x in xs]
    for i in range(1, len(ys)):               # 건너뛰기 greedy는 예산이 늘 때 줄어들 수 있다 -> 그 예산까지의 최댓값
        ys[i] = max(ys[i], ys[i - 1])
    sel, u, _ = select_protected(rows, BUDGET, "coco")
    pc = sum(max(r["coco"], 0.0) for r in rows if r["name"] in sel) / tot * 100
    return xs, ys, (u / W * 100, pc)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": AXIS, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.4, 3.15))
    ax.axvline(BUDGET * 100, color=MUTED, linewidth=0.8, linestyle=":", zorder=1)
    ax.text(BUDGET * 100 + 0.12, 107, "P budget 1.5%", fontsize=6.5, color=MUTED, va="top")
    ax.plot([0, XMAX], [0, XMAX], color=MUTED, linewidth=1.0, linestyle=(0, (3, 2)), zorder=1)
    ax.text(XMAX - 0.1, XMAX + 1.5, "if damage were uniform (y = x)", fontsize=6.5, color=MUTED, ha="right", va="bottom")
    print("| 모델 | 층 수 | P 선택 weight | P 선택이 담는 손상 | 1.5%에서 곡선 값 |")
    print("|---|---|---|---|---|")
    for name, f, col, ls in MODELS:
        rows = json.load(open(os.path.join(PC, f)))
        xs, ys, (px, py) = curve(rows)
        ax.plot(xs, ys, drawstyle="steps-post", color=col, linestyle=ls, linewidth=1.2, zorder=3)
        ax.plot([px], [py], "o", markersize=4.5, color=col, markeredgecolor="white", markeredgewidth=1.0, zorder=4)
        at = max(y for x, y in zip(xs, ys) if x <= BUDGET * 100)
        print(f"| {name} | {len(rows)} | {px:.2f}% | {py:.0f}% | {at:.0f}% |")
    ax.set_xlim(0, XMAX); ax.set_ylim(0, 108); ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("weights kept at 8 bit, most damaging first (%)")
    ax.set_ylabel("diagnosed damage covered (%)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    # 범례는 축 아래 한 묶음: 계열마다 한 열(색 = 구조 계열, 선 모양 = 모델), 마지막에 P 선택 점
    h = [Line2D([], [], color=c, linestyle=ls, linewidth=1.2, label=n) for n, _, c, ls in MODELS]
    h.insert(4, Line2D([], [], linestyle="none", label=" "))   # 왼쪽 열 = YOLO-World 네 개
    h.append(Line2D([], [], color=MUTED, marker="o", linestyle="none", markersize=4.5, markeredgecolor="white",
                    label="P selection (1.5%)"))
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=6, frameon=False, handlelength=2.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.21, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, "figures", f"concentration_curve.{ext}"), dpi=300)
    print("그림 -> figures/concentration_curve.{pdf,png}")


if __name__ == "__main__":
    main()
