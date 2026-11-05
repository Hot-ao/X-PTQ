"""M 사전 검사가 제대로 고르는가 (11-04, 부록 그림 후보). 사용(저장소 루트): .venv/bin/python figures/src/mcheck_scatter.py
GP와 GPM을 둘 다 만든 실행(runs/234_rerun_s0, runs/236_fill_s0)의 로그에서
  x = 검사 flip 차이 (calibration 밖 train2017 200장, COCO 어휘; flip_GPM - flip_GP, %p)   -> x <= 0 이면 GPM을 고른다(rho = 1.0)
  y = LVIS AP 차이 (GPM - GP, 점)                                                         -> 사후 확인용. 선택에는 쓰지 않는다
를 읽어 점으로 찍는다. 검사가 AP가 더 높은 쪽을 고르면 맞음(파란 원), 아니면 틀림(빨간 x). 틀림 사분면은 옅게 칠한다.
출력: figures/mcheck_scatter.{pdf,png}, 표준출력에 요약(맞힌 수, 고정 구성 대비 평균·최대 손해)."""
from __future__ import annotations
import glob, os, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INK, MUTED, GRID, AXIS = "#0b0b0b", "#6b6a64", "#e1e0d9", "#c3c2b7"
OKC, BADC, SHADE = "#2a78d6", "#d03b3b", "#f6efe9"
SHORT = {"yolov8s-world": "S", "yolov8s-worldv2": "S-v2", "yolov8m-world": "M", "yolov8l-world": "L",
         "yoloe-v8s-seg": "YOLOE-v8s", "yoloe-11s-seg": "YOLOE-11s"}


def collect():
    R = []
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", "23[46]_*", "*.log"))):
        t = open(f, errors="ignore").read()
        m = re.search(r"'model': '([^']+)'.*?'w_bits': (\d+), 'a_bits': (\d+).*?'seed': (\d+).*?'first_last_bits': (\d+)", t)
        tf = re.search(r"\[train-flip\] calib 밖 flip\(%\): (.*)", t)
        if not (m and tf) or m.group(5) != "8":          # 첫·마지막 16bit 변형(fl16)은 뺀다
            continue
        tfd = {k: float(v) for k, v in re.findall(r"(\S+) ([\d.]+)", tf.group(1))}
        ap = {k: float(v) * 100 for k, v in re.findall(r"\[LVIS-AP\]\[(\S+)\] AP=([\d.]+)", t)}
        for b in ("brecq", "qdrop"):
            gp, gpm = f"{b}+GP", f"{b}+GPM"
            if gp in tfd and gpm in tfd and gp in ap and gpm in ap:
                R.append(dict(model=SHORT.get(m.group(1).replace(".pt", ""), m.group(1)), bits=f"W{m.group(2)}A{m.group(3)}",
                              seed=int(m.group(4)), base=b, dx=tfd[gpm] - tfd[gp], dy=ap[gpm] - ap[gp],
                              ap_gp=ap[gp], ap_gpm=ap[gpm]))
    return R


def main():
    R = collect()
    for r in R:
        r["pick_gpm"] = r["dx"] <= 0
        r["ok"] = r["pick_gpm"] == (r["dy"] >= 0)
    n = len(R); ok = sum(r["ok"] for r in R)
    pick = [r["ap_gpm"] if r["pick_gpm"] else r["ap_gp"] for r in R]
    orc = [max(r["ap_gp"], r["ap_gpm"]) for r in R]
    big = [r for r in R if abs(r["dy"]) >= 1]
    print(f"{n}개 중 {ok}개 맞음, |ΔAP| >= 1 {len(big)}개 중 {sum(r['ok'] for r in big)}개 맞음")
    for name, v in (("검사로 선택", pick), ("항상 GP", [r["ap_gp"] for r in R]), ("항상 GPM", [r["ap_gpm"] for r in R])):
        print(f"  {name:8s} 평균 LVIS AP {sum(v) / n:.2f}, 사후 최선 대비 최대 손해 {max(o - p for o, p in zip(orc, v)):.2f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.edgecolor": AXIS, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.set_xscale("symlog", linthresh=0.5, linscale=0.8); ax.set_yscale("symlog", linthresh=1.0, linscale=0.8)
    X, Y = 60, 40
    ax.set_xlim(-X, X); ax.set_ylim(-12, Y)
    # 틀린 선택 사분면: 검사는 GPM(x<=0)인데 GP가 낫다(y<0), 검사는 GP(x>0)인데 GPM이 낫다(y>=0)
    ax.fill_between([-X, 0], -12, 0, color=SHADE, zorder=0, lw=0)
    ax.fill_between([0, X], 0, Y, color=SHADE, zorder=0, lw=0)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1); ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
    ax.text(-0.12, Y * 0.85, "← check picks GPM", fontsize=6.0, color=MUTED, ha="right", va="top")
    ax.text(0.12, -12 * 0.85, "check picks GP →", fontsize=6.0, color=MUTED, ha="left", va="bottom")
    ax.text(X * 0.93, Y * 0.82, "wrong pick\n(GPM was better)", fontsize=5.8, color=BADC, ha="right", va="top")
    ax.text(-X * 0.93, -12 * 0.85, "wrong pick\n(GP was better)", fontsize=5.8, color=BADC, ha="left", va="bottom")
    for r in R:
        if r["ok"]:
            ax.scatter(r["dx"], r["dy"], s=22, color=OKC, edgecolor="white", linewidth=0.8, zorder=4)
        else:
            ax.scatter(r["dx"], r["dy"], s=26, marker="X", color=BADC, edgecolor="white", linewidth=0.6, zorder=5)
    # 선택적 직접 표기: 차이가 큰 대표 사례와 반복해서 틀리는 설정만
    seen = set()
    notes = {("S", "W4A4", "brecq"): (7, 0, "S, M W4A4"), ("M", "W4A5", "brecq"): (-6, 7, "M W4A5"),
             ("L", "W4A5", "brecq"): (7, 2, "L W4A5"), ("S", "W4A8", "brecq"): (-7, -6, "S W4A8"),
             ("YOLOE-11s", "W4A8", "brecq"): (6, 5, "YOLOE-11s W4A8"), ("M", "W4A8", "brecq"): (-7, 0, "M W4A8")}
    for r in R:
        k = (r["model"], r["bits"], r["base"])
        if k in notes and k not in seen and r["seed"] == 0:
            seen.add(k)
            ox, oy, lab = notes[k]
            ax.annotate(lab, (r["dx"], r["dy"]), xytext=(ox, oy), textcoords="offset points",
                        fontsize=5.6, color=INK, ha="left" if ox >= 0 else "right", va="center")
    ax.set_xticks([-50, -10, -1, 0, 1, 10, 50]); ax.set_xticklabels(["−50", "−10", "−1", "0", "1", "10", "50"])
    ax.set_yticks([-10, -1, 0, 1, 10]); ax.set_yticklabels(["−10", "−1", "0", "1", "10"])
    ax.set_xlabel("check flip, GPM − GP (pp)\nheld-out train images, COCO vocab.", fontsize=6.6, linespacing=1.2)
    ax.set_ylabel("LVIS AP, GPM − GP", fontsize=6.8)
    ax.set_title(f"M check picks the better candidate in {ok}/{n} runs\n"
                 f"all {len(big)} runs with |ΔAP| ≥ 1 correct; worst miss {max(o - p for o, p in zip(orc, pick)):.1f} AP",
                 loc="left", fontsize=6.9, color=INK, linespacing=1.25)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", color=OKC, linestyle="none", markersize=4.5, markeredgecolor="white", label="correct pick"),
         Line2D([], [], marker="X", color=BADC, linestyle="none", markersize=5, markeredgecolor="white", label="wrong pick")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, fontsize=6, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, "figures", f"mcheck_scatter.{ext}"), dpi=300, bbox_inches="tight")
    print("그림 -> figures/mcheck_scatter.{pdf,png}")


if __name__ == "__main__":
    main()
