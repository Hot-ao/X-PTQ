"""M 사전 검사가 제대로 고르는가 (11-04, 부록 그림 후보). 사용(저장소 루트): .venv/bin/python figures/src/mcheck_scatter.py
GP와 GPM을 둘 다 만든 실행(runs/234_rerun_s0, 236_fill_s0, 240, 247; 11-10에 240·247 추가)의 로그에서
  x = 검사 flip 차이 (calibration 밖 train2017 200장, COCO 어휘; flip_GPM - flip_GP, %p)   -> x <= 0 이면 GPM을 고른다(rho = 1.0)
  y = LVIS AP 차이 (GPM - GP, 점)                                                         -> 사후 확인용. 선택에는 쓰지 않는다
를 읽어 점으로 찍는다. 검사가 AP가 더 높은 쪽을 고르면 맞음(파란 원), 아니면 틀림(빨간 x). 틀림 사분면은 옅게 칠한다.
출력: figures/analysis/mcheck_scatter.{pdf,png}, 표준출력에 요약(맞힌 수, 고정 구성 대비 평균·최대 손해)."""
from __future__ import annotations
import glob, os, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INK, MUTED, GRID, AXIS = "#0b0b0b", "#6b6a64", "#e1e0d9", "#c3c2b7"
OKC, BADC, SHADE = "#2a78d6", "#d03b3b", "#f6efe9"
SHORT = {"yolov8s-world": "S", "yolov8s-worldv2": "S-v2", "yolov8m-world": "M", "yolov8l-world": "L",
         "yoloe-v8s-seg": "YOLOE-v8s", "yoloe-11s-seg": "YOLOE-11s"}


SEEDS = tuple(int(x) for x in os.environ.get("PTQ_DRAFT_SEEDS", "0").split(","))   # 11-11: 논문 결과 seed 0 통일. 3 seed판은 PTQ_DRAFT_SEEDS=0,1,2


def collect():
    R = []
    files = [f for d in ("234", "236", "240", "247") for f in glob.glob(os.path.join(ROOT, "runs", f"{d}_*", "*.log"))]
    for f in sorted(files):   # 11-10: 240(W4A4)·247(W4A6) 추가 -> 73건
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
                              ap_gp=ap[gp], ap_gpm=ap[gpm], mtime=os.path.getmtime(f)))
    latest = {}   # 11-10: 같은 칸(모델·비트·seed·기반)이 여러 번 돌았으면 가장 늦은 실행만 (S W4A6 BRECQ: 234 -> 247 재실행)
    for r in R:
        k = (r["model"], r["bits"], r["seed"], r["base"])
        if k not in latest or latest[k]["mtime"] < r["mtime"]:
            latest[k] = r
    return [r for r in latest.values() if r["seed"] in SEEDS]   # 11-11: seed 0만(Ours 구성을 정한 바로 그 검사들)


# 11-15: OWLv2·Grounding DINO 사후 검사(runs/262, YOLO와 같은 기준) 중 두 후보의 LVIS AP를 모두 잰 3건. dx, dy = (M 켬 - M 끔).
#   OWLv2 W8A8: 검사 2.22 vs 3.25, AP M 29.80 vs BRECQ 28.14 (RESULTS_FINAL §3)
#   OWLv2 W4A8: 검사 5.98 vs 6.67, AP PM 21.81 vs P 19.40 (runs/184)
#   GDINO W4A5: 검사 8.79 vs 8.73, AP PM 30.23 vs P 27.83 (runs/263 merged.json, LVIS 앞 2,000장)
#   OWLv2 W4A5, GDINO W4A8은 M 끔 후보의 AP를 재지 않아 넣지 않는다(검사는 둘 다 M 켬).
# 11-15(2): 사용자 결정 -- 두 후보의 AP를 모두 잰 확장 3건만으로는 대표성이 없어 그림에서 뺀다(본문 5.3에 문장으로만 적는다).
USE_EXT = False
EXT = [dict(model="OWLv2", bits="W8A8", dx=2.22 - 3.25, dy=29.80 - 28.14),
       dict(model="OWLv2", bits="W4A8", dx=5.98 - 6.67, dy=21.81 - 19.40),
       dict(model="G-DINO", bits="W4A5", dx=8.79 - 8.73, dy=30.23 - 27.83)]


def main():
    R = collect()
    E = [dict(r) for r in EXT] if USE_EXT else []
    for r in E:
        r["ok"] = (r["dx"] <= 0) == (r["dy"] >= 0)
    for r in R:
        r["pick_gpm"] = r["dx"] <= 0
        r["ok"] = r["pick_gpm"] == (r["dy"] >= 0)
    n = len(R); ok = sum(r["ok"] for r in R)
    pick = [r["ap_gpm"] if r["pick_gpm"] else r["ap_gp"] for r in R]
    orc = [max(r["ap_gp"], r["ap_gpm"]) for r in R]
    big = [r for r in R if abs(r["dy"]) >= 1]
    print(f"{n}개 중 {ok}개 맞음, |ΔAP| >= 1 {len(big)}개 중 {sum(r['ok'] for r in big)}개 맞음")
    eok = sum(r["ok"] for r in E)
    print(f"확장 {len(E)}건 중 {eok}건 맞음 -> 전체 {n + len(E)}건 중 {ok + eok}건")
    for name, v in (("검사로 선택", pick), ("항상 GP", [r["ap_gp"] for r in R]), ("항상 GPM", [r["ap_gpm"] for r in R])):
        print(f"  {name:8s} 평균 LVIS AP {sum(v) / n:.2f}, 사후 최선 대비 최대 손해 {max(o - p for o, p in zip(orc, v)):.2f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.edgecolor": AXIS, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.set_xscale("symlog", linthresh=0.5, linscale=0.8); ax.set_yscale("symlog", linthresh=1.0, linscale=0.8)
    XL, X, Y = 100, 60, 40     # 11-10: W4A4(runs/240) 추가로 왼쪽이 -70까지 간다
    ax.set_xlim(-XL, X); ax.set_ylim(-12, Y)
    # 틀린 선택 사분면: 검사는 GPM(x<=0)인데 GP가 낫다(y<0), 검사는 GP(x>0)인데 GPM이 낫다(y>=0)
    ax.fill_between([-XL, 0], -12, 0, color=SHADE, zorder=0, lw=0)
    ax.fill_between([0, X], 0, Y, color=SHADE, zorder=0, lw=0)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1); ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
    ax.text(-0.12, Y * 0.85, "← check picks TSM on", fontsize=6.0, color=MUTED, ha="right", va="top")
    ax.text(0.12, -12 * 0.85, "check picks TSM off →", fontsize=6.0, color=MUTED, ha="left", va="bottom")
    ax.text(X * 0.93, Y * 0.82, "wrong pick\n(TSM on was better)", fontsize=5.8, color=BADC, ha="right", va="top")
    ax.text(-XL * 0.93, -12 * 0.85, "wrong pick\n(TSM off was better)", fontsize=5.8, color=BADC, ha="left", va="bottom")
    for r in R:
        if r["ok"]:
            ax.scatter(r["dx"], r["dy"], s=22, color=OKC, edgecolor="white", linewidth=0.8, zorder=4)
        else:
            ax.scatter(r["dx"], r["dy"], s=26, marker="X", color=BADC, edgecolor="white", linewidth=0.6, zorder=5)
    for r in E:   # 11-15: 확장 모델은 마름모(채움 없음)로 구분
        ax.scatter(r["dx"], r["dy"], s=30, marker="D", facecolor="white", edgecolor=OKC if r["ok"] else BADC, linewidth=1.3, zorder=6)
    gd = next((r for r in E if r["model"] == "G-DINO"), None)
    if gd: ax.annotate("G-DINO W4A5", (gd["dx"], gd["dy"]), xytext=(7, 5), textcoords="offset points", fontsize=5.6, color=INK, va="center")
    # OWLv2 두 점은 이름을 달지 않는다(파란 점과 겹침). 캡션에서 "이름 없는 마름모 = OWLv2"로 설명한다.
    # 선택적 직접 표기: 차이가 큰 대표 사례와 반복해서 틀리는 설정만
    seen = set()
    notes = {("S", "W4A4", "brecq"): (7, 0, "W4A4"), ("M", "W4A5", "brecq"): (-4, 16, "M W4A5"),
             ("L", "W4A5", "brecq"): (7, 2, "L W4A5"), ("S", "W4A8", "brecq"): (-7, -6, "S W4A8"),
             ("YOLOE-11s", "W4A8", "brecq"): (6, 5, "YOLOE-11s W4A8"), ("M", "W4A8", "brecq"): (-7, 0, "M W4A8")}
    for r in R:
        k = (r["model"], r["bits"], r["base"])
        if k in notes and k not in seen and r["seed"] == 0:
            seen.add(k)
            ox, oy, lab = notes[k]
            ax.annotate(lab, (r["dx"], r["dy"]), xytext=(ox, oy), textcoords="offset points",
                        fontsize=5.6, color=INK, ha="center" if ox == 0 else "left" if ox > 0 else "right", va="center")
    ax.set_xticks([-100, -10, -1, 0, 1, 10, 50]); ax.set_xticklabels(["−100", "−10", "−1", "0", "1", "10", "50"])
    ax.set_yticks([-10, -1, 0, 1, 10]); ax.set_yticklabels(["−10", "−1", "0", "1", "10"])
    ax.set_xlabel("check flip rate, TSM on − off (pp)\n200 train images disjoint from calibration, COCO vocab.", fontsize=6.6, linespacing=1.2)
    ax.set_ylabel("LVIS AP, TSM on − off", fontsize=6.8)
    worst_e = max((abs(r["dy"]) for r in E if not r["ok"]), default=0)
    worst = max(o - p for o, p in zip(orc, pick))
    ax.set_title((f"TSM check picks the better candidate in {ok + eok}/{n + len(E)} runs\n"
                  f"(real-time {ok}/{n}, OWLv2·G-DINO {eok}/{len(E)}); worst miss {worst:.1f} / {worst_e:.1f} AP") if E else
                 f"TSM check picks the better candidate in {ok}/{n} runs\nworst miss {worst:.2f} AP",
                 loc="left", fontsize=6.9, color=INK, linespacing=1.25)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", color=OKC, linestyle="none", markersize=4.5, markeredgecolor="white", label="correct pick"),
         Line2D([], [], marker="X", color=BADC, linestyle="none", markersize=5, markeredgecolor="white", label="wrong pick"),
         Line2D([], [], marker="D", color=MUTED, markerfacecolor="white", linestyle="none", markersize=4.2, markeredgewidth=1.2,
                label="OWLv2 / G-DINO (post-hoc)")][:3 if E else 2]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=len(h), fontsize=5.8, frameon=False, columnspacing=0.8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, "figures", "analysis", f"mcheck_scatter.{ext}"), dpi=300, bbox_inches="tight")
    print("그림 -> figures/analysis/mcheck_scatter.{pdf,png}")


if __name__ == "__main__":
    main()
