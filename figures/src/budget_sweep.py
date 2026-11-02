"""보호 예산 스윕 그림 (11-02).

혼합 정밀도 연구(HAWQ-V3, BRECQ, OMPQ 등)는 예산을 여러 값으로 바꿔 정확도-크기 곡선으로 보고한다. 같은 형식으로
YOLO-World-S의 P 예산 스윕(runs/225_repro, 태그 bud-*, brecq+GP 단독, W4A8·W4A5, 예산 0~5%, seed 0~2)을 그린다.

- 가로축: 이론적 모델 크기(MiB). weight 비트만으로 정해지므로 W4A8과 W4A5가 같은 예산에서 같은 크기다.
  위쪽 눈금에 명목 예산(%)을 같이 적는다.
- 세로축(패널 3개)
  (a) calib 밖 train2017 200장 flip(--train-flip) -- 예산을 정하는 기준(누수 없음)
  (b) LVIS AP, (c) LVIS flip -- 보고용(예산 결정에 쓰지 않음)
- 점: seed 평균, 막대: seed 표준편차. 채택한 예산(1.5%)은 고리로 표시한다.
- --protect-refill-g 여부(11-02 기본 켬)가 섞여 있으면 따로 그린다(꺼짐 = 점선).

사용(저장소 루트): .venv/bin/python figures/src/budget_sweep.py [--runs runs/225_repro] [--chosen 0.015]
출력: figures/budget_sweep.{pdf,png}, figures/budget_sweep.csv
"""
from __future__ import annotations
import argparse, ast, csv, glob, os, re, statistics as st, sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "leaderboard"))
from make_leaderboard import parse_tables   # 로그 결과 표 파서(리더보드와 같은 것)

COND = "brecq+GP"
SETTINGS = ["W4A8", "W4A5"]
COLOR = {"W4A8": "#2a78d6", "W4A5": "#eb6834"}   # dataviz 기준 팔레트 슬롯 1, 2
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def collect(run_dirs):
    """{(설정, refill, 예산): {seed: dict(size, train_flip, coco, lvis, lvis_flip)}}"""
    out = defaultdict(dict)
    for f in sorted(p for d in run_dirs for p in glob.glob(os.path.join(d, "*.log"))):
        txt = open(f, errors="ignore").read()
        m = re.search(r"^\[args\] (\{.*\})\s*$", txt, re.M)
        if not m or "이론적 모델 크기" not in txt:
            continue
        try:
            a = ast.literal_eval(m.group(1))
        except Exception:
            continue
        if "yolov8s-world." not in os.path.basename(a.get("model", "")) + ".":
            continue
        setting = f"W{a.get('w_bits')}A{a.get('a_bits')}"
        if setting not in SETTINGS:
            continue
        res = parse_tables(txt).get(COND)
        if res is None:
            continue
        sizes = dict(re.findall(r"^\s+(\S+) 빌드 [\d.]+s\s+\(이론적 크기 ([\d.]+) MiB\)", txt, re.M))
        tf = re.search(r"^\[train-flip\] calib 밖 flip\(%\): (.*)$", txt, re.M)
        tfd = dict((k, float(v)) for k, v in re.findall(r"(\S+) ([\d.]+)", tf.group(1))) if tf else {}
        key = (setting, bool(a.get("protect_refill_g", False)), float(a.get("protect_budget", 0.015)))
        out[key][int(a.get("seed", 0))] = dict(size=float(sizes[COND]) if COND in sizes else None,
                                               train_flip=tfd.get(COND), coco=res.get("coco"), lvis=res.get("lvis"),
                                               lvis_flip=res.get("lvis_flip"), src=os.path.relpath(f, ROOT))
    return out


def agg(seeds, k):
    v = [s[k] for s in seeds.values() if s.get(k) is not None]
    if not v:
        return None, None
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[os.path.join(ROOT, "runs", "225_repro")])
    ap.add_argument("--chosen", type=float, default=0.015, help="채택한 예산(고리로 표시)")
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "budget_sweep"))
    args = ap.parse_args()

    data = collect(args.runs)
    if not data:
        sys.exit(f"[budget_sweep] {args.runs}에서 {COND} 예산 스윕 로그를 찾지 못했다.")

    rows = []
    for (setting, refill, bud), seeds in sorted(data.items()):
        r = dict(setting=setting, refill=refill, budget=bud, n=len(seeds), seeds=",".join(map(str, sorted(seeds))))
        for k in ("size", "train_flip", "coco", "lvis", "lvis_flip"):
            r[k], r[k + "_sd"] = agg(seeds, k)
        rows.append(r)
    with open(args.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[budget_sweep] {len(rows)}개 점 -> {args.out}.csv")
    for r in rows:
        print(f"  {r['setting']} refill={int(r['refill'])} 예산 {r['budget']:.1%} (n={r['n']}): "
              f"{r['size'] or float('nan'):.2f} MiB, train flip {r['train_flip'] or float('nan'):.2f}%, "
              f"LVIS {r['lvis'] or float('nan'):.2f}, LVIS flip {r['lvis_flip'] or float('nan'):.2f}%")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    panels = [("train_flip", "Held-out train flip (%) ↓\n(selection criterion)"),
              ("lvis", "LVIS AP ↑\n(reported only)"),
              ("lvis_flip", "LVIS flip (%) ↓\n(reported only)")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    ticks = {}
    for ax, (k, ylabel) in zip(axes, panels):
        for setting in SETTINGS:
            for refill in (True, False):
                pts = sorted((r for r in rows if r["setting"] == setting and r["refill"] == refill
                              and r["size"] is not None and r[k] is not None), key=lambda r: r["size"])
                if not pts:
                    continue
                xs, ys, es = [p["size"] for p in pts], [p[k] for p in pts], [p[k + "_sd"] for p in pts]
                ls = "-" if refill else "--"
                lab = setting + ("" if refill or not any(r["refill"] for r in rows) else " (no refill)")
                ax.errorbar(xs, ys, yerr=es, color=COLOR[setting], ls=ls, lw=2, marker="o", ms=6, mfc=COLOR[setting],
                            mec="white", mew=1.2, capsize=2.5, elinewidth=1, label=lab, zorder=3)
                for p in pts:
                    ticks.setdefault(round(p["size"], 2), p["budget"])
                    if abs(p["budget"] - args.chosen) < 1e-9:
                        ax.plot(p["size"], p[k], marker="o", ms=13, mfc="none", mec=INK, mew=1.4, zorder=4)
        ax.set_xlabel("Model size (MiB)")
        ax.set_ylabel(ylabel)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    # 위쪽 눈금: 명목 예산. 같은 크기에 예산이 여럿이면 처음 것.
    for ax in axes:
        top = ax.secondary_xaxis("top")
        xs = sorted(ticks)
        top.set_xticks(xs)
        top.set_xticklabels([f"{ticks[x] * 100:g}%" for x in xs], fontsize=7, color=INK2)
        top.set_xlabel("Protection budget", fontsize=8, color=INK2)
    from matplotlib.lines import Line2D
    h, l = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], ls="none", marker="o", ms=10, mfc="none", mec=INK, mew=1.4))
    l.append(f"selected budget ({args.chosen * 100:g}%)")
    fig.legend(h, l, frameon=False, loc="outside lower center", ncol=len(h), fontsize=8)
    fig.savefig(args.out + ".pdf")
    fig.savefig(args.out + ".png", dpi=200)
    print(f"[budget_sweep] 그림 -> {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
