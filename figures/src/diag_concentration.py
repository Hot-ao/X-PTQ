"""손상 집중도 분석과 층별 민감도 지도 (10-19).

보호 진단 캐시(configs/protect_cache, 층 하나만 W4로 내렸을 때의 calibration 쪽 COCO flip 증가)로 모델마다
  - 상위 1층 비중: 가장 민감한 층 하나 / 전체 양수 증가 합
  - 예산 포착률: P 규칙(파라미터 1.5% 예산, diag_w4_sensitivity.select_protected)으로 고른 층의 증가 합 / 전체 양수 증가 합
  - 잡음 대비 최댓값: 최댓값 / 잡음 수준(음수 값들의 RMS. 200장 측정이라 증가가 없는 층은 0 근처에서 양쪽으로 흔들린다)
  - 예산 포착량(%p): P가 고른 층들의 증가 합(절대값). 10-19: 비중(%)은 모델 안에서만 정규화해서, 손상이 작고 퍼진
    모델(OWLv2: 한 층만 4.1%p로 잡음 0.8%p를 겨우 넘음)에서 한 층이 31%로 부풀려 보였다. P의 효과는 절대량과 순서가 맞는다.
를 계산하고, 관측된 P의 효과(BRECQ 대비 P만 더한 조건의 짝 Δ LVIS AP, CVPR2027/runs)를 나란히 출력한다.
그림: figures/sensitivity_maps.{pdf,png}. 여섯 모델의 층별 증가 비중(깊이 순서), P가 고른 층은 파란색,
YOLO-World에서 G가 대신 다루는 텍스트 융합 conv는 빗금.
사용(저장소 루트): .venv/bin/python figures/src/diag_concentration.py
"""
from __future__ import annotations
import glob, json, math, os, re, statistics as st, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "core")); sys.path.insert(0, os.path.join(ROOT, "scripts", "leaderboard"))
from diag_w4_sensitivity import select_protected

PC = os.path.join(ROOT, "configs", "protect_cache")
BUDGET = 0.015
KIND_EN = {"CNN, 텍스트 게이트": "CNN, text-gated fusion", "CNN, 게이트 없음": "CNN, no gate",
           "CNN(YOLO11), 게이트 없음": "CNN (YOLO11), no gate", "ViT, late fusion": "ViT, late fusion"}
MODELS = [  # (표시 이름, 캐시, 구조, G가 대신 다루는 conv)
    ("YOLO-World-S", "yolov8s-world_n200_c256_img640_la16_aqattn_cls.json", "CNN, 텍스트 게이트", {f"{i}.cv2.conv" for i in (12, 15, 19, 22)}),
    ("YOLO-World-S v2", "yolov8s-worldv2_n200_c256_img640_la16_aqattn_cls.json", "CNN, 텍스트 게이트", {f"{i}.cv2.conv" for i in (12, 15, 18, 21)}),
    ("YOLO-World-M", "yolov8m-world_n200_c256_img640_la16_aqattn_cls.json", "CNN, 텍스트 게이트", {f"{i}.cv2.conv" for i in (12, 15, 19, 22)}),
    ("YOLOE-v8s", "yoloe-v8s-seg_n200_c256_img640_la16_aqattn_cls.json", "CNN, 게이트 없음", set()),
    ("YOLOE-11s", "yoloe-11s-seg_n200_c256_img640_la16_aqattn_cls.json", "CNN(YOLO11), 게이트 없음", set()),
    ("OWLv2-B/16", "owlv2-base-patch16_n200_c256_img960_la16_aqattn_cls_p16.json", "ViT, late fusion", set()),
]
# 관측된 P 효과: (모델 키, P 조건). YOLOE는 G가 아무것도 하지 않아 GP = P. v2는 P만 단독인 조건이 없다.
P_COND = {"YOLO-World-S": ("yolov8s-world", "brecq+P"), "YOLO-World-M": ("yolov8m-world", "brecq+P"),
          "YOLOE-v8s": ("yoloe-v8s-seg", "brecq+GP"), "YOLOE-11s": ("yoloe-11s-seg", "brecq+GP"),
          "OWLv2-B/16": ("owl", "brecq+P")}
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def metrics(rows):
    pos = [max(r["coco"], 0.0) for r in rows]
    tot = sum(pos)
    neg = [r["coco"] for r in rows if r["coco"] < 0]
    noise = math.sqrt(sum(v * v for v in neg) / len(neg)) if neg else float("nan")
    sel, used, total = select_protected(rows, BUDGET, "coco")
    cap = sum(max(r["coco"], 0.0) for r in rows if r["name"] in sel)
    top = max(rows, key=lambda r: r["coco"])
    return dict(n=len(rows), total=tot, top1=top["coco"] / tot, top_name=top["name"], capture=cap / tot, cap_abs=cap,
                top_abs=top["coco"],
                budget_frac=used / total, noise=noise, snr=top["coco"] / noise if noise == noise else float("nan"),
                sel=sel)


def paired(A, B):
    ss = sorted(set(A) & set(B))
    d = [A[s] - B[s] for s in ss]
    if len(d) < 2:
        return None
    m = st.mean(d)
    return m, T95.get(len(d) - 1, 1.96) * st.stdev(d) / math.sqrt(len(d)), len(d)


def p_effects():
    """{표시 이름: {설정: (Δ, CI, n)}} -- BRECQ 대비 P 조건의 LVIS AP 짝 차이."""
    os.environ["PTQ_EXTRA_RUNS"] = ""
    import make_leaderboard as ml
    recs, *_ = ml.collect()
    lv = {}
    for (m, p, s, c, seed), r in recs.items():
        if p.startswith("확정") and r["metrics"].get("lvis") is not None:
            lv.setdefault((m, s, c), {})[seed] = r["metrics"]["lvis"]
    for f in glob.glob(os.path.join(ROOT, "runs", "184_owl_main", "owl-W*-s*.log")):   # run_owl.py 로그
        txt = open(f, errors="ignore").read()
        m = re.search(r"^\[json\] (\{.*\})$", txt, re.M)
        if not m:
            continue
        s, seed = re.search(r"owl-(W\dA\d)-s(\d)", f).groups()
        for c, v in json.loads(m.group(1)).items():
            lv.setdefault(("owl", s, c), {})[int(seed)] = v["lvis"]
    out = {}
    for name, (mk, cond) in P_COND.items():
        for s in ("W4A8", "W4A5"):
            r = paired(lv.get((mk, s, cond), {}), lv.get((mk, s, "brecq"), {}))
            if r:
                out.setdefault(name, {})[s] = r
    return out


def figure(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    BLUE, GRAY, INK, MUTED, GRID = "#2a78d6", "#c3c2b7", "#0b0b0b", "#898781", "#e1e0d9"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRAY, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(len(data), 1, figsize=(7.0, 1.15 * len(data)), sharex=False, sharey=True)
    for ax, (name, kind, rows, mt, gset) in zip(axes, data):
        x = list(range(len(rows)))
        h = [r["coco"] for r in rows]
        if mt["noise"] == mt["noise"]:
            ax.axhspan(-2 * mt["noise"], 2 * mt["noise"], color=GRID, alpha=0.6, linewidth=0, zorder=0)
        col = [BLUE if r["name"] in mt["sel"] else GRAY for r in rows]
        bars = ax.bar(x, h, width=0.8, color=col, edgecolor="#fcfcfb", linewidth=0.4)
        for b, r in zip(bars, rows):
            if r["name"] in gset:
                b.set_facecolor("none"); b.set_edgecolor(BLUE); b.set_hatch("////"); b.set_linewidth(0.8)
        ax.set_xlim(-1, len(rows))
        ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
        ax.set_ylabel("Δ flip (pp)", fontsize=7)
        ax.set_title(f"{name}  ·  {KIND_EN[kind]}  ·  max {mt['top_abs']:.1f} pp, {BUDGET*100:.1f}% budget captures {mt['cap_abs']:.1f} pp,"
                     f" noise ±{2*mt['noise']:.1f} pp",
                     loc="left", fontsize=8, color=INK)
        # 블록 경계를 x 눈금으로: 이름 앞의 블록 번호가 바뀌는 곳
        blk = [r["name"].split(".")[0] for r in rows]
        # 눈금: 강조된 층(P 선택, G 대상)이 있는 블록을 먼저, 나머지 블록 시작은 겹치지 않을 때만. 위치는 그 층의 막대 자리.
        gap = max(3, len(rows) // 25)
        key = [i for i, r in enumerate(rows) if r["name"] in mt["sel"] or r["name"] in gset]
        ticks = []
        for i in key:
            if all(blk[i] != blk[j] for j in ticks) and all(abs(i - j) >= 2 for j in ticks):
                ticks.append(i)
        for i in range(len(rows)):
            if (i == 0 or blk[i] != blk[i - 1]) and all(blk[i] != blk[j] for j in ticks) and all(abs(i - j) >= gap for j in ticks):
                ticks.append(i)
        ticks.sort()
        ax.set_xticks(ticks); ax.set_xticklabels([blk[i] for i in ticks], fontsize=6)
        ti = max(range(len(rows)), key=lambda i: rows[i]["coco"])          # 가장 큰 막대에 층 이름을 직접
        ax.annotate(rows[ti]["name"].removesuffix(".conv"), (ti, rows[ti]["coco"]), xytext=(4, 0), textcoords="offset points",
                    fontsize=6.5, color=INK, va="center" if rows[ti]["coco"] > 15 else "bottom")
    axes[-1].set_xlabel("block index (input → output)", fontsize=7)
    fig.legend(handles=[Patch(color=BLUE, label="protected by P (1.5% budget)"), Patch(color=GRAY, label="other layers"), Patch(color=GRID, label="noise band (±2σ)"),
                        Patch(facecolor="none", edgecolor=BLUE, hatch="////", label="text-fusion conv handled by G")],
               loc="upper right", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, "figures", f"sensitivity_maps.{ext}"), dpi=200)
    print("그림 -> figures/sensitivity_maps.{pdf,png}")


def main():
    data = []
    for name, f, kind, gset in MODELS:
        rows = json.load(open(os.path.join(PC, f)))
        data.append((name, kind, rows, metrics(rows), gset))
    eff = p_effects()
    print("| 모델 | 구조 | 최민감 층 | 최대 증가 (%p) | 잡음 ±2σ (%p) | 예산 1.5% 포착량 (%p) | 포착 비중 | P 효과 W4A8 (Δ LVIS) | P 효과 W4A5 |")
    print("|---|---|---|---|---|---|---|---|---|")
    fe = lambda e: "–" if not e else f"{e[0]:+.2f} ± {e[1]:.2f} (n={e[2]})"
    for name, kind, rows, mt, gset in data:
        e = eff.get(name, {})
        print(f"| {name} | {kind} | {mt['top_name'].removesuffix('.conv')} | {mt['top_abs']:.1f} | {2*mt['noise']:.2f} | "
              f"{mt['cap_abs']:.1f} | {mt['capture']*100:.0f}% | {fe(e.get('W4A8'))} | {fe(e.get('W4A5'))} |")
    figure(data)


if __name__ == "__main__":
    main()
