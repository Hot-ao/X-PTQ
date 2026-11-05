"""정성 비교 2단계 (11-03): qual_dets.py가 저장한 검출에서 규칙으로 사례를 고르고 FP / BRECQ / Ours 그림을 그린다(CPU).

사례 규칙(손으로 고르지 않는다. 캡션에 그대로 쓴다):
  - 대상 = FP의 판정(위치마다 최고 점수 박스, 점수 >= --t-fp) 중 GT와 맞는 것(같은 LVIS 클래스, IoU >= 0.5). FP도 틀린 판정은 보지 않는다.
  - 각 대상에 대해 BRECQ / Ours에서 IoU >= 0.5로 겹치는 점수 >= --t-show 박스 중 가장 높은 점수의 클래스를 본다(없으면 놓침).
  - win = Ours는 FP와 같은 클래스, BRECQ는 다른 클래스이거나 놓침. loss = 그 반대.
  - 후보 이미지 = win >= 1, loss = 0. 희귀(r) > 흔함(c) > 잦음(f) 클래스 순, 그다음 win 수 순으로 정렬한다.
전체 win / loss 수를 함께 출력한다(그림이 대표성 있는지 캡션·부록에 밝히기 위해).
사용(저장소 루트):
  .venv/bin/python figures/src/qual_figure.py --tag s_w4a8                 # 후보 목록만(상위 30)
  .venv/bin/python figures/src/qual_figure.py --tag s_w4a8 --pick 1234,5678,91011   # 그 이미지로 그림
  .venv/bin/python figures/src/qual_figure.py --tag s_w4a8 --each 20         # 후보 상위 20장을 한 장씩
출력: figures/qual_<tag>.{pdf,png}, 후보 목록 figures/qual_<tag>_candidates.md, (--each) figures/qual_<tag>_wins/
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LVIS_ANN = "/data/taeho/lvis_datasets/labels_dl/extracted/lvis/annotations/lvis_v1_minival.json"
COCO_ROOT = "/data/taeho/coco_datasets"
FREQ_RANK = {"r": 0, "c": 1, "f": 2}
OK, BAD, OTHER = "#2a78d6", "#d03b3b", "#c3c2b7"


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match(box, dets, t_show):
    """box와 IoU >= 0.5로 겹치는 점수 >= t_show 검출 중 점수가 가장 높은 것의 클래스(없으면 None)."""
    best = None
    for d in dets:
        if d[4] >= t_show and iou(box, d[:4]) >= 0.5 and (best is None or d[4] > best[4]):
            best = d
    return None if best is None else best[5]


def top_per_location(dets, t):
    """점수 >= t 검출을 점수 순으로 보며, 이미 고른 박스와 IoU >= 0.5로 겹치면 버린다(클래스 무관).
    NMS가 클래스별이라 한 물체에 여러 클래스 박스가 남는데, 위치마다 가장 높은 점수의 박스가 모델의 판정이다."""
    out = []
    for d in sorted([d for d in dets if d[4] >= t], key=lambda d: -d[4]):
        if all(iou(d[:4], o[:4]) < 0.5 for o in out):
            out.append(d)
    return out


def analyse(dets, gt, t_fp, t_show):
    per_img, tot = {}, {"target": 0, "win": 0, "loss": 0, "both_ok": 0, "both_bad": 0}
    for iid, fp in dets["fp"].items():
        g = gt.get(int(iid), [])
        wins, losses = [], []
        for f in top_per_location(fp, t_fp):          # FP의 판정(위치별 최고 점수)이 GT와 맞는 것만 대상
            if not any(c == f[5] and iou(f[:4], b) >= 0.5 for b, c in g):
                continue
            tot["target"] += 1
            lb, lo = match(f[:4], dets["brecq"][iid], t_show), match(f[:4], dets["ours"][iid], t_show)
            okb, oko = lb == f[5], lo == f[5]
            if oko and not okb:
                wins.append((f, lb)); tot["win"] += 1
            elif okb and not oko:
                losses.append((f, lo)); tot["loss"] += 1
            elif okb and oko:
                tot["both_ok"] += 1
            else:
                tot["both_bad"] += 1
        per_img[int(iid)] = (wins, losses)
    return per_img, tot


def draw(ax, img, dets, focus, names, t_show, title, max_boxes):
    """점수 >= t_show 상위 max_boxes개. focus 박스(FP 기준 사례 위치)와 겹치는 검출만 색으로, 나머지는 회색."""
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=8, loc="left")
    # NMS가 클래스별이라 한 물체에 클래스가 다른 박스가 여러 개 겹친다. 위치마다 점수가 가장 높은 박스(= 모델의 판정,
    # match()가 보는 것과 같다) 하나만 그린다.
    # 사례 위치의 판정은 점수와 무관하게 항상 그리고, 나머지는 점수 순으로 max_boxes개까지 채운다
    tops = top_per_location(dets, t_show)
    key = [d for d in tops if any(iou(f[:4], d[:4]) >= 0.5 for f in focus)]
    shown = key + [d for d in tops if d not in key][:max(0, max_boxes - len(key))]
    for d in shown:
        hit = [f for f in focus if iou(f[:4], d[:4]) >= 0.5]
        if hit:
            col, lw, z = (OK if d[5] == hit[0][5] else BAD), 2.0, 3
        else:
            col, lw, z = OTHER, 0.8, 2
        x1, y1, x2, y2 = d[:4]
        ax.add_patch(__import__("matplotlib").patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=col,
                                                                 linewidth=lw, zorder=z))
        if hit:                           # 라벨이 이미지 오른쪽 밖으로 넘칠 것 같으면 박스 오른쪽 끝에 맞춘다
            lab = f"{names[d[5]]} {d[4]:.2f}"
            px = ax.get_window_extent().width                   # 패널 폭(픽셀); 글자 폭 약 6.3pt * 0.6
            est = len(lab) * 6.3 * 0.6 * ax.figure.dpi / 72 / px * img.width
            right = x1 + est > img.width
            ax.text(x2 if right else x1, y1 - 2, lab, fontsize=6.3, color="white", va="bottom",
                    ha="right" if right else "left", zorder=4, bbox=dict(facecolor=col, edgecolor="none", pad=1.0))
    for f in focus:                       # 이 모델이 사례 위치를 놓쳤으면 점선으로 위치만 표시
        if not any(iou(f[:4], d[:4]) >= 0.5 for d in shown):
            x1, y1, x2, y2 = f[:4]
            ax.add_patch(__import__("matplotlib").patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=BAD,
                                                                     linewidth=1.4, linestyle=(0, (3, 2)), zorder=3))
            ax.text(x1, y1 - 2, "missed", fontsize=6.3, color="white", va="bottom", zorder=4,
                    bbox=dict(facecolor=BAD, edgecolor="none", pad=1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--t-fp", type=float, default=0.3, help="사례 대상이 되는 FP 박스 점수 하한")
    ap.add_argument("--t-show", type=float, default=0.15, help="BRECQ/Ours 박스를 '검출'로 치는 점수 하한(그림에도 같은 값)")
    ap.add_argument("--pick", default="", help="그릴 image_id 목록(쉼표). 비우면 후보 목록만 만든다")
    ap.add_argument("--max-boxes", type=int, default=8)
    ap.add_argument("--each", type=int, default=0, help="후보 상위 N장을 이미지마다 따로 figures/qual_<tag>_wins/에 저장")
    ap.add_argument("--labels", default="FP32,BRECQ,Ours", help="열 제목 3개(쉼표)")
    args = ap.parse_args()
    sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "core"))
    from lvis import LVIS
    d = f"/data/taeho/cvpr2027_dets/qual_{args.tag}"
    dets = {k: json.load(open(os.path.join(d, f"{k}.json"))) for k in ("fp", "brecq", "ours")}
    lv = LVIS(LVIS_ANN)
    gt = defaultdict(list)
    for a in lv.dataset["annotations"]:
        x, y, w, h = a["bbox"]
        gt[a["image_id"]].append(([x, y, x + w, y + h], a["category_id"] - 1))
    freq = {c["id"] - 1: c["frequency"] for c in lv.dataset["categories"]}
    names = {c["id"] - 1: c["name"].replace("_", " ") for c in lv.dataset["categories"]}
    per_img, tot = analyse(dets, gt, args.t_fp, args.t_show)
    print(f"대상(FP 맞힘, 점수 >= {args.t_fp}) {tot['target']}개: Ours만 맞힘 {tot['win']} / BRECQ만 맞힘 {tot['loss']} / "
          f"둘 다 맞힘 {tot['both_ok']} / 둘 다 틀림 {tot['both_bad']}")
    cand = [(iid, w) for iid, (w, l) in per_img.items() if w and not l]
    cand.sort(key=lambda x: (min(FREQ_RANK[freq[f[5]]] for f, _ in x[1]), -len(x[1])))
    lines = [f"# 정성 후보 ({args.tag})", "",
             f"규칙: FP 점수 >= {args.t_fp}이고 GT와 맞는 박스 중 Ours만 FP와 같은 클래스(BRECQ는 다른 클래스/놓침), loss 0인 이미지.",
             f"전체 대상 {tot['target']}개: Ours만 맞힘 {tot['win']}, BRECQ만 맞힘 {tot['loss']}, 둘 다 맞힘 {tot['both_ok']}, 둘 다 틀림 {tot['both_bad']}.",
             "", "| image_id | 사례 (FP·Ours 클래스 [빈도] → BRECQ) | FP 박스 수(점수 >= t_fp) |", "|---|---|---|"]
    for iid, w in cand[:60]:
        desc = "; ".join(f"{names[f[5]]} [{freq[f[5]]}] → {names[b] if b is not None else '놓침'}" for f, b in w)
        nfp = len(top_per_location(dets["fp"][str(iid)], args.t_fp))
        lines.append(f"| {iid} | {desc} | {nfp} |")
    cpath = os.path.join(ROOT, "figures", f"qual_{args.tag}_candidates.md")
    open(cpath, "w").write("\n".join(lines) + "\n")
    print(f"후보 {len(cand)}장 -> {cpath}")
    if not args.pick and not args.each:
        print("\n".join(lines[6:36]))
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    labels = args.labels.split(",")

    def render(picks, stem, exts):
        # 행 높이는 그 행 이미지의 세로/가로 비율에 맞춘다(열 폭 7.0/3 in)
        ims = [Image.open(os.path.join(COCO_ROOT, "val2017", f"{iid:012d}.jpg")).convert("RGB") for iid in picks]
        hr = [im.height / im.width for im in ims]
        fig, axes = plt.subplots(len(picks), 3, figsize=(7.0, sum(hr) * 7.0 / 3 + 0.25), squeeze=False,
                                 gridspec_kw={"height_ratios": hr})
        for r, iid in enumerate(picks):
            focus = [f for f, _ in per_img[iid][0]]
            for c, key in enumerate(("fp", "brecq", "ours")):
                draw(axes[r][c], ims[r], dets[key][str(iid)], focus, names, args.t_show, labels[c] if r == 0 else "",
                     args.max_boxes)
        fig.subplots_adjust(left=0.005, right=0.995, top=1 - 0.22 / fig.get_figheight(), bottom=0.005, wspace=0.02, hspace=0.04)
        for ext in exts:
            fig.savefig(f"{stem}.{ext}", dpi=250)
        plt.close(fig)

    if args.pick:
        render([int(x) for x in args.pick.split(",")], os.path.join(ROOT, "figures", f"qual_{args.tag}"), ("pdf", "png"))
        print(f"그림 -> figures/qual_{args.tag}.{{pdf,png}}")
    if args.each:
        # 후보 순서(위 규칙) 상위 N장을 이미지마다 한 파일로. 목록은 같은 폴더 README.md
        out = os.path.join(ROOT, "figures", f"qual_{args.tag}_wins")
        os.makedirs(out, exist_ok=True)
        rows = [f"# Ours만 맞힌 사례 상위 {args.each}장 ({args.tag})", "", lines[2], lines[3], "",
                "| 파일 | image_id | 사례 (FP·Ours 클래스 [빈도] → BRECQ) |", "|---|---|---|"]
        for k, (iid, w) in enumerate(cand[:args.each], 1):
            fn = f"{k:02d}_{iid}"
            render([iid], os.path.join(out, fn), ("png",))
            desc = "; ".join(f"{names[f[5]]} [{freq[f[5]]}] → {names[b] if b is not None else '놓침'}" for f, b in w)
            rows.append(f"| {fn}.png | {iid} | {desc} |")
        open(os.path.join(out, "README.md"), "w").write("\n".join(rows) + "\n")
        print(f"{args.each}장 -> {out}")

if __name__ == "__main__":
    main()
