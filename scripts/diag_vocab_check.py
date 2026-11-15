"""11-13 3.2 검증: COCO 어휘로 잰 층별 진단이 LVIS(배포) 어휘로 잰 층별 진단과 같은 위치를 가리키는가.

rank_convs_leakfree(core/diag_w4_sensitivity.py)와 같은 절차(naive W8A8, conv 하나씩 weight만 W4, train2017 앞 200장,
확정 프로토콜)를 따르되, measure()의 LVIS 슬롯에 실제 LVIS 1,203 텍스트 임베딩을 넣어 두 어휘의 flip 변화량을
한 번에 잰다. 이 결과는 검증에만 쓴다. 보호 위치(P)는 여전히 COCO 진단(configs/protect_cache)만으로 정한다.
기존 캐시는 읽기만 하고(재현 확인) 쓰지 않는다.

사용(저장소 루트):
  CUDA_DEVICE_ORDER=PCI_BUS_ID .venv/bin/python scripts/diag_vocab_check.py --model yolov8s-world.pt --device 6
출력: runs/260_diag_vocab_check/<stem>.json (+ 요약 출력)
"""
import argparse, glob, json, os, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "core"))
import torch
import run_comparison as rc
import diag_w4_sensitivity as dg
from quant.pdquant import _find_head
from quant.vocab_metric import VocabMetric, encode_text_bank

OUT = os.path.join(ROOT, "runs", "260_diag_vocab_check")


def spearman(a, b):
    import numpy as np
    ra = np.argsort(np.argsort(-np.asarray(a))); rb = np.argsort(np.argsort(-np.asarray(b)))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="6")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="앞쪽 conv 그룹 몇 개만(속도 확인용)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = rc.resolve_device(args.device) if hasattr(rc, "resolve_device") else f"cuda:{args.device}"
    stem = os.path.splitext(os.path.basename(args.model))[0]
    from model_zoo import model_factory
    M = model_factory(args.model)
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    coco, lvis = rc.load_names("coco"), rc.load_names("lvis")
    paths = sorted(glob.glob(os.path.join(args.coco_root, "train2017", "*.jpg")))
    calib = [rc.preprocess(p, 640) for p in paths[:args.calib]]
    evals = [rc.preprocess(p, 640) for p in paths[:args.n_eval]]
    fp = rc.build(M, args.model, coco, device, calib, "fp")
    dg.HS = dg.HeadSim(_find_head(fp.model))
    vm_coco = VocabMetric(encode_text_bank(M, args.model, coco, device))
    vm_lvis = VocabMetric(encode_text_bank(M, args.model, lvis, device))
    refs = dg.reference(fp, evals, device, vm_lvis, vm_coco, 0.25, "coco")
    q = rc.build(M, args.model, coco, device, calib, "naive", fp=fp, w_bits=8, a_bits=8, skip_head=False,
                 first_last_bits=8, last_abits=16, attn_quant="attn_cls")
    groups = dg.groups_of(q.model, set(range(len(q.model.model))))
    if args.limit:
        groups = groups[:args.limit]
    t0 = time.time()
    base = dg.measure(q, evals, refs, device, vm_lvis, vm_coco)
    print(f"[base] coco {base['coco']:.2f}% (n={base['n_coco']})  lvis {base['lvis']:.2f}% (n={base['n_lvis']})  {time.time()-t0:.0f}s", flush=True)
    rows = []
    for i, (name, convs) in enumerate(groups):
        dg.set_wbits(convs, 4)
        m = dg.measure(q, evals, refs, device, vm_lvis, vm_coco)
        dg.set_wbits(convs, 8)
        rows.append(dict(name=name, mparam=sum(c.conv.weight.numel() for c in convs) / 1e6,
                         coco=m["coco"] - base["coco"], lvis=m["lvis"] - base["lvis"]))
        el = time.time() - t0
        print(f"[{i+1}/{len(groups)}] {name:18s} dcoco {rows[-1]['coco']:6.2f}  dlvis {rows[-1]['lvis']:6.2f}  "
              f"{el:.0f}s (남은 예상 {el/(i+2)*(len(groups)-i-1):.0f}s)", flush=True)
    json.dump(dict(model=args.model, base=base, rows=rows), open(os.path.join(OUT, f"{stem}.json"), "w"), indent=1)

    # 요약: 재현 확인, 순위 상관, 1.5% 보호 집합 겹침, 상위 층
    cache = os.path.join(ROOT, "configs", "protect_cache", f"{stem}_n200_c256_img640_la16_aqattn_cls.json")
    if os.path.exists(cache):
        old = {r["name"]: r["coco"] for r in json.load(open(cache))}
        diffs = [abs(r["coco"] - old[r["name"]]) for r in rows if r["name"] in old]
        print(f"[재현] 기존 COCO 캐시와 최대 차이 {max(diffs):.3f}%p (n={len(diffs)})")
    if args.limit:
        return
    pos = lambda k: [max(r[k], 0.0) for r in rows]
    print(f"[Spearman] 층별 Δ COCO vs LVIS: {spearman([r['coco'] for r in rows], [r['lvis'] for r in rows]):.3f}")
    sc, _, tot = dg.select_protected(rows, 0.015, "coco")
    sl, _, _ = dg.select_protected(rows, 0.015, "lvis")
    w = {r["name"]: r["mparam"] for r in rows}
    inter = set(sc) & set(sl)
    print(f"[보호 1.5%] COCO {sc}\n           LVIS {sl}\n           겹침 {sorted(inter)} "
          f"(weight 기준 {sum(w[n] for n in inter)/max(sum(w[n] for n in sc),1e-12)*100:.0f}%)")
    lv = {r["name"]: max(r["lvis"], 0.0) for r in rows}; tl = sum(lv.values())
    print(f"[집중] COCO로 고른 집합이 담는 LVIS 손상 {sum(lv[n] for n in sc)/tl*100:.1f}%  vs  LVIS로 고른 집합 {sum(lv[n] for n in sl)/tl*100:.1f}%")
    for k in ("coco", "lvis"):
        top = sorted(rows, key=lambda r: r[k], reverse=True)[:5]
        print(f"[상위5 {k}] " + ", ".join(f"{r['name']}({r[k]:.1f})" for r in top))


if __name__ == "__main__":
    main()
