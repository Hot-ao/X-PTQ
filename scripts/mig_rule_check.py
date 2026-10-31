"""M의 xmax 규칙 비교(10-30): 기존(α 탐색 8장에서 max|x|) vs calibration 전체 max|x| vs 전체+하한(단위 최댓값의 1%).
본 실험과 같은 빌드 경로로 M 직전까지 가서, 같은 FP 모델 복사본에 세 규칙을 각각 적용하고 비교한다(재구성·평가 없음).
  - 단위별 고른 α(로그의 a=)가 바뀌는지, 소비 conv의 s가 얼마나 바뀌는지
  - 규칙별 M 소요 시간과 GPU 최대 메모리
  - (GDINO) calibration 밖 200장에서 max|x|가 s 기준 max|x|를 몇 배 넘는지 -- 하한이 필요한지
사용: python scripts/mig_rule_check.py <model> <device> [w_bits a_bits]
  model: yolov8s-world.pt | yolov8m-world.pt | yolov8l-world.pt | yoloe-v8s-seg.pt | yoloe-11s-seg.pt | owl | gdino"""
import contextlib, copy, glob, io, os, re, sys, time
import torch

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [R, os.path.join(R, "core")]
os.chdir(R)
MODEL, DEV = sys.argv[1], f"cuda:{sys.argv[2]}"
WB, AB = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (4, 8)
COCO = "/data/taeho/coco_datasets"
TRAIN = sorted(glob.glob(os.path.join(COCO, "train2017", "*.jpg")))
RULES = [("old8", dict()), ("old8+floor", dict(xmax_floor=0.01)), ("all", dict(stat_images=True, xmax_floor=0.0)),
         ("all+floor", dict(stat_images=True, xmax_floor=0.01))]
FLOORS = [float(f) for f in os.environ.get("MIG_FLOORS", "").split(",") if f]   # 주면 8장 + 하한 후보만 비교
if FLOORS:
    RULES = [("old8", dict())] + [(f"old8+f{f:g}", dict(xmax_floor=f)) for f in FLOORS]


class Stop(Exception):
    pass


def migs(root):
    return {n: m.mig.detach().flatten().float().cpu() for n, m in root.named_modules()
            if type(m).__name__ == "QuantConv2d" and getattr(m, "mig", None) is not None}


def run_rules(fn, model_module, calib, call):
    """call(fn, module, kw) -> M 적용. 규칙마다 FP 복사본에 적용하고 (α 목록, s 사전, 시간, 메모리)를 돌려준다."""
    out = {}
    for name, kw in RULES:
        mm = copy.deepcopy(model_module)
        kw = dict(kw)
        if kw.get("stat_images"):
            kw["stat_images"] = calib
        torch.cuda.synchronize(DEV); torch.cuda.reset_peak_memory_stats(DEV)
        base = torch.cuda.memory_allocated(DEV)
        buf = io.StringIO(); t0 = time.time()
        with contextlib.redirect_stdout(buf):
            call(fn, mm, kw)
        torch.cuda.synchronize(DEV)
        dt, peak = time.time() - t0, (torch.cuda.max_memory_allocated(DEV) - base) / 2**30
        alphas = re.findall(r"a=\s*([-0-9.]+)", buf.getvalue())
        out[name] = (alphas, migs(mm), dt, peak)
        del mm
        print(f"  [{name}] M {dt:.0f}s, GPU 추가 최대 {peak:.2f} GiB, 단위 {len(alphas)}개 중 이전 {sum(a != '-' for a in alphas)}개",
              flush=True)
    a0, s0 = out["old8"][0], out["old8"][1]
    for name, _ in RULES[1:]:
        a1, s1 = out[name][0], out[name][1]
        ch = [(i, x, y) for i, (x, y) in enumerate(zip(a0, a1)) if x != y]
        keys = set(s0) | set(s1)
        onoff = sum((k in s0) != (k in s1) for k in keys)
        rel = [float((s1[k] / s0[k] - 1).abs().max()) for k in keys if k in s0 and k in s1]
        rel.sort()
        print(f"  old8 -> {name}: α 바뀐 단위 {len(ch)}/{len(a0)} {ch[:12]}, 이전 on/off 바뀐 conv {onoff}, "
              f"s 최대 상대변화 중앙값 {rel[len(rel)//2] if rel else 0:.3f} / 최대 {rel[-1] if rel else 0:.3g}", flush=True)
    return out


def yolo():
    import run_comparison as rc
    from quant import channel_graph
    from model_zoo import model_factory
    calib = [rc.preprocess(p, 640) for p in TRAIN[:256]]
    orig = channel_graph.search_and_apply_tied

    def patched(model_module, images, device, alphas, verbose=True, rule="tied", **_):
        run_rules(orig, model_module, calib,
                  lambda f, mm, kw: f(mm, images, device, alphas, verbose=True, rule=rule, **kw))
        raise Stop
    channel_graph.search_and_apply_tied = patched
    names = rc.load_names("coco")
    cls = model_factory(MODEL)
    fp = rc.build(cls, MODEL, names, DEV, calib, "fp")
    try:                       # 본 실험(brecq+GPM)과 같은 설정; naive로 M 직전까지만 간다
        rc.build(cls, MODEL, names, DEV, calib, "naive", fp=fp, w_bits=WB, a_bits=AB, skip_head=False,
                 first_last_bits=8, last_abits=16, attn_quant="attn_cls", migrate="M",
                 gate_commute=not MODEL.startswith("yoloe"))
    except Stop:
        pass


def owl():
    import run_owl
    from owl_zoo import preprocess
    calib = [preprocess(p)[0] for p in TRAIN[:256]]
    orig = run_owl.migrate

    def patched(m, images, device, alphas, rule="tied", verbose=True, **_):
        run_rules(orig, m, calib, lambda f, mm, kw: f(mm, images, device, alphas, rule=rule, verbose=True, **kw))
        raise Stop
    run_owl.migrate = patched
    try:
        run_owl.build_q("naive+M", WB, AB, calib, DEV, prot=())
    except Stop:
        pass


def gdino():
    from gdino_zoo import build_gdino, preprocess
    from gdino_quant import apply_protocol, migrate, mig_units, qconv
    from run_comparison import load_names, MIG_ALPHAS, MIG_IMAGES
    calib = [preprocess(p)[0] for p in TRAIN[:256]]
    held = [preprocess(p)[0] for p in TRAIN[256:456]]
    m = build_gdino(DEV); m.set_classes(load_names("coco"))
    apply_protocol(m, WB, AB, attn_bits=8, p_bits=16)
    res = run_rules(migrate, m, calib, lambda f, mm, kw: f(mm, calib[:MIG_IMAGES], DEV, MIG_ALPHAS, rule="tied",
                                                           verbose=True, **{"stat_images": None, "xmax_floor": 0.0, **kw}))
    # calibration 밖 200장: 단위 입력 max|x|가 s 기준 max|x|(규칙별)를 몇 배 넘나
    units = mig_units(m)
    xin = [None] * len(units); xh = [None] * len(units); x8 = [None] * len(units)

    def hook(buf, k):
        def h(_m, inp):
            a = inp[0].detach().abs().amax(dim=(0, 2, 3)).float()
            buf[k] = a if buf[k] is None else torch.maximum(buf[k], a)
        return h
    for buf, imgs in ((x8, calib[:MIG_IMAGES]), (xin, calib), (xh, held)):
        hs = [qconv(m.get_submodule(u[0])).register_forward_pre_hook(hook(buf, k)) for k, u in enumerate(units)]
        with torch.no_grad():
            for t in imgs:
                m(t.to(DEV))
        for h in hs:
            h.remove()
    for src, ref_x in (("8장", x8), ("256장", xin)):
        for fl in [0.0] + (FLOORS or [0.01]):
            worst = []
            for a, b in zip(ref_x, xh):
                ref = a.clamp(min=max(1e-5, fl * float(a.max())))
                worst.append(float((b / ref).max()))
            bad = sum(w > 10 for w in worst)
            print(f"  [held-out 200장] 기준 {src}, 하한 {fl}: 단위별 max(밖 max|x| / 기준 max|x|) 최대 {max(worst):.3g}, "
                  f"10배 넘는 단위 {bad}/{len(units)}", flush=True)


print(f"=== {MODEL} W{WB}A{AB}", flush=True)
{"owl": owl, "gdino": gdino}.get(MODEL, yolo)()
