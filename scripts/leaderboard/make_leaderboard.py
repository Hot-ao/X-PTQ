"""실험 리더보드 생성기.

runs/*/*.log(run_comparison 출력)를 모두 읽어 results/LEADERBOARD.html과 results/leaderboard.csv를 만든다.
사용:  .venv/bin/python scripts/leaderboard/make_leaderboard.py
      (환경변수 PTQ_LEADERBOARD=1이면 run_comparison.py가 끝날 때 자동으로 부른다. 기본은 꺼짐)

- 각 로그의 `[args] {...}` 줄에서 모델, 비트, 프로토콜, seed를 읽고, 결과 표(COCO AP | ..., Heval_flip | ...)를 읽는다.
- 같은 (모델, 프로토콜, 설정, 조건, seed)가 여러 로그에 있으면 가장 최근 로그를 쓴다.
- 그룹(모델 × 프로토콜 × 설정)마다 조건별 seed 평균 ± 표준편차, BRECQ 대비 짝 Δ(공통 seed, 평균 ± 95% CI)를 낸다.
- 진행 중 작업은 runs/*/chain.log에서 start는 있고 done이 없는 태그로 판단한다.
- runs/_archive, old_* 폴더(예: LSQ 버그 전 결과)는 제외한다.
- 11-02: run_owl.py·run_gdino.py 로그([json] 줄)도 읽는다. 페이지는 실험 대기열 / 주요 표(고정 칸, 끝나면 채워짐) /
  결과 분류(중요도 자동 판단, 표 배정은 페이지에서 바꾸고 table_assign.json으로 고정) 세 탭이다.
- 모델 크기: 10-03 14:25 이전에 시작한 실행의 로그는 8bit attention Linear weight가 빠진 값이라
  (그 뒤 코드는 포함) 모델별 상수(ATTN_LINEAR_MIB)를 더해 맞춘다. 시작 시각은 chain.log, 없으면 로그 mtime.
"""
from __future__ import annotations
import ast, csv, glob, json, math, os, re, statistics as st, time
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "results")   # 10-27: 결과물은 results/
RUNS = os.path.join(ROOT, "runs")
# 10-17: 기간별로 볼 수 있게 이전 저장소(promptcal-ptq)의 runs/도 함께 읽는다. 없으면 건너뛴다.
# 환경변수 PTQ_EXTRA_RUNS(콜론 구분)로 바꿀 수 있고, 빈 문자열이면 이 저장소의 runs/만 읽는다.
EXTRA_RUNS = [d for d in os.environ.get("PTQ_EXTRA_RUNS", os.path.join(ROOT, "..", "promptcal-ptq", "runs")).split(":")
              if d and os.path.isdir(d) and os.path.realpath(d) != os.path.realpath(RUNS)]
RUN_ROOTS = [RUNS] + [os.path.abspath(d) for d in EXTRA_RUNS]
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

COLS_A = ['coco', 's_ap', 'heval_ap', 'lvis', 'apr', 'apc', 'apf']
COLS_B = ['heval_flip', 'top1', 'upir', 'lost', 'corr', 'lvis_flip', 'lvis_lost', 'lcorr', 't']
SCALE100 = {'lvis', 'apr', 'apc', 'apf'}
KEYS = ['coco', 'lvis', 'apr', 'apc', 'apf', 'lvis_flip', 'lvis_lost', 'heval_ap', 'heval_flip', 'top1', 'lost']


def parse_tables(txt):
    out = {}
    for blk, cols in (('COCO AP |', COLS_A), ('Heval_flip |', COLS_B)):
        if blk not in txt:
            continue
        lines = txt[txt.index(blk):].split('\n')[1:]
        for ln in lines:
            if not ln.strip() or ln.startswith('='):
                break
            parts = [p.strip() for p in ln.split('|')]
            vals = []
            for p in parts[1:]:
                try:
                    vals.append(float(p.rstrip('%')))
                except ValueError:
                    vals.append(None)
            d = out.setdefault(parts[0], {})
            for c, v in zip(cols, vals):
                if v is not None:
                    d[c] = v * 100 if c in SCALE100 else v
    return out


def protocol_of(a):
    if a.get('skip_head', True):
        base = 'head 제외'
    else:
        aq, la = a.get('attn_quant', 'none'), a.get('last_abits', 0)
        if aq == 'attn_cls' and la == 16:
            base = '확정 (attn8+A16)'
        elif aq != 'none':
            base = f'{aq} 8bit (마지막 A8)'
        elif la:
            base = f'conv만 + A{la}'
        else:
            base = 'conv만 (09-30)'
    extra = []
    if a.get('w_bits', 8) < 8 and not a.get('first_last_bits', 0):
        extra.append('첫·마지막 미고정')
    if a.get('calib', 256) != 256:
        extra.append(f"calib {a.get('calib')}")
    if a.get('recon_iters_strong', 2000) != 2000:
        extra.append(f"iter {a.get('recon_iters_strong')}")
    if a.get('first_last_bits', 0) == 16:
        extra.append('첫·마지막 FP')
    if a.get('calib_source', 'coco') != 'coco':
        extra.append(f"calib {a.get('calib_source')}")
    if a.get('eval_cap', 0):
        extra.append(f"평가 {a.get('eval_cap')}장")
    if not a.get('deterministic', False):
        extra.append('비결정적')
    return base + (' · ' + ', '.join(extra) if extra else '')


# 11-02: P·M 규칙. 결과마다 "기존 규칙" 꼬리표를 단다(빈 목록 = 현재 규칙이거나 규칙과 무관한 조건).
RULE_TIME = time.mktime((2026, 11, 2, 9, 0, 0, 0, 0, -1))   # MCHECK_RATIO 1.5 -> 1.0


def rule_tags(cond, a, mtime):
    fl = cond.split('+', 1)[1] if '+' in cond else ''
    tags = []
    if any(x in fl for x in 'PRH'):
        b = a.get('protect_budget', 0.015)
        if abs(b - 0.015) > 1e-9:
            tags.append(f'P 예산 {b * 100:g}%')
    if 'G' in fl and 'P' in fl and not a.get('protect_refill_g', False):
        tags.append('P 기존')
    if 'M' in fl:
        if a.get('mig_stat', '8') != 'all':
            tags.append('M 8장')
        xf = a.get('xmax_floor', 0.0)
        if abs(xf - 0.003) > 1e-9:
            tags.append(f'하한 {xf * 100:g}%')
        if fl.endswith('A') and mtime < RULE_TIME:
            tags.append('ρ 1.5')
    return tags


def collect_json_logs():
    """run_owl.py / run_gdino.py 로그(마지막 [json] 줄)를 allrecs와 같은 형식으로 읽는다."""
    out = []
    starts = start_times()
    files = [f for r in RUN_ROOTS for f in glob.glob(os.path.join(r, '*', '*.log'))
             if '/_archive' not in f and '/old_' not in f]
    for f in files:
        try:
            txt = open(f, errors='ignore').read()
        except OSError:
            continue
        js = re.findall(r'^\[json\] (\{.*\})\s*$', txt, re.M)
        m = re.search(r'^\[args\] (\{.*\})\s*$', txt, re.M)
        if not js or not m:
            continue
        try:
            a = ast.literal_eval(m.group(1))
            res = json.loads(js[-1])
        except Exception:
            continue
        if 'diag_n' in a:                                   # run_gdino.py
            model = 'gdino-tiny'
            st0, cap = a.get('lvis_start', 0), a.get('lvis_cap', 0)
            proto = 'GDINO' + (f' · LVIS {st0}–{st0 + cap}' if cap else '')
        elif 'no_qkv_quant' in a or 'swap_eval' in a:       # run_owl.py
            model = 'owlv2-b16'
            extra = [k for k in ('no_qkv_quant', 'swap_eval') if a.get(k)]
            proto = 'OWL' + (' · ' + ', '.join(extra) if extra else '')
        else:
            continue
        if a.get('eval_cap', 0):
            proto += f" · 평가 {a['eval_cap']}장"
        mtime = os.path.getmtime(f)
        rel = os.path.relpath(f, ROOT)
        setting = f"W{a.get('w_bits')}A{a.get('a_bits')}"
        for c, met in res.items():
            v = {k: met[k] for k in KEYS if isinstance(met.get(k), (int, float)) and not math.isnan(met[k])}
            if not v:
                continue
            out.append(dict(m=model, p=proto, s=setting, c=c, seed=-1 if c == 'FP32' else int(a.get('seed', 0)),
                            t=round(starts.get(f, mtime)), end=round(mtime), src=os.path.dirname(rel),
                            size=met.get('size_mib'), build=None, tags=rule_tags(c, a, mtime), v=v))
    return out


def cond_name(c, a):
    if c.startswith('combined'):
        return f"{c}(s1={a.get('combined_stage1', 'none')})"
    if c == 'adaround':
        return f"adaround({a.get('adaround_act_observer', 'minmax')})"
    return c


# 10-03 14:25(커밋 "model size includes 8-bit attention Linear weights") 전에 시작한 실행은 로그의 "이론적 크기"에
# attention Linear weight(8bit, --attn-quant를 켰을 때)가 빠져 있다. 조건·비트 설정과 무관한 모델별 상수라 더해서 맞춘다.
SIZE_FIX_TIME = time.mktime((2026, 10, 3, 14, 25, 0, 0, 0, -1))
ATTN_LINEAR_MIB = {'yolov8s-world': 0.44, 'yolov8m-world': 0.49, 'yolov8s-worldv2': 0.0}


def start_times():
    """{로그 경로: 시작 시각} -- runs/*/chain.log의 "<태그> start gpuN mm-dd HH:MM:SS" 줄에서."""
    out = {}
    for ch in [c for r in RUN_ROOTS for c in glob.glob(os.path.join(r, '*', 'chain.log'))]:
        for m in re.finditer(r'^(\S+) start gpu\d+ (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)', open(ch, errors='ignore').read(), re.M):
            mo, d, h, mi, s = map(int, m.groups()[1:])
            out[os.path.join(os.path.dirname(ch), m.group(1) + '.log')] = time.mktime((2026, mo, d, h, mi, s, 0, 0, -1))
    return out


def fix_size(size, model, a, started):
    if size is None or a.get('attn_quant', 'none') == 'none' or started >= SIZE_FIX_TIME:
        return size
    return round(size + ATTN_LINEAR_MIB.get(model, 0.0), 2)


def collect():
    recs = {}
    allrecs = []            # 10-17: 기간 필터용. 같은 칸이 여러 번 돌았으면 모두 남긴다(페이지가 기간 안의 최신을 고른다).
    starts = start_times()
    files = [f for r in RUN_ROOTS for f in glob.glob(os.path.join(r, '*', '*.log'))
             if '/_archive' not in f and '/old_' not in f]
    n_ok = 0
    for f in files:
        try:
            txt = open(f, errors='ignore').read()
        except OSError:
            continue
        m = re.search(r'^\[args\] (\{.*\})\s*$', txt, re.M)
        if not m or '이론적 모델 크기' not in txt:
            continue
        try:
            a = ast.literal_eval(m.group(1))
        except Exception:
            continue
        res = parse_tables(txt)
        if not res:
            continue
        n_ok += 1
        tfm = re.search(r'^\[train-flip\] calib 밖 flip\(%\): (.*)$', txt, re.M)
        tf = {c: float(v) for c, v in re.findall(r'([\w+]+) ([\d.]+)', tfm.group(1))} if tfm else {}
        sizes = dict(re.findall(r'^\s+(\S+) 빌드 [\d.]+s\s+\(이론적 크기 ([\d.]+) MiB\)', txt, re.M))
        builds = dict(re.findall(r'^\s+(\S+) 빌드 ([\d.]+)s', txt, re.M))
        model = os.path.splitext(os.path.basename(a.get('model', '?')))[0]
        setting = f"W{a.get('w_bits')}A{a.get('a_bits')}"
        proto = protocol_of(a)
        mtime = os.path.getmtime(f)
        rel = os.path.relpath(f, ROOT)
        for c, met in res.items():
            if c == 'FP32':
                key = (model, proto, setting, 'FP32', -1)
            else:
                key = (model, proto, setting, cond_name(c, a), int(a.get('seed', 0)))
            size = fix_size(float(sizes[c]) if c in sizes else None, model, a, starts.get(f, mtime))
            build_s = float(builds[c]) if c in builds else None
            v = {k: met.get(k) for k in KEYS if met.get(k) is not None}
            if c in tf:
                v['tf'] = tf[c]
            allrecs.append(dict(m=model, p=proto, s=setting, c=key[3], seed=key[4], t=round(starts.get(f, mtime)),
                                end=round(mtime), src=os.path.dirname(rel), size=size, build=build_s,
                                tags=rule_tags(c, a, mtime), v=v))
            if key in recs and recs[key]['mtime'] >= mtime:
                continue
            recs[key] = dict(metrics=met, mtime=mtime, src=rel, size=size, build_s=build_s)
    return recs, allrecs, n_ok, len(files)


def mean_sd(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None
    return st.mean(v), (st.stdev(v) if len(v) > 1 else None)


def paired(A, B, k):
    ss = sorted(s for s in A if s in B and A[s].get(k) is not None and B[s].get(k) is not None)
    if not ss:
        return None
    x = [A[s][k] - B[s][k] for s in ss]
    m = st.mean(x)
    ci = T95.get(len(x) - 1, 1.96) * st.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else None
    return dict(m=m, ci=ci, n=len(x))




def build_groups(recs):
    by = defaultdict(lambda: defaultdict(dict))
    fp = {}
    meta = defaultdict(dict)
    for (model, proto, setting, cond, seed), r in recs.items():
        if cond == 'FP32':
            fp[model] = r['metrics']
            continue
        by[(model, proto, setting)][cond][seed] = r['metrics']
        meta[(model, proto, setting, cond)].setdefault('size', r['size'])
        meta[(model, proto, setting, cond)].setdefault('build', []).append(r['build_s'])
        meta[(model, proto, setting, cond)].setdefault('src', set()).add(os.path.dirname(r['src']))
        meta[(model, proto, setting, cond)]['mtime'] = max(meta[(model, proto, setting, cond)].get('mtime', 0), r['mtime'])
    groups = []
    for (model, proto, setting), conds in by.items():
        rows = []
        base_b = conds.get('brecq')
        for cond, seeds in conds.items():
            row = dict(cond=cond, seeds=sorted(seeds), n=len(seeds), stats={}, per_seed={})
            for k in KEYS:
                m, sd = mean_sd([seeds[s].get(k) for s in seeds])
                if m is not None:
                    row['stats'][k] = [m, sd]
            for s in seeds:
                row['per_seed'][s] = {k: seeds[s].get(k) for k in ('coco', 'lvis', 'lvis_flip')}
            cocos = [seeds[s].get('coco') for s in seeds if seeds[s].get('coco') is not None]
            if len(cocos) >= 3:
                med = st.median(cocos)
                row['collapse'] = [s for s in seeds if seeds[s].get('coco') is not None and seeds[s]['coco'] < med - 5]
            else:
                row['collapse'] = []
            mt = meta[(model, proto, setting, cond)]
            row['size'] = mt.get('size')
            b = [x for x in mt.get('build', []) if x]
            row['build_min'] = st.mean(b) / 60 if b else None
            row['src'] = sorted(mt.get('src', []))
            row['mtime'] = mt.get('mtime')
            base_name = cond.split('+')[0] if '+' in cond else None
            row['d_brecq'] = {k: paired(seeds, base_b, k) for k in ('coco', 'lvis', 'lvis_flip')} if base_b and cond != 'brecq' else {}
            bb = conds.get(base_name) if base_name and base_name != 'brecq' else None
            row['d_base'] = ({k: paired(seeds, bb, k) for k in ('coco', 'lvis', 'lvis_flip')} if bb else row['d_brecq'])
            row['base_name'] = base_name if bb else ('brecq' if row['d_brecq'] else None)
            rows.append(row)
        groups.append(dict(model=model, protocol=proto, setting=setting, rows=rows,
                           fp=fp.get(model), mtime=max(r['mtime'] or 0 for r in rows)))
    return groups


def _elapsed(info):
    """'gpu4 10-17 21:44:10' -> ('gpu4', '1.3h'). 형식이 다르면 (info, '')."""
    m = re.match(r'(\S+)\s+(\d\d-\d\d \d\d:\d\d:\d\d)', info)
    if not m:
        return info, ''
    t = time.mktime(time.strptime(f"{time.localtime().tm_year}-{m.group(2)}", '%Y-%m-%d %H:%M:%S'))
    return m.group(1), f"{max(time.time() - t, 0) / 3600:.1f}h"


def running_jobs():
    """10-17: queue.tsv나 chain.log가 있는 run 디렉터리를 모두 본다. 아직 아무 작업도 시작하지 않은 대기열은
    state='대기', 하나라도 돌고 있으면 '진행', 다 끝났으면 목록에서 뺀다. 대기열 밖 작업(scripts/track.sh로 띄운 것)도
    chain.log에 같은 형식으로 남으므로 여기에 잡힌다(queue.tsv가 없으면 total은 None)."""
    out = []
    dirs = {os.path.dirname(p) for p in glob.glob(os.path.join(RUNS, '*', 'chain.log'))
            + glob.glob(os.path.join(RUNS, '*', 'queue.tsv'))}
    for d in dirs:
        if os.path.basename(d).startswith(('_archive', 'old_')):
            continue
        started, done, failed = {}, set(), set()
        ch = os.path.join(d, 'chain.log')
        if os.path.exists(ch):
            for ln in open(ch, errors='ignore'):
                p = ln.split()
                if len(p) >= 2 and p[1] == 'start':
                    started[p[0]] = ' '.join(p[2:])
                elif len(p) >= 2 and p[1] == 'done':
                    done.add(p[0])
                    if len(p) >= 3 and p[2].startswith('rc=') and p[2] != 'rc=0':
                        failed.add(p[0])
        q = os.path.join(d, 'queue.tsv')
        total = sum(1 for ln in open(q) if ln.strip()) if os.path.exists(q) else None
        run = [(t, v) for t, v in started.items() if t not in done]
        if not run and not (total and len(done) < total):
            continue
        jobs = []
        for t, v in run:
            gpu, el = _elapsed(v)
            jobs.append(dict(tag=t, info=v, gpu=gpu, elapsed=el))
        out.append(dict(dir=os.path.relpath(d, ROOT), done=len(done), total=total, failed=len(failed),
                        state='진행' if run else '대기', running=jobs))
    return sorted(out, key=lambda r: (r['state'] != '진행', r['dir']))


# ---------------------------------------------------------------------------------------------------------------
# 11-02 페이지 개편: 결과 묶음(모델 × 프로토콜 × 비트 × 조건 × 규칙) + 중요도·표 배정 + 고정 표 + 대기열 상세
# ---------------------------------------------------------------------------------------------------------------
MAIN_PROTOS = {'확정 (attn8+A16)', 'OWL', 'GDINO · LVIS 0–1000', 'GDINO · LVIS 0–2000', 'GDINO'}
def is_main(p):
    return p in MAIN_PROTOS or p.startswith('GDINO')


MODELS = [('yolov8s-world', 'YOLO-World-S'), ('yolov8s-worldv2', 'YOLO-World-S v2'), ('yolov8m-world', 'YOLO-World-M'),
          ('yolov8l-world', 'YOLO-World-L'), ('yoloe-v8s-seg', 'YOLOE-v8s'), ('yoloe-11s-seg', 'YOLOE-11s'),
          ('owlv2-b16', 'OWLv2-B/16'), ('gdino-tiny', 'Grounding DINO-T')]
MODEL_NAME = dict(MODELS)
SETTINGS = ['W8A8', 'W4A8', 'W4A6', 'W4A5', 'W4A4']
TABLES = ['주요지표', '추가지표1', '추가지표2', '추가지표3', '미지정', '제외']
TABLE_DESC = {'주요지표': '본문 표 1·2: 기준선과 우리 방법', '추가지표1': 'ablation (G/P/M 조합, 보호 대조, 이전 규칙 교체)',
              '추가지표2': '일반화 보조 (QDrop 위, 다른 calibration, 첫·마지막 FP)', '추가지표3': '분석·민감도 (P 예산, xmax, 하한)',
              '미지정': '판단 보류', '제외': '대체됐거나 프로토콜이 다른 결과'}
BASELINES = {'naive', 'brecq', 'qdrop', 'adaround(minmax)', 'adaround(mse)', 'brecq+S', 'brecq+W', 'FP32'}
ABLATION = {'brecq+G', 'brecq+P', 'brecq+M', 'brecq+GM', 'brecq+PM', 'brecq+GP', 'brecq+GPM', 'brecq+R', 'brecq+H',
            'brecq+GPS', 'brecq+GPW', 'naive+S', 'naive+W', 'naive+M', 'brecq+GMA', 'brecq+PMA', 'brecq+GPMA'}


def ours_conds(model):
    """표에서 '우리 방법'으로 보는 조건. YOLO는 GP/GPM 중 사전 검사(ρ=1.0)로 고른다."""
    if model == 'owlv2-b16':
        return {'brecq+PM', 'brecq+M'}
    if model == 'gdino-tiny':
        return {'brecq+PM', 'brecq+M'}
    return {'brecq+GP', 'brecq+GPM', 'brecq+GPMA'}


def group_key(r):
    return '|'.join([r['m'], r['p'], r['s'], r['c'], ','.join(r['tags'])])


def build_results(allrecs):
    """같은 칸(seed)이 여러 번 돌았으면 가장 늦게 끝난 것을 쓴다."""
    latest = {}
    for r in allrecs:
        k = (group_key(r), r['seed'])
        if k not in latest or latest[k]['end'] < r['end']:
            latest[k] = r
    by = defaultdict(dict)
    for (gk, seed), r in latest.items():
        by[gk][seed] = r
    res = {}
    for gk, seeds in by.items():
        r0 = next(iter(seeds.values()))
        stats = {}
        for k in ('coco', 'lvis', 'lvis_flip', 'lvis_lost', 'tf', 'apr'):
            m, sd = mean_sd([s['v'].get(k) for s in seeds.values()])
            if m is not None:
                stats[k] = [m, sd]
        res[gk] = dict(key=gk, m=r0['m'], p=r0['p'], s=r0['s'], c=r0['c'], tags=r0['tags'],
                       seeds=sorted(seeds), per={s: r['v'] for s, r in seeds.items()}, stats=stats,
                       end=max(r['end'] for r in seeds.values()), src=sorted({r['src'] for r in seeds.values()}),
                       size=r0['size'])
    # BRECQ 대비 짝 Δ (같은 모델·프로토콜·비트, 공통 seed)
    base = {(g['m'], g['p'], g['s']): g for g in res.values() if g['c'] == 'brecq' and not g['tags']}
    for g in res.values():
        b = base.get((g['m'], g['p'], g['s']))
        if b and g['c'] not in ('brecq', 'FP32'):
            g['d'] = {k: paired({s: v for s, v in g['per'].items()}, b['per'], k) for k in ('lvis', 'lvis_flip')}
    return res


def judge(g, res):
    """(중요도 3/2/1, 기본 표, 이유). 사용자가 페이지에서 바꾸면 table_assign.json이 우선한다."""
    m, p, c, tags = g['m'], g['p'], g['c'], g['tags']
    if m not in MODEL_NAME:
        return 1, '제외', '논문 대상 모델 아님'
    if c == 'FP32':
        return 1, '제외', 'FP 기준값 (주요 표에 자동 표시)'
    if not is_main(p):
        if '첫·마지막 FP' in p or 'calib o365' in p:
            return 2, '추가지표2', '보조 프로토콜'
        return 1, '제외', '확정 프로토콜과 다름'
    if g['s'] not in SETTINGS:
        return 1, '제외', '논문 비트 설정 아님'
    old = [t for t in tags if t in ('P 기존', 'M 8장', 'ρ 1.5') or t.startswith('하한')]
    sweep = [t for t in tags if t.startswith('P 예산')]
    if old:
        newer = any(o['m'] == m and o['p'] == p and o['s'] == g['s'] and o['c'] == c and not o['tags'] for o in res.values())
        if newer:
            return 1, '제외', '새 규칙 결과로 대체'
        if sweep or (tags == ['M 8장'] and 'G' in c):
            return 2, '추가지표3', '규칙 비교·예산'
        return 2, '미지정', '기존 규칙 (새 규칙 재실험 대기)'
    if sweep:
        return 2, '추가지표3', 'P 예산 sweep'
    if c in BASELINES or c in ours_conds(m):
        return 3, '주요지표', '기준선·우리 방법'
    if c.startswith('qdrop+'):
        return 2, '추가지표2', 'QDrop 위 plug-in'
    if c in ABLATION:
        return 2, '추가지표1', 'ablation'
    return 1, '미지정', '기타 조건'


def pick_ours(res, m, s):
    """'우리 방법' 칸: 새 규칙 GP·GPM을 seed마다 train flip(ρ=1.0)으로 고른다. OWL·GDINO는 고정 조건."""
    def g(c, proto=None):
        cands = [only_seed(x) for x in res.values() if x['m'] == m and x['s'] == s and x['c'] == c and not x['tags']
                 and (x['p'] == proto if proto else is_main(x['p']))]
        cands = [x for x in cands if x]
        return max(cands, key=lambda x: x['end']) if cands else None
    if m == 'owlv2-b16':
        x = g('brecq+M' if s == 'W8A8' else 'brecq+PM')
        return (x, x and x['c'])
    if m == 'gdino-tiny':
        x = g('brecq+PM')
        return (x, x and x['c'])
    gp, gpm = g('brecq+GP'), g('brecq+GPM')
    if not gp or not gpm:
        return (None, None)
    per, pick, ends = {}, {}, []
    for seed in sorted(set(gp['per']) & set(gpm['per'])):
        a, b = gp['per'][seed], gpm['per'][seed]
        if a.get('tf') is None or b.get('tf') is None:
            continue
        use_m = b['tf'] <= a['tf']
        per[seed] = b if use_m else a
        pick[seed] = 'GPM' if use_m else 'GP'
    if not per:
        return (None, None)
    stats = {k: list(mean_sd([v.get(k) for v in per.values()])) for k in ('coco', 'lvis', 'lvis_flip')}
    x = dict(m=m, s=s, c='ours', per=per, seeds=sorted(per), stats=stats, end=max(gp['end'], gpm['end']),
             src=sorted(set(gp['src']) | set(gpm['src'])), tags=[], p=gp['p'])
    return (x, ', '.join(f's{k}:{v}' for k, v in pick.items()))


def cell(x, base=None, note=None):
    if not x:
        return None
    st_ = x['stats']
    out = dict(lvis=st_.get('lvis', [None])[0], flip=st_.get('lvis_flip', [None])[0], coco=st_.get('coco', [None])[0],
               sd=st_.get('lvis', [None, None])[1], n=len(x['seeds']), end=x['end'], src=x['src'], note=note,
               old=bool(x['tags']), tags=x['tags'])
    if base:
        d = paired(x['per'], base['per'], 'lvis')
        if d:
            out['d'] = [d['m'], d['ci'], d['n']]
    return out


MAIN_SEED = 0   # 11-03: 주요 표는 seed 0 하나로 통일한다(결과 분류 탭은 그대로 seed 평균)


def only_seed(x, seed=MAIN_SEED):
    """결과 묶음 x에서 seed 하나만 남긴 사본. 그 seed가 없으면 None."""
    if x and x['c'] == 'FP32' and seed not in x['per'] and x['per']:
        seed = min(x['per'])                         # FP는 seed와 무관
    if not x or seed not in x['per']:
        return None
    v = x['per'][seed]
    stats = {k: [v[k], None] for k in ('coco', 'lvis', 'lvis_flip', 'lvis_lost', 'tf', 'apr') if v.get(k) is not None}
    return dict(x, per={seed: v}, seeds=[seed], stats=stats)


def best(res, m, s, c, proto=None):
    """새 규칙(꼬리표 없음)을 먼저, 없으면 기존 규칙을 쓴다(페이지에서 회색으로 표시). 주요 표용이라 seed 0만 본다."""
    cands = [only_seed(x) for x in res.values() if x['m'] == m and x['s'] == s and x['c'] == c
             and (x['p'] == proto if proto else is_main(x['p']))
             and not any(t.startswith('P 예산') for t in x['tags'])]
    cands = [x for x in cands if x]
    if not cands:
        return None
    return min(cands, key=lambda x: (bool(x['tags']), -x['end']))


def fixed_tables(res, pending):
    """고정 표: 칸이 비어 있으면 대기열에 같은 실험이 있는지 표시한다."""
    S = 'yolov8s-world'
    rows1 = [('FP32', 'FP32'), ('RTN', 'naive'), ('AdaRound', 'adaround(mse)'), ('BRECQ', 'brecq'), ('QDrop', 'qdrop'),
             ('BRECQ + SmoothQuant', 'brecq+S'), ('BRECQ + AWQ', 'brecq+W'), ('Ours (BRECQ, ρ=1.0 선택)', 'ours'),
             ('  └ GP', 'brecq+GP'), ('  └ GPM', 'brecq+GPM'), ('Ours (QDrop + GPM)', 'qdrop+GPM')]
    t1 = []
    for label, c in rows1:
        row = dict(label=label, cond=c, cells={})
        for s in SETTINGS:
            b = best(res, S, s, 'brecq')
            if c == 'FP32':
                x = best(res, S, 'W8A8', 'FP32') or best(res, S, s, 'FP32')
                row['cells'][s] = cell(x)
                continue
            if c == 'ours':
                x, note = pick_ours(res, S, s)
            else:
                x, note = best(res, S, s, c), None
            row['cells'][s] = merge(cell(x, b if c != 'brecq' else None, note), pend(pending, S, s, 'brecq+GPM' if c == 'ours' else c))
        t1.append(row)
    t2 = []
    for m, name in MODELS:
        row = dict(label=name, model=m, cells={})
        for s in ('W8A8', 'W4A8', 'W4A5', 'W4A4'):
            proto = None
            if m == 'gdino-tiny':
                for pr in ('GDINO · LVIS 0–2000', 'GDINO · LVIS 0–1000'):
                    if best(res, m, s, 'brecq', pr):
                        proto = pr
                        break
            b = best(res, m, s, 'brecq', proto)
            q = best(res, m, s, 'qdrop', proto)
            if m == 'gdino-tiny':
                x = best(res, m, s, 'brecq+PM', proto)
                note = 'brecq+PM'
            else:
                x, note = pick_ours(res, m, s)
            fp = best(res, m, s, 'FP32', proto) or next((best(res, m, s2, 'FP32', proto) for s2 in SETTINGS
                                                         if best(res, m, s2, 'FP32', proto)), None)
            if not x and m not in ('owlv2-b16', 'gdino-tiny'):   # 새 규칙 결과가 없으면 기존 결과(회색)
                olds = [y for y in (best(res, m, s, c) for c in ('brecq+GPMA', 'brecq+GPM', 'brecq+PMA', 'brecq+GP')) if y]
                x = max(olds, key=lambda y: len(y['seeds'])) if olds else None
                note = x and x['c']
            ours = merge(cell(x, b, note), pend(pending, m, s, 'brecq+GPM' if m not in ('owlv2-b16', 'gdino-tiny') else
                                                 ('brecq+M' if s == 'W8A8' else 'brecq+PM')))
            if not (b or ours):
                continue
            row['cells'][s] = dict(fp=cell(fp), brecq=cell(b), qdrop=cell(q), ours=ours,
                                   proto=proto.split('· ')[-1] if proto else None)
        if row['cells']:
            t2.append(row)
    abl = ['brecq', 'brecq+G', 'brecq+P', 'brecq+M', 'brecq+GP', 'brecq+GM', 'brecq+PM', 'brecq+GPM',
           'brecq+R', 'brecq+H', 'brecq+GPS', 'brecq+GPW']
    t3 = []
    for c in abl:
        row = dict(label=c, cond=c, cells={})
        for s in ('W4A8', 'W4A5', 'W4A4'):
            b = best(res, S, s, 'brecq')
            row['cells'][s] = merge(cell(best(res, S, s, c), b if c != 'brecq' else None), pend(pending, S, s, c))
        t3.append(row)
    return dict(t1=t1, t2=t2, t3=t3, settings=SETTINGS)


def merge(c, p):
    """값이 없으면 대기 표시, 기존 규칙 값이면 재실험 대기 표시를 덧붙인다."""
    if c is None:
        return p
    if p and c.get('old'):
        c['pending'] = p['pending']
    return c


def pend(pending, m, s, c):
    st_ = pending.get((m, s, c))
    return dict(pending=st_) if st_ else None


QARGS = re.compile(r'--(model|w-bits|a-bits|conditions|seed|first-last-bits|load-model|odinw-only)(?:[ =](\S+))?')


def queue_details():
    """runs/*/queue.tsv와 chain.log로 작업별 상태를 만든다. 최근 3일 안에 움직였거나 남은 작업이 있는 폴더만."""
    dirs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(RUNS, '*', 'chain.log'))
                   + glob.glob(os.path.join(RUNS, '*', 'queue.tsv'))})
    out, pending = [], {}
    now = time.time()
    try:   # 작업자(worker.sh, gpu.sh, track.sh)가 살아 있는 폴더. 없으면 남은 작업은 '정지'로 본다
        import subprocess
        ps = subprocess.run(['ps', '-eo', 'args'], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        ps = ''
    alive = set(re.findall(r'runs/([\w.-]+)', ps))
    for d in dirs:
        name = os.path.basename(d)
        if name.startswith(('_archive', 'old_')):
            continue
        ch = os.path.join(d, 'chain.log')
        events = {}
        if os.path.exists(ch):
            for ln in open(ch, errors='ignore'):
                mm = re.match(r'^(\S+) (start|done)(?: rc=(\d+))? gpu(\d+) (\d\d-\d\d \d\d:\d\d:\d\d)', ln)
                if not mm:
                    continue
                tag, ev, rc, gpu, ts = mm.groups()
                t = time.mktime(time.strptime(f"{time.localtime().tm_year}-{ts}", '%Y-%m-%d %H:%M:%S'))
                e = events.setdefault(tag, {})
                if ev == 'start':
                    e.update(start=t, gpu=gpu)
                    e.pop('done', None)
                else:
                    e.update(done=t, rc=int(rc or 0))
        q = os.path.join(d, 'queue.tsv')
        items = []
        if os.path.exists(q):
            for ln in open(q, errors='ignore'):
                if '\t' not in ln:
                    continue
                tag, args = ln.rstrip('\n').split('\t', 1)
                items.append((tag, args))
        tagged = {t for t, _ in items}
        items += [(t, '') for t in events if t not in tagged]
        taken = set(open(os.path.join(d, 'taken.log')).read().split()) if os.path.exists(os.path.join(d, 'taken.log')) else set()
        paused = set(open(os.path.join(d, 'paused_tags.txt')).read().split()) if os.path.exists(os.path.join(d, 'paused_tags.txt')) else set()
        jobs = []
        for tag, args in items:
            e = events.get(tag, {})
            if 'done' in e:
                state = '완료' if e.get('rc', 0) == 0 else '실패'
            elif 'start' in e:
                state = '진행'
            elif tag in paused or (tag in taken and not e):
                state = '보류'
            else:
                state = '대기'
            if name not in alive and state in ('대기', '진행'):
                state = '정지'
            kv = dict(QARGS.findall(args))
            model = os.path.splitext(os.path.basename(kv.get('model', '')))[0]
            setting = f"W{kv['w-bits']}A{kv['a-bits']}" if 'w-bits' in kv and 'a-bits' in kv else ''
            conds = kv.get('conditions', '')
            if not model:   # gpu.sh로 띄운 OWL·GDINO 작업은 태그에서 읽는다
                mt = re.match(r'(owl|gdino)\S*?-(W\dA\d)-(\w+)-s(\d)', tag)
                if mt:
                    model = 'owlv2-b16' if mt.group(1) == 'owl' else 'gdino-tiny'
                    setting, conds = mt.group(2), 'brecq+' + mt.group(3)
            if state in ('대기', '진행') and model and setting and 'odinw-only' not in kv and 'first-last-bits' not in kv:
                for c in conds.split(','):
                    if c:
                        pending.setdefault((model, setting, c), f"{state}{' gpu' + e['gpu'] if state == '진행' else ''} · {name}")
            jobs.append(dict(tag=tag, state=state, model=MODEL_NAME.get(model, model), setting=setting, conds=conds,
                             seed=kv.get('seed', ''), extra=' '.join(k for k in ('odinw-only', 'first-last-bits', 'load-model') if k in kv),
                             gpu=e.get('gpu'), start=e.get('start'), done=e.get('done')))
        if not jobs:
            continue
        last = max([j['done'] or j['start'] or 0 for j in jobs] + [os.path.getmtime(ch) if os.path.exists(ch) else 0])
        active = any(j['state'] in ('대기', '진행') for j in jobs)
        if not active and now - last > 3 * 86400:
            continue
        out.append(dict(dir=name, jobs=jobs, active=active, last=last,
                        n={s: sum(j['state'] == s for j in jobs) for s in ('완료', '진행', '대기', '정지', '보류', '실패')}))
    out.sort(key=lambda r: (not r['active'], -r['last']))
    return out, pending


def gpu_status():
    try:
        import subprocess
        q = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20).stdout
        return [dict(zip(('i', 'name', 'util', 'mem', 'tot'), [p.strip() for p in ln.split(',')])) for ln in q.strip().split('\n') if ln]
    except Exception:
        return []


def main():
    recs, allrecs, n_ok, n_files = collect()
    groups = build_groups(recs)
    allrecs += collect_json_logs()
    res = build_results(allrecs)
    try:
        assign = json.load(open(os.path.join(os.path.dirname(__file__), 'table_assign.json')))
    except (OSError, ValueError):
        assign = {}
    for g in res.values():
        g['imp'], g['tbl'], g['why'] = judge(g, res)
        if g['key'] in assign:
            g['tbl'] = assign[g['key']]
        g['name'] = MODEL_NAME.get(g['m'], g['m'])
    queues, pending = queue_details()
    data = dict(generated=time.strftime('%Y-%m-%d %H:%M'), now=round(time.time()), n_logs=n_ok, n_files=n_files,
                fixed=fixed_tables(res, pending), queues=queues, gpus=gpu_status(), tables=TABLES, table_desc=TABLE_DESC,
                models=[m for m, _ in MODELS], settings=SETTINGS,
                results=sorted([{k: v for k, v in g.items() if k != 'per'} for g in res.values()],
                               key=lambda g: -g['end']))
    csv_tmp = os.path.join(OUT, f'.leaderboard.csv.{os.getpid()}')
    with open(csv_tmp, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['model', 'protocol', 'setting', 'condition', 'n_seeds', 'seeds', 'size_MiB']
                   + [f'{k}_mean' for k in KEYS] + [f'{k}_sd' for k in KEYS]
                   + ['dLVIS_vs_brecq', 'dLVIS_ci', 'dFlip_vs_brecq', 'dFlip_ci', 'collapse_seeds', 'sources'])
        for g in groups:
            for r in g['rows']:
                dl, df = r['d_brecq'].get('lvis'), r['d_brecq'].get('lvis_flip')
                w.writerow([g['model'], g['protocol'], g['setting'], r['cond'], r['n'], ' '.join(map(str, r['seeds'])), r['size']]
                           + [r['stats'].get(k, [None])[0] for k in KEYS] + [(r['stats'].get(k) or [None, None])[1] for k in KEYS]
                           + [dl and dl['m'], dl and dl['ci'], df and df['m'], df and df['ci'],
                              ' '.join(map(str, r['collapse'])), ' '.join(r['src'])])
    os.replace(csv_tmp, os.path.join(OUT, 'leaderboard.csv'))   # 통째로 바꿔 끼움(쓰다 만 파일이 보이지 않게)
    tpl = open(os.path.join(os.path.dirname(__file__), 'leaderboard_template.html')).read()
    html = tpl.replace('/*__DATA__*/null', json.dumps(data, ensure_ascii=False, default=str))
    html_tmp = os.path.join(OUT, f'.LEADERBOARD.html.{os.getpid()}')
    with open(html_tmp, 'w') as fh:
        fh.write(html)
    os.replace(html_tmp, os.path.join(OUT, 'LEADERBOARD.html'))
    print(f"로그 {n_ok}/{n_files}개, 그룹 {len(groups)}개 -> results/LEADERBOARD.html, results/leaderboard.csv")


if __name__ == '__main__':
    main()
