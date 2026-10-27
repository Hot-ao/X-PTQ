"""실험 리더보드 생성기.

runs/*/*.log(run_comparison 출력)를 모두 읽어 results/LEADERBOARD.html과 results/leaderboard.csv를 만든다.
사용:  .venv/bin/python scripts/leaderboard/make_leaderboard.py
      (환경변수 PTQ_LEADERBOARD=1이면 run_comparison.py가 끝날 때 자동으로 부른다. 기본은 꺼짐)

- 각 로그의 `[args] {...}` 줄에서 모델, 비트, 프로토콜, seed를 읽고, 결과 표(COCO AP | ..., Heval_flip | ...)를 읽는다.
- 같은 (모델, 프로토콜, 설정, 조건, seed)가 여러 로그에 있으면 가장 최근 로그를 쓴다.
- 그룹(모델 × 프로토콜 × 설정)마다 조건별 seed 평균 ± 표준편차, BRECQ 대비 짝 Δ(공통 seed, 평균 ± 95% CI)를 낸다.
- 진행 중 작업은 runs/*/chain.log에서 start는 있고 done이 없는 태그로 판단한다.
- runs/_archive, old_* 폴더(예: LSQ 버그 전 결과)는 제외한다.
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
    if not a.get('deterministic', False):
        extra.append('비결정적')
    return base + (' · ' + ', '.join(extra) if extra else '')


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
            allrecs.append(dict(m=model, p=proto, s=setting, c=key[3], seed=key[4], t=round(starts.get(f, mtime)),
                                end=round(mtime), src=os.path.dirname(rel), size=size, build=build_s,
                                v={k: met.get(k) for k in KEYS if met.get(k) is not None}))
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


def main():
    recs, allrecs, n_ok, n_files = collect()
    groups = build_groups(recs)
    # 페이지는 records로 기간 안에서 다시 집계한다(groups는 CSV용, 전체 기간의 최신값).
    data = dict(generated=time.strftime('%Y-%m-%d %H:%M'), now=round(time.time()), n_logs=n_ok, n_files=n_files,
                roots=[os.path.relpath(r, ROOT) for r in RUN_ROOTS], records=allrecs, running=running_jobs())
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
