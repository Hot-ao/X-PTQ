"""현재 규칙 결과의 고정본을 documents/에 뽑는다(11-06).

사용:  .venv/bin/python scripts/leaderboard/export_final.py
출력:  documents/RESULTS_FINAL.md   논문 표 단위로 정리한 표 (손으로 고치지 말고 이 스크립트를 다시 돌린다)
       documents/results_final.csv  seed별 원자료 (한 줄 = 모델 × 프로토콜 × 비트 × 조건 × seed), 칸별 출처(log, 저장 모델)

- 수치는 make_leaderboard.py와 같은 함수(collect → build_results → best / pick_ours / main_cell)로 모은다.
  leaderboard.csv(build_groups)와 달리 규칙 꼬리표로 묶음을 나누므로 기존 규칙 seed가 섞이지 않는다.
- "현재 규칙" = 11-02 규칙(P는 G 뒤 다시 채움, M xmax calibration 전체 + 하한 0.3%, ρ = 1.0). 꼬리표 'P 기존', 'M 8장',
  'ρ 1.5', '하한 x%'가 붙은 결과는 표에서 빼고 §6에 남은 목록만 적는다. 'P 예산 x%'(예산 sweep)는 규칙 위반이 아니라 §5에 둔다.
- GDINO의 LVIS flip은 다른 모델과 정의가 달라(어휘 chunk 안 top-1) 재측정 전까지 md와 csv 모두 비운다.
"""
from __future__ import annotations
import csv, json, os, subprocess, sys, time
from collections import defaultdict

os.environ['PTQ_EXTRA_RUNS'] = ''   # 이전 저장소(promptcal-ptq) runs/는 10-16 결정성 수정 전 결과라 고정본에 넣지 않는다
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_leaderboard as L

DOC = os.path.join(L.ROOT, 'documents')
S = 'yolov8s-world'
S3 = L.DRAFT_SEEDS
YOLO = [m for m, _ in L.MODELS if m not in ('owlv2-b16', 'gdino-tiny')]
OLD = ('P 기존', 'M 8장', 'ρ 1.5')
SUP = {1: '⁽¹⁾', 2: '⁽²⁾', 4: '⁽⁴⁾', 5: '⁽⁵⁾', 6: '⁽⁶⁾'}
GDINO_PROTOS = ('GDINO · LVIS 0–2000', 'GDINO · LVIS 0–1000', 'GDINO')


def is_old(tags):
    return any(t in OLD or t.startswith('하한') for t in tags)


def label(m, pick):
    """YOLOE는 G가 적용되지 않으므로 GP/GPM을 P/PM으로 적는다."""
    if not pick:
        return ''
    pick = pick.split(':')[-1].replace('brecq+', '').replace('qdrop+', '')
    return pick.replace('G', '') if m.startswith('yoloe') else pick


def ap_flip(c, n_ref=3, flip=True):
    if not c or c.get('old'):                    # best()는 새 규칙 값이 없으면 기존 규칙 값을 돌려준다. 고정본에서는 빈칸
        return '–'
    if c.get('lvis') is None:
        return '⏳' if c.get('pending') else '–'
    s = f"{c['lvis']:.2f}"
    if flip and c.get('flip') is not None:
        s += f" / {c['flip']:.1f}"
    if n_ref and c.get('n') != n_ref:
        s += ' ' + SUP.get(c.get('n'), f"⁽{c.get('n')}⁾")
    return s


def delta(c):
    d = c and not c.get('old') and c.get('d')
    if not d:
        return '–'
    return f'{d[0]:+.2f} ± {d[1]:.2f}' if d[1] is not None else f'{d[0]:+.2f}'


def smoke(proto):
    """평가 이미지를 잘라 돌린 점검용 실행(--eval-cap)은 결과로 쓰지 않는다."""
    return '평가 ' in proto


def fpv(fp):
    return f"{fp['stats']['lvis'][0]:.2f}" if fp and fp['stats'].get('lvis') else '–'


class Src:
    """절마다 쓴 run 디렉토리를 모은다."""
    def __init__(self):
        self.by = defaultdict(set)

    def add(self, sec, c):
        if c and c.get('src'):
            self.by[sec].update(c['src'])
        return c


def git_info():
    try:
        h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=L.ROOT, capture_output=True, text=True).stdout.strip()
        st = subprocess.run(['git', 'status', '--porcelain', '-uno'], cwd=L.ROOT, capture_output=True, text=True).stdout
        dirty = [ln[3:] for ln in st.splitlines() if ln[3:].endswith('.py')]
        return h, dirty
    except OSError:
        return '?', []


def gdino(res, s, c):
    for p in GDINO_PROTOS:
        x = L.best(res, 'gdino-tiny', s, c, p, seeds=S3)
        if x and not x['tags']:
            return x, p
    return None, None


def main():
    recs, allrecs, n_ok, n_files = L.collect()
    allrecs += L.collect_json_logs()
    res = L.build_results(allrecs)
    try:
        assign = json.load(open(os.path.join(os.path.dirname(__file__), 'table_assign.json')))
    except (OSError, ValueError):
        assign = {}
    for g in res.values():
        g['imp'], g['tbl'], g['why'] = L.judge(g, res)
        g['tbl'] = assign.get(g['key'], g['tbl'])
    _, pending = L.queue_details()
    src = Src()
    out = []
    w = out.append
    head, dirty = git_info()

    w('# 실험 결과 고정본 (현재 규칙)\n')
    w(f"**생성:** {time.strftime('%Y-%m-%d %H:%M')} · `scripts/leaderboard/export_final.py` · 코드 커밋 `{head}`"
      + (f" (**커밋 안 된 코드 변경 있음:** {', '.join(f'`{d}`' for d in dirty)})" if dirty else '') + '\n')
    w('**대상 독자:** 공저자, 그리고 논문 표를 만드는 사람. 이 파일은 스크립트가 만든다. 손으로 고치지 말고 다시 돌린다. '
      '해석과 논의는 [RESULTS_2026-11-04.md](RESULTS_2026-11-04.md), [PAPER_STRUCTURE_2026-11-05.md](PAPER_STRUCTURE_2026-11-05.md)에 둔다.\n')
    w('**포함 기준**')
    w('- 범위: 이 저장소의 `runs/`만 읽는다(이전 저장소 promptcal-ptq 제외). 평가 이미지를 잘라 돌린 점검용 실행도 뺀다.')
    w('- 규칙: 11-02 규칙만 쓴다. P는 G 대상을 먼저 빼고 1.5% 예산을 다시 채우고(분모 = 양자화 weight 전체), '
      'M은 xmax를 calibration 전체에서 재고(하한 0.3%), 사전 검사는 ρ = 1.0이다. 기존 규칙 결과는 표에 넣지 않는다(§6).')
    w('- 프로토콜: attn_cls(conv, attention, region–텍스트 내적 양자화), 첫·마지막 W8A8, head 마지막 1×1 입력 A16, 재구성 2,000 iters, '
      'calibration 256장, 결정적 실행. OWLv2·GDINO는 각자의 확정 프로토콜이다.')
    w('- 값: **LVIS minival AP(×100) / LVIS top-1 flip(%)**. 표 1–4는 seed 0–2 평균이고, 위첨자 ⁽ⁿ⁾는 seed 수가 3이 아닌 칸이다. '
      'Δ는 같은 seed끼리 짝지은 차이의 평균 ± 95% t 신뢰구간 반폭이다(seed 1개면 ± 없음). ⏳ = 대기열에 있음, – = 없음.')
    w('- Ours: BRECQ 위 G+P(+M). seed 0의 calibration 밖 검사(ρ = 1.0)로 GP / GPM 중 하나를 골라 모든 seed에 쓴다. '
      'QDrop + Ours는 같은 선택을 QDrop 위에 쓴다. YOLOE는 G가 없으므로 P / PM으로 적는다.')
    w('- GDINO의 LVIS flip은 정의가 달라 재측정 전까지 비운다. seed별 원자료와 칸별 출처(log, 저장 모델)는 '
      '[results_final.csv](results_final.csv)에 있다.\n')
    w('---\n')

    # 1. 메인 표
    w('## 1. 메인 표: YOLO 계열 6개 모델\n')
    w('| 모델 | 비트 | FP32 | BRECQ | + Ours | Δ AP (vs BRECQ) | 선택 | QDrop | + Ours | Δ AP (vs QDrop) |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    for m in YOLO:
        fp = next((L.best(res, m, s, 'FP32') for s in L.SETTINGS if L.best(res, m, s, 'FP32')), None)
        for s in ('W8A8', 'W4A8', 'W4A5', 'W4A4'):
            b = src.add(1, L.main_cell(res, pending, m, s, 'brecq'))
            o = src.add(1, L.main_cell(res, pending, m, s, 'ours'))
            q = src.add(1, L.main_cell(res, pending, m, s, 'qdrop'))
            qo = src.add(1, L.main_cell(res, pending, m, s, 'qdrop+ours'))
            if not any(x and x.get('lvis') is not None for x in (b, o, q, qo)):
                continue
            w(f"| {L.MODEL_NAME[m]} | {s} | {fpv(fp)} | {ap_flip(b)} | **{ap_flip(o)}** | {delta(o)} | "
              f"{label(m, o and o.get('note'))} | {ap_flip(q)} | **{ap_flip(qo)}** | {delta(qo)} |")
    w('')

    # 2. 표 1 후보
    w('## 2. 표 1 후보: YOLO-World-S, 비교 방법 전체\n')
    bits1 = L.SETTINGS
    w('| 방법 | ' + ' | '.join(bits1) + ' |')
    w('|---|' + '---|' * len(bits1))
    fp = L.best(res, S, 'W8A8', 'FP32')
    w('| FP32 | ' + fpv(fp) + ' |' + ' |' * (len(bits1) - 1))
    rows1 = [('RTN', 'naive'), ('AdaRound', 'adaround(mse)'), ('BRECQ', 'brecq'), ('QDrop', 'qdrop'),
             ('BRECQ + SmoothQuant', 'brecq+S'), ('BRECQ + AWQ', 'brecq+W'), ('**Ours (BRECQ)**', 'ours'),
             ('  └ GP', 'brecq+GP'), ('  └ GPM', 'brecq+GPM'), ('Ours (QDrop)', 'qdrop+ours')]
    for lab, c in rows1:
        cells = []
        for s in bits1:
            x = src.add(2, L.main_cell(res, pending, S, s, c))
            t = ap_flip(x)
            if c in ('ours', 'qdrop+ours') and x and x.get('lvis') is not None:
                t += f" ({label(S, x.get('note'))})"
            cells.append(t)
        w(f'| {lab} | ' + ' | '.join(cells) + ' |')
    w('\n- Δ와 신뢰구간은 §1(BRECQ·QDrop 대비)과 §4(조건별)에 있다.\n')

    # 3. ViT·DETR
    w('## 3. ViT·DETR 계열: OWLv2-B/16, Grounding DINO-T\n')
    w('| 모델 | 비트 | FP32 | BRECQ | QDrop | Ours | 구성 | Δ AP (vs BRECQ) | 평가 |')
    w('|---|---|---|---|---|---|---|---|---|')
    m = 'owlv2-b16'
    fp = next((L.best(res, m, s, 'FP32') for s in L.SETTINGS if L.best(res, m, s, 'FP32')), None)
    for s in ('W8A8', 'W4A8', 'W4A5', 'W4A4'):
        b = src.add(3, L.main_cell(res, pending, m, s, 'brecq'))
        q = src.add(3, L.main_cell(res, pending, m, s, 'qdrop'))
        o = src.add(3, L.main_cell(res, pending, m, s, 'ours'))
        if not any(x and x.get('lvis') is not None for x in (b, q, o)):
            continue
        w(f"| OWLv2-B/16 | {s} | {fpv(fp)} | {ap_flip(b)} | {ap_flip(q)} | **{ap_flip(o)}** | "
          f"{label(m, o and o.get('note'))} | {delta(o)} | LVIS minival |")
    for s in ('W8A8', 'W4A8', 'W4A5', 'W4A4'):
        b, p = gdino(res, s, 'brecq')
        if not b:
            continue
        q, _ = gdino(res, s, 'qdrop')
        o, _ = gdino(res, s, 'brecq+PM')
        fpx = L.best(res, 'gdino-tiny', s, 'FP32', p)
        cb, cq, co = src.add(3, L.cell(b)), src.add(3, L.cell(q)), src.add(3, L.cell(o, b))
        w(f"| Grounding DINO-T | {s} | {fpv(fpx)} | {ap_flip(cb, flip=False)} | {ap_flip(cq, flip=False)} | "
          f"**{ap_flip(co, flip=False)}** | PM | {delta(co)} | {p.split('· ')[-1] if '·' in p else '전체'} |")
    w('\n- OWLv2: W8A8은 M, W4는 PM(고정 구성). GDINO: PM(고정 구성), flip은 재측정 전이라 비운다.\n')

    # 4. ablation
    w('## 4. 표 3 후보: YOLO-World-S ablation (Δ AP vs BRECQ)\n')
    abl = ['brecq+G', 'brecq+P', 'brecq+M', 'brecq+GP', 'brecq+GM', 'brecq+PM', 'brecq+GPM',
           'brecq+R', 'brecq+H', 'brecq+GPS', 'brecq+GPW']
    bits3 = ('W4A8', 'W4A5', 'W4A4')
    w('| 조건 | ' + ' | '.join(f'{s} AP / flip | {s} Δ' for s in bits3) + ' |')
    w('|---|' + '---|---|' * len(bits3))
    for c in ['brecq'] + abl:
        cells = []
        for s in bits3:
            x = src.add(4, L.main_cell(res, pending, S, s, c))
            cells += [ap_flip(x), delta(x) if c != 'brecq' else '']
        w(f"| {c.replace('brecq+', '+') if c != 'brecq' else 'BRECQ'} | " + ' | '.join(cells) + ' |')
    w('\n- R = 같은 예산의 무작위 보호, H = HAWQ식 보호, GPS / GPW = M 대신 SmoothQuant / AWQ식 이전.\n')

    # 5. 나머지 현재 규칙 결과
    shown = {(S, s, c) for s in bits3 for c in abl}
    w('## 5. 그 밖의 현재 규칙 결과 (부록 후보)\n')
    w('| 분류 | 모델 | 프로토콜 | 비트 | 조건 | 꼬리표 | seed | LVIS AP / flip | Δ AP (vs BRECQ) | 출처 |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    extra = [g for g in res.values() if g['m'] in L.MODEL_NAME and g['c'] != 'FP32' and not is_old(g['tags'])
             and g['tbl'] in ('추가지표1', '추가지표2', '추가지표3', '미지정') and (g['m'], g['s'], g['c']) not in shown
             and not smoke(g['p'])]
    for g in sorted(extra, key=lambda g: (g['tbl'], L.MODELS.index((g['m'], L.MODEL_NAME[g['m']])), g['p'], g['s'], g['c'])):
        lv, fl = g['stats'].get('lvis', [None])[0], g['stats'].get('lvis_flip', [None])[0]
        if g['m'] == 'gdino-tiny':
            fl = None
        d = g.get('d', {}).get('lvis')
        dt = '–' if not d else (f"{d['m']:+.2f} ± {d['ci']:.2f}" if d['ci'] is not None else f"{d['m']:+.2f}")
        v = '–' if lv is None else f'{lv:.2f}' + (f' / {fl:.1f}' if fl is not None else '')
        w(f"| {g['tbl']} | {L.MODEL_NAME[g['m']]} | {g['p']} | {g['s']} | {g['c']} | {', '.join(g['tags']) or '–'} | "
          f"{' '.join(map(str, g['seeds']))} | {v} | {dt} | {', '.join(os.path.basename(x) for x in g['src'])} |")
    w('\n- 분류는 리더보드의 표 배정(judge + table_assign.json)을 따른다. 추가지표1 = ablation, 2 = 일반화 보조, 3 = 분석·민감도.\n')

    # 6. 기존 규칙만 남은 칸
    w('## 6. 아직 기존 규칙 결과만 있는 묶음\n')
    left = [g for g in res.values() if g['m'] in L.MODEL_NAME and is_old(g['tags']) and g['tbl'] != '제외' and not smoke(g['p'])]
    n_rep = sum(1 for g in res.values() if g['m'] in L.MODEL_NAME and is_old(g['tags']) and g['tbl'] == '제외')
    w(f'새 규칙 결과로 대체된 기존 규칙 묶음 {n_rep}개는 쓰지 않는다. 아래는 새 규칙 결과가 아직 없어 표에서 빠진 묶음이다.\n')
    if left:
        w('| 분류 | 모델 | 프로토콜 | 비트 | 조건 | 꼬리표 | seed | 출처 |')
        w('|---|---|---|---|---|---|---|---|')
        for g in sorted(left, key=lambda g: (g['tbl'], g['m'], g['s'], g['c'])):
            w(f"| {g['tbl']} | {L.MODEL_NAME[g['m']]} | {g['p']} | {g['s']} | {g['c']} | {', '.join(g['tags'])} | "
              f"{' '.join(map(str, g['seeds']))} | {', '.join(os.path.basename(x) for x in g['src'])} |")
    else:
        w('(없음)')
    w('')

    # 7. 출처
    w('## 7. 절별 출처 (run 디렉토리)\n')
    for sec, name in ((1, '§1 메인 표'), (2, '§2 표 1'), (3, '§3 ViT·DETR'), (4, '§4 ablation')):
        w(f"- {name}: " + ', '.join(f'`{os.path.basename(x)}`' for x in sorted(src.by[sec])))
    w('- 칸별 log와 저장 모델 경로는 csv의 `log`, `save_model` 열에 있다.')
    open(os.path.join(DOC, 'RESULTS_FINAL.md'), 'w').write('\n'.join(out) + '\n')

    # csv: seed별 원자료 (build_results와 같은 규칙으로 같은 칸이 여러 번 돌았으면 가장 늦게 끝난 것)
    latest = {}
    for r in allrecs:
        k = (L.group_key(r), r['seed'])
        if k not in latest or latest[k]['end'] < r['end']:
            latest[k] = r
    picks = {}
    for m in YOLO:
        for s in L.SETTINGS:
            x, note = L.pick_ours(res, m, s)
            if note:
                picks[(m, s)] = 'brecq+' + note.split(':')[-1]
    keys = ['coco', 'lvis', 'apr', 'apc', 'apf', 'lvis_flip', 'lvis_lost', 'tf']
    rows = []
    for (gk, seed), r in latest.items():
        g = res[gk]
        if r['m'] not in L.MODEL_NAME or is_old(r['tags']) or smoke(r['p']) or (g['tbl'] == '제외' and r['c'] != 'FP32'):
            continue
        v = dict(r['v'])
        if r['m'] == 'gdino-tiny':
            v.pop('lvis_flip', None); v.pop('lvis_lost', None)
        ours = r['c'] == picks.get((r['m'], r['s'])) and L.is_main(r['p'])
        rows.append([r['m'], r['p'], r['s'], r['c'], seed, ', '.join(r['tags']), g['tbl'], int(ours)]
                    + [v.get(k) for k in keys] + [r['size'], r['log'], r['save'] or ''])
    rows.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
    with open(os.path.join(DOC, 'results_final.csv'), 'w', newline='') as fh:
        cw = csv.writer(fh)
        cw.writerow(['model', 'protocol', 'bits', 'condition', 'seed', 'tags', 'table', 'is_ours']
                    + keys + ['size_MiB', 'log', 'save_model'])
        cw.writerows(rows)
    print(f'documents/RESULTS_FINAL.md, documents/results_final.csv ({len(rows)}줄)'
          + (f' · 커밋 안 된 코드: {len(dirty)}개' if dirty else ''))


if __name__ == '__main__':
    main()
