"""표 5 — YOLOE-v8s 일반화 (확정 프로토콜, 6 seed). 사용: .venv/bin/python pipeline/scripts/tables/table5_yoloe.py  (어디서 실행해도 된다)
YOLOE에는 텍스트 게이트가 없어 G가 아무것도 하지 않는다: GPM = PM, GP = P. M 사전 검사는 켬(runs/164 check)."""
import glob, os, re, math, statistics as st
from scipy import stats
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logparse import parse, load_s, RUNS
os.chdir(RUNS)

D = {}
for f in glob.glob('164_yoloe/yoloe-W*-s[0-9].log'):
    b = os.path.basename(f)[:-4]; stg = re.search(r'W\dA\d', b).group(0); s = int(b.split('-s')[-1])
    for c, m in parse(f).items(): D.setdefault((stg, c), {})[s] = m
K = [('coco','COCO AP',1),('lvis','LVIS AP',100),('apr','APr',100),('lvis_flip','LVIS_flip (%)',1),('lvis_lost','LVIS_lost',1)]
ROWS = ['naive','brecq','brecq+GPM','brecq+GP']
NAME = {'naive':'naive (RTN)','brecq':'BRECQ','brecq+GPM':'**우리 (GPM = PM)**','brecq+GP':'GP = P (M 없음)'}
def paired(stg, c, k, sc):
    A, B = D[(stg, c)], D[(stg, 'brecq')]; ss = [s for s in range(6) if s in A and s in B]
    x = [A[s][k]*sc - B[s][k]*sc for s in ss]; n = len(x); m = st.mean(x); sd = st.stdev(x); se = sd/math.sqrt(n)
    return m, stats.t.ppf(0.975, n-1)*se, (2*stats.t.sf(abs(m/se), n-1) if sd > 0 else 1.0)
fam = [((stg, c, k), paired(stg, c, k, sc)[2]) for stg in ('W8A8','W4A8','W4A5') for c in ROWS[2:] for k, sc in (('lvis',100),('lvis_flip',1))]
fam.sort(key=lambda z: z[1]); M = len(fam); H = {}; run = 0
for i, (key, p) in enumerate(fam): run = max(run, min(1, (M-i)*p)); H[key] = run
for stg in ('W8A8','W4A8','W4A5'):
    print(f"\n**YOLOE-v8s · {stg}**\n")
    print("| 방법 | seed | " + " | ".join(k[1] for k in K) + " | Δ LVIS AP (p_Holm) | Δ LVIS_flip (p_Holm) |"); print("|" + "---|" * (len(K) + 4))
    for c in ROWS:
        d = D[(stg, c)]; ss = sorted(d)
        cells = [(f"{st.mean([d[s][k]*sc for s in ss]):.0f} ± {st.stdev([d[s][k]*sc for s in ss]):.0f}" if k == 'lvis_lost'
                  else f"{st.mean([d[s][k]*sc for s in ss]):.2f} ± {st.stdev([d[s][k]*sc for s in ss]):.2f}") for k, _, sc in K]
        dd = ' |  |' if c in ('naive','brecq') else ' | ' + ' | '.join(f"{m_:+.2f} ± {ci:.2f} ({H[(stg, c, k)]:.2g})" for (k, sc) in (('lvis',100),('lvis_flip',1)) for m_, ci, _ in [paired(stg, c, k, sc)]) + ' |'
        cocos = [d[s]['coco'] for s in ss]; med = st.median(cocos); col = [s for s in ss if d[s]['coco'] < med - 5]
        print(f"| {NAME[c]}{' ⚠ seed ' + ','.join(map(str,col)) if col else ''} | {len(ss)} | " + " | ".join(cells) + dd)
print(f"\n_Holm family(표 5): {M}개. FP32: COCO 31.29, LVIS 25.94, APr 18.25._")
