"""논문 표 생성기들이 함께 쓰는 로그 읽기 도우미.

run_comparison.py의 로그 끝에 있는 두 결과 표(COCO AP | ..., Heval_flip | ...)를 읽는다.
LVIS AP·APr은 로그에 0~1 값으로 적혀 있다(표 스크립트가 ×100 해서 점수로 쓴다).
표 스크립트는 저장소 루트의 runs/를 기준으로 로그를 찾는다(어디서 실행해도 된다).
"""
import glob, os, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS = os.path.join(ROOT, "runs")


def parse(path):
    """로그 하나 -> {조건: {지표: 값}}. 결과 표가 없는 로그(중단, 진단 전용)는 빈 dict."""
    out = {}
    if not os.path.exists(path):
        return out
    txt = open(path, errors='ignore').read()
    if '이론적 모델 크기' not in txt:
        return out
    for blk, cols in (('COCO AP |', ['coco', 's_ap', 'heval_ap', 'lvis', 'apr', 'apc', 'apf']),
                      ('Heval_flip |', ['heval_flip', 'top1', 'upir', 'lost', 'corr', 'lvis_flip', 'lvis_lost', 'lcorr', 't'])):
        i = txt.index(blk); lines = txt[i:].split('\n')[1:]
        for ln in lines:
            if not ln.strip() or ln.startswith('='):
                break
            parts = [p.strip() for p in ln.split('|')]
            name = parts[0]
            if name == 'FP32':
                continue
            vals = [float(p.rstrip('%')) if p not in ('-', '') else None for p in parts[1:]]
            out.setdefault(name, {}).update(dict(zip(cols, vals)))
    return out


def _g(pattern):
    return glob.glob(os.path.join(RUNS, pattern))


def load_s():
    """YOLO-World-S, 확정 프로토콜: D[(설정, 조건)][seed] = 지표.
    runs/158_final(표 1~3) + runs/154_protocol(W4A8·W4A5 seed 0·1의 naive/brecq/PM/GPM 재사용) + runs/159_gpm_all."""
    D = {}
    for f in _g('158_final/T*-s[0-9].log') + _g('154_protocol/s-W4A*-all-s[0-9].log') + _g('159_gpm_all/W*-s[0-9].log'):
        b = os.path.basename(f)[:-4]; s = int(b.split('-s')[-1])
        st_ = re.search(r'W\dA\d', b).group(0)
        for c, m in parse(f).items():
            D.setdefault((st_, c), {}).setdefault(s, m)
    return D
