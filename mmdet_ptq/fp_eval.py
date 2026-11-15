"""FP 체크포인트 COCO val2017 평가(논문 FP 행과 대조). 사용: fp_eval.py <모델 이름> <work_dir>"""
import sys, json
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import build_runner
name, wd = sys.argv[1], sys.argv[2]
r = build_runner(name, wd)
m = r.test()
print("[fp]", name, json.dumps({k: round(float(v) * 100, 1) for k, v in m.items() if "bbox" in k}))
