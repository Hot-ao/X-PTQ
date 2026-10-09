"""보호 진단 캐시를 미리 만든다. 사용(저장소 루트에서):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 .venv/bin/python pipeline/scripts/make_protect_cache.py yolov8s-world.pt

run_comparison.py는 캐시가 없으면 스스로 만든다. 하지만 같은 모델의 작업 여러 개를 한꺼번에 띄우면 각자 캐시를
만들느라 시간을 버리고 서로 덮어쓴다. 새 모델이나 새 프로토콜로 대기열을 돌리기 전에 이 스크립트를 한 번 실행한다.
캐시 경로와 인자는 run_comparison.py(확정 프로토콜)와 같다: configs/protect_cache/<모델>_n200_c256_img640_la16_aqattn_cls.json
"""
import sys, os, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from diag_w4_sensitivity import rank_convs_leakfree, select_protected

model = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
coco_root = os.environ.get("COCO_ROOT", "/data/taeho/coco_datasets")
out = f"configs/protect_cache/{os.path.splitext(os.path.basename(model))[0]}_n200_c256_img640_la16_aqattn_cls.json"
if os.path.exists(out):
    rows = json.load(open(out)); print("이미 있음:", out)
else:
    rows = rank_convs_leakfree(model, coco_root, dev, n_eval=200, n_calib=256, imgsz=640,
                               first_last_bits=8, last_abits=16, attn_quant="attn_cls")
    json.dump(rows, open(out, "w"), indent=1)
    print("저장:", out, f"(conv {len(rows)}개)")
print("보호 목록(예산 1.5%):", select_protected(rows, 0.015, "coco")[0])
