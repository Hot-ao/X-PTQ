"""
저비트 PTQ 비교 실험의 진입점. FP 모델과 조건별 양자화 모델을 빌드하고 COCO-80과
LVIS-1203(vocabulary 이식)에서 AP와 판정 충실도(flip/lost)를 잰다. 설치·프로토콜·
표 재현은 `README.md`를 볼 것.

조건(`--conditions`)은 `<기반>+<접미사>`로 쓴다.
  - 기반: naive / adaround / qdrop / brecq / combined
  - 접미사: G(텍스트 게이트 교환), P(민감 conv W8 보호), M(공유 제약 scale 이전),
    A(M을 사전 검사로 켤지 결정), R/H(무작위/HAWQ식 보호, 대조군)
  - 우리 방법 = `brecq+GPM`

데이터:
  - calibration: COCO train2017 정렬 순서 앞 `--calib`장(평가 데이터와 완전 분리).
  - COCO-80 평가: COCO val2017 전체(5000장).
  - LVIS 평가: 공식 lvis_v1_minival.json 중 val2017과 겹치는 4809장.

측정 지표:
  - COCO_AP(전체+S/H_eval subset), LVIS_AP(+APr/APc/APf, Fixed AP 프로토콜)
  - Heval_flip(H_eval 열을 가린 top-1 flip) / Top1_flip(표준)
  - GT_MRR/R@1, lost/gained/lateral/corrective_rate, UPIR (COCO·LVIS 양쪽)
  - lost의 S/H_cal/H_eval 그룹별 분해
  - 조건별 빌드 시간, 이론적 모델 크기

GT 앵커 매칭: GT 박스를 letterbox 변환으로 640 공간에 옮긴 뒤, 3개 FPN
level(80x80/40x40/20x20, stride 8/16/32)에서 그리드 셀 중심이 박스 안에 드는
후보를 모으고, 그중 FP32가 정답 class에 가장 confident한 anchor 하나를 GT의
대표 anchor로 선택한다(공식 TaskAlignedAssigner와 완전히 동일하지 않은 근사).

LVIS(P=1203) 쪽은 probe 전체(4809장)의 sim 행렬을 리스트로 들고 있으면 CPU
RAM이 터지므로(fp_sims 하나만 이론상 ~194GB) `compute_lvis_flip_gt_streaming`이
이미지 한 장씩 처리해서 즉시 소비·폐기하는 스트리밍 방식으로 flip/GT 지표를
누적한다. AP 자체는 lvis-api(`LVISEval`)로 별도 채점.

실행(반드시 저장소 루트에서, 확정 프로토콜 인자는 scripts/protocol.sh의 $PROTOCOL):
    source scripts/protocol.sh
    .venv/bin/python run_comparison.py $PROTOCOL --model yolov8s-world.pt \
        --w-bits 4 --a-bits 5 --conditions naive,brecq,brecq+GPM --seed 0 --device 4

스모크 테스트(--eval-cap으로 probe 수 제한, calib도 줄임. COCO AP는 항상 전체 val로 잰다):
    ... $PROTOCOL --calib 8 --eval-cap 16 --conditions naive --seed 0 --device 0
"""
import argparse, glob, os, sys, time

# 09-28: CUDA 장치 인덱스를 nvidia-smi 인덱스와 일치시킨다. torch 기본값
# CUDA_DEVICE_ORDER=FASTEST_FIRST는 이 서버(L40S 3장 + RTX 4000 Ada 5장)에서
# L40S를 먼저 정렬해버려서 --device 0이 실제로는 nvidia-smi GPU 1을,
# --device 1이 GPU 2를 잡는다(실측: 09-28 14:30, runs/116 seed0이 의도한 GPU 0
# 대신 GPU 1에 올라갔다). 공용 서버에서 남의 GPU를 덮칠 수 있는 사고라
# torch를 import하기 전에 PCI_BUS_ID로 고정한다. setdefault라 바깥에서
# 명시적으로 지정한 값은 그대로 존중한다.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import cv2, numpy as np, torch

if not hasattr(np, "float"):
    np.float = float

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))   # 10-27: 보조 모듈(model_zoo 등)은 core/
from harness import SimilarityHarness
from quant.quant_model import wrap_convs, calibrate, set_first_last_bits, set_last_abits, set_conv_wbits
from quant.adaround import convert_to_adaround, optimize_adaround, AdaRoundQuantConv2d, free_cpu_mem
from quant.fake_quant import QuantConv2d
from quant.brecq import optimize_brecq
from quant.promptcal import optimize_promptcal_scale_neighbor


def load_names(which):
    import ultralytics, yaml
    from pathlib import Path
    p = Path(ultralytics.__file__).parent / "cfg" / "datasets" / f"{which}.yaml"
    d = yaml.safe_load(open(p))
    n = d["names"]
    return [n[i] for i in range(len(n))]


def letterbox(im, new=640, color=(114, 114, 114)):
    h, w = im.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im_r = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (new - nh) // 2, (new - nw) // 2
    return cv2.copyMakeBorder(im_r, top, new - nh - top, left, new - nw - left,
                              cv2.BORDER_CONSTANT, value=color)


def preprocess(path, imgsz):
    """CPU 텐서로 반환 -- probe가 5000장 규모라 전부 GPU에 올리면 OOM난다.
    run_image/calibrate/optimize_* 전부 내부에서 자체적으로 .to(device)를 하므로
    여기서 미리 옮길 필요가 없다."""
    im = letterbox(cv2.imread(path), imgsz)
    im = np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))
    t = torch.from_numpy(im).unsqueeze(0).float()
    t.div_(255.0)          # in-place: out-of-place `/255.0`는 이미지당 여분의
    return t                # float32 버퍼를 만들고 버려서, probe 5000장 기준
                            # RSS가 이론치(~24GB)의 2배(~47GB)로 부풀었었다(09-08 발견).


def switch_vocab(model, names, device):
    from model_zoo import switch_vocab_yoloe
    if type(model.model).__name__.startswith("YOLOE"):            # 10-04: YOLOE는 텍스트 임베딩(pe)을 직접 넣는다
        return switch_vocab_yoloe(model, names, device)
    # cache_clip_model=False를 쓰려고 내부 model.model.set_classes를 직접
    # 호출하는데(YOLOWorld.set_classes wrapper는 이 인자를 안 받음), 그러면
    # wrapper가 하는 model.model.names 갱신/predictor 리셋이 같이 스킵된다.
    # 지금은 verbose=False+예측 후처리가 nc=0으로 안전하게 동작해서 수치
    # 결과에는 영향 없지만(09-16 확인), 시각화/verbose를 켜면 이름-인덱스가
    # 어긋나 IndexError가 날 수 있어 직접 맞춰준다(09-16 수정).
    model.model.to("cpu")
    model.model.set_classes(names, cache_clip_model=False)
    model.model.names = list(names)
    model.predictor = None
    model.model.to(device).eval()


def measure_ap(model, data, imgsz, device):
    # workers=0: 기본값(8)이면 DataLoader가 os.fork()로 worker를 여러 개 띄우는데,
    # 이 시점 부모 프로세스가 이미 커져 있으면(q_sims 등) 그 메모리를 통째로
    # 복사해서 순식간에 수백 GB로 터질 수 있다(09-09 실측: RSS 66GB 부모에서
    # worker 8개가 fork되며 시스템 전체가 OOM 직전까지 감). single-process로
    # 강제해서 fork 자체를 없앤다.
    metrics = model.val(data=data, imgsz=imgsz, device=device, save_json=False,
                        verbose=False, workers=0)
    overall = float(metrics.box.map) * 100, float(metrics.box.map50) * 100
    # 09-16 버그 수정: metrics.box.maps는 이미 클래스 id로 직접 인덱싱된
    # nc-길이 배열이라(ultralytics.utils.metrics.Metric.maps 참고) ap_class_index와
    # zip으로 "위치" 짝짓기하면 안 된다 -- ap_class_index가 [0,1,...,nc-1] 풀레인지일
    # 때만 우연히 맞는다. 클래스별로 직접 인덱싱해야 맞다. (또한 구버전 호환용
    # fallback이던 all_ap[:, 0]은 AP50이라 map(AP@0.5:0.95)과 지표 자체가 달라서
    # 같이 제거 -- 현재 ultralytics(8.4.121)는 .maps를 항상 갖고 있어 불필요.)
    per_class = {int(c): float(metrics.box.maps[c]) for c in metrics.box.ap_class_index}
    # 10-19: ultralytics는 COCO면 save_json을 강제로 켜서(models/yolo/detect/val.py) 평가마다 runs/detect/val-N에
    # predictions.json(약 140MB)과 그래프를 남긴다. 공식 COCO AP 계산에 쓰인 뒤로는 읽지 않으므로 여기서 지운다
    # (위 값은 이미 메모리에 있다). 평가 약 4000번 동안 517GB가 쌓였었다.
    sd = getattr(metrics, "save_dir", None)
    if sd is not None and os.path.basename(os.path.dirname(os.path.abspath(sd))) == "detect" \
            and os.path.basename(str(sd)).startswith("val"):
        import shutil
        shutil.rmtree(sd, ignore_errors=True)
    return overall, per_class


def subset_map(per_class, idx, scale=100.0):
    vals = [per_class[i] for i in idx if i in per_class]
    return float(np.mean(vals)) * scale if vals else float("nan")


def group_flip(fp_sims, q_sims, group_idx, conf=0.25):
    Gm = torch.zeros(fp_sims[0].shape[-1], dtype=torch.bool)
    Gm[group_idx] = True
    tot = fl = 0
    for sf, sq in zip(fp_sims, q_sims):
        prob = sf.sigmoid(); mp, c_fp = prob.max(-1); conf_m = mp > conf
        target = conf_m & Gm[c_fp]
        if target.sum() == 0:
            continue
        idx = target.nonzero(as_tuple=True)[0]
        fp_K = sf.clone(); fp_K[:, Gm] = -1e9
        q_K = sq.clone();  q_K[:, Gm] = -1e9
        fl += int((fp_K.argmax(-1)[idx] != q_K.argmax(-1)[idx]).sum())
        tot += len(idx)
    return fl / max(tot, 1) * 100, tot


def standard_flip(fp_sims, q_sims, conf=0.25):
    tot = fl = 0
    for sf, sq in zip(fp_sims, q_sims):
        prob = sf.sigmoid(); mp, c_fp = prob.max(-1); conf_m = mp > conf
        if conf_m.sum() == 0:
            continue
        idx = conf_m.nonzero(as_tuple=True)[0]
        c_q = sq.argmax(-1)
        fl += int((c_fp[idx] != c_q[idx]).sum())
        tot += len(idx)
    return fl / max(tot, 1) * 100, tot


def load_coco_gt_by_path(ann_path, img_paths):
    from pycocotools.coco import COCO
    from ultralytics.data.converter import coco91_to_coco80_class
    coco = COCO(ann_path)
    map91to80 = coco91_to_coco80_class()
    fname_to_id = {info["file_name"]: img_id for img_id, info in coco.imgs.items()}
    out = {}
    for p in img_paths:
        fname = os.path.basename(p)
        img_id = fname_to_id.get(fname)
        boxes = []
        if img_id is not None:
            for a in coco.loadAnns(coco.getAnnIds(imgIds=[img_id], iscrowd=False)):
                cls80 = map91to80[a["category_id"] - 1]
                if cls80 is None:
                    continue
                x, y, w, h = a["bbox"]
                boxes.append((cls80, x + w / 2, y + h / 2, w, h))
        out[p] = boxes
    return out


def load_lvis_gt_by_path(lvis_gt, img_paths, img_ids):
    ann_by_img = {}
    for a in lvis_gt.dataset["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)
    out = {}
    for p, iid in zip(img_paths, img_ids):
        boxes = []
        for a in ann_by_img.get(iid, []):
            cls = a["category_id"] - 1
            x, y, w, h = a["bbox"]
            boxes.append((cls, x + w / 2, y + h / 2, w, h))
        out[p] = boxes
    return out


def get_grid_specs(h_fp, sample_img):
    h_fp.run_image(sample_img, -1)
    specs = []
    for i in sorted(h_fp._level_buf):
        B, P, H, W = h_fp._level_buf[i].shape
        specs.append((H, W))
    return specs


def level_offsets(grid_specs, imgsz):
    """FPN level별 (anchor 시작 offset, H, W, stride)."""
    offsets = []
    off = 0
    for (H, W) in grid_specs:
        stride = imgsz // W
        offsets.append((off, H, W, stride))
        off += H * W
    return offsets


def match_gt_anchors(sf, img_path, boxes, offsets, imgsz):
    """이미지 한 장의 GT 박스마다 대표 anchor를 고른다 -> [(anchor_idx, cls)].
    sf: 그 이미지의 FP sim [anchor, class]. 후보 anchor가 없는 박스는 건너뛴다."""
    im0 = cv2.imread(img_path)
    h0, w0 = im0.shape[:2]
    r = min(imgsz / h0, imgsz / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    img_targets = []
    for (cls, cx, cy, bw, bh) in boxes:
        x0 = (cx - bw / 2) * r + left; x1 = (cx + bw / 2) * r + left
        y0 = (cy - bh / 2) * r + top;  y1 = (cy + bh / 2) * r + top
        candidates = []
        for (off0, H, W, s) in offsets:
            col0 = max(0, int(x0 / s - 0.5)); col1 = min(W - 1, int(x1 / s - 0.5) + 1)
            row0 = max(0, int(y0 / s - 0.5)); row1 = min(H - 1, int(y1 / s - 0.5) + 1)
            for row in range(row0, row1 + 1):
                for col in range(col0, col1 + 1):
                    ccx, ccy = (col + 0.5) * s, (row + 0.5) * s
                    if x0 <= ccx <= x1 and y0 <= ccy <= y1:
                        candidates.append(off0 + row * W + col)
        if not candidates:
            continue
        best = max(candidates, key=lambda a: float(sf[a, cls]))
        img_targets.append((best, cls))
    return img_targets


def build_gt_targets(fp_sims, img_paths, gt_by_path, grid_specs, imgsz):
    offsets = level_offsets(grid_specs, imgsz)
    return [match_gt_anchors(fp_sims[i], p, gt_by_path.get(p, []), offsets, imgsz)
            for i, p in enumerate(img_paths)]


def gt_rank_change(fp_s, q_s, cls):
    """GT-anchor 하나에서 (fp_rank, q_rank, lateral)을 낸다. rank는 정답 class의 순위(1=정답).
    lateral: FP/quant 둘 다 오답인데 quant의 예측 자체가 FP와 다르게 바뀐 경우(오답->다른 오답 flip).
    lost/gained만 보면 "GT-anchor에서 일어난 flip 중 실제로 정답이 되는 비율"의 분모에서 이게 빠진다."""
    fp_rank = int((fp_s > fp_s[cls]).sum()) + 1
    q_rank = int((q_s > q_s[cls]).sum()) + 1
    lateral = fp_rank != 1 and q_rank != 1 and int(fp_s.argmax()) != int(q_s.argmax())
    return fp_rank, q_rank, lateral


def summarize_gt(recip_ranks, lost, gained, lateral):
    """corrective_rate = gained / (lost + gained + lateral)."""
    n = len(recip_ranks)
    total_flips = lost + gained + lateral
    return dict(mrr=sum(recip_ranks) / max(n, 1),
                r1=sum(1 for r in recip_ranks if r == 1.0) / max(n, 1),
                lost=lost, gained=gained, lateral=lateral, total_flips=total_flips,
                corrective_rate=gained / max(total_flips, 1) * 100, n=n)


def gt_metrics_for_method(fp_sims, q_sims, gt_targets, H_eval_set=None, S_set=None, H_cal_set=None):
    recip_ranks = []
    lost = gained = lateral = 0
    upir_num = upir_den = 0
    # S/H_cal/H_eval 그룹별 lost 분해 -- "COCO-80 lost가 H_eval(비보호
    # 클래스)에 몰리는가"라는 가설을 직접 확인하기 위함. denom은 "FP32는
    # 맞혔던(fp_rank==1) GT-anchor 수"(그룹별) -- lost가 나올 수 있는 최대
    # 후보군이라 이걸로 나눠야 그룹 크기(S=40,H_cal=20,H_eval=20) 차이를
    # 보정한 공정한 비율이 나온다.
    lost_by_group = {"S": 0, "H_cal": 0, "H_eval": 0}
    denom_by_group = {"S": 0, "H_cal": 0, "H_eval": 0}
    for i in range(len(fp_sims)):
        sf = fp_sims[i]; sq = q_sims[i]
        for (aidx, cls) in gt_targets[i]:
            q_s = sq[aidx]
            fp_rank, q_rank, is_lateral = gt_rank_change(sf[aidx], q_s, cls)
            recip_ranks.append(1.0 / q_rank)
            group = None
            if S_set is not None and cls in S_set:
                group = "S"
            elif H_cal_set is not None and cls in H_cal_set:
                group = "H_cal"
            elif H_eval_set is not None and cls in H_eval_set:
                group = "H_eval"
            if fp_rank == 1 and q_rank != 1:
                lost += 1
                if group is not None:
                    lost_by_group[group] += 1
            if group is not None and fp_rank == 1:
                denom_by_group[group] += 1
            if fp_rank != 1 and q_rank == 1:
                gained += 1
            if is_lateral:
                lateral += 1
            if H_eval_set is not None and fp_rank == 1 and cls not in H_eval_set:
                upir_den += 1
                if int(q_s.argmax()) in H_eval_set:
                    upir_num += 1
    out = summarize_gt(recip_ranks, lost, gained, lateral)
    if H_eval_set is not None:
        out["upir"] = upir_num / max(upir_den, 1) * 100
        out["upir_n"] = upir_den
    if S_set is not None and H_cal_set is not None:
        out["lost_by_group"] = lost_by_group
        out["denom_by_group"] = denom_by_group
        out["lost_rate_by_group"] = {
            g: lost_by_group[g] / max(denom_by_group[g], 1) * 100 for g in lost_by_group
        }
    return out


def predict_lvis_results(model, img_paths, img_ids, imgsz, device, conf=0.001, max_det=1000):
    # 09-17 버그 수정: DetectionValidator(COCO_AP가 쓰는 model.val() 경로)는
    # NMS를 multi_label=True로 호출하는데, DetectionPredictor(여기서 쓰는
    # model.predict() 경로)는 이 인자를 아예 안 넘겨서 non_max_suppression
    # 기본값 multi_label=False가 조용히 적용되고 있었다. LVIS처럼
    # 1203-class 동의어/federated annotation이 밀집한 vocabulary에서는
    # anchor당 top-1 class만 후보로 내는 multi_label=False가 recall을
    # 심하게 깎는다(실측: FP32 AP 0.126→0.233, APr 0.031→0.158, 4809장
    # 공식 minival 기준). model.predict()는 self.args에서 multi_label을
    # 아예 안 읽어서(DetectionPredictor.postprocess 확인) predict()에
    # multi_label=True를 인자로 넘겨도 무시된다 -- nms 모듈 함수 자체를
    # 임시로 patch해야 실제로 적용된다.
    from ultralytics.utils import nms
    import functools
    orig_nms = nms.non_max_suppression
    nms.non_max_suppression = functools.partial(orig_nms, multi_label=True)
    try:
        results = []
        for path, img_id in zip(img_paths, img_ids):
            # 10-04: agnostic_nms=False를 명시한다. YOLOE.predict()는 기본으로 클래스 무관 NMS를 강제로 켜서
            # (ultralytics models/yolo/model.py) LVIS처럼 비슷한 클래스가 겹치는 어휘에서 recall이 무너진다
            # (YOLOE-v8s FP LVIS AP 11.0). YOLO-World는 원래 기본값이 False라 결과가 같다.
            r = model.predict(source=path, imgsz=imgsz, device=device, conf=conf,
                              max_det=max_det, verbose=False, agnostic_nms=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), sc, c in zip(xyxy, confs, clss):
                results.append({"image_id": int(img_id), "category_id": int(c) + 1,
                                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                "score": float(sc)})
    finally:
        nms.non_max_suppression = orig_nms
    return results


def run_lvis_eval(lvis_gt, results, img_ids):
    # 09-17: Fixed AP 프로토콜 채택(YOLO-World 논문 및 LVIS long-tail
    # 문헌 관행, Dave et al. -- 이미지당 dets 상한(표준 AP의 max_dets=300)이
    # rare class를 구조적으로 불리하게 만든다는 지적). 이미지당 상한 없이,
    # 클래스당 confidence 상위 10000개만 유지해서 채점한다. 위
    # multi_label=True 수정과 합쳐서 실측(FP32, 공식 4809장 minival):
    # AP 0.126→0.259, APr 0.031→0.177 -- 공개 수치(AP 0.243, APr 0.166)와
    # 6% 이내로 일치. **논문에 disclosure 필요**: 이건 버그 수정이 아니라
    # 프로토콜 선택이므로 "Fixed AP를 썼다"고 명시할 것.
    from lvis import LVISEval, LVISResults
    from collections import defaultdict
    if not results:
        return dict(AP=0.0, AP50=0.0)
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category_id"]].append(r)
    fixed = [r for rs in by_cat.values() for r in sorted(rs, key=lambda x: -x["score"])[:10000]]
    lvis_dt = LVISResults(lvis_gt, fixed, max_dets=-1)
    ev = LVISEval(lvis_gt, lvis_dt, iou_type="bbox")
    ev.params.img_ids = img_ids
    ev.run()
    return ev.get_results()


def compute_lvis_flip_gt_streaming(h_fp, h_models, probe_paths, gt_by_path, grid_specs, imgsz, conf=0.25):
    """LVIS(P=1203)는 8400 anchor x 1203 x probe장수 sim을 리스트로 다 들고 있으면
    (4809장 기준 fp_sims 하나만 ~194GB) CPU RAM이 터진다. 이미지 하나씩 처리해서
    FP/각 모델의 sim을 즉시 소비하고 버리는 스트리밍 방식으로 표준 flip과
    GT_MRR/R@1/lost/gained/lateral을 동시에 누적한다. h_models: {mode: SimilarityHarness}."""
    offsets = level_offsets(grid_specs, imgsz)

    tot = {m: 0 for m in h_models}; fl = {m: 0 for m in h_models}
    recip = {m: [] for m in h_models}
    lost = {m: 0 for m in h_models}; gained = {m: 0 for m in h_models}
    lateral = {m: 0 for m in h_models}

    for i, p in enumerate(probe_paths):
        t = preprocess(p, imgsz)
        sf = h_fp.run_image(t, i).sim
        img_targets = match_gt_anchors(sf, p, gt_by_path.get(p, []), offsets, imgsz)

        prob = sf.sigmoid(); mp, c_fp = prob.max(-1); conf_m = mp > conf
        conf_idx = conf_m.nonzero(as_tuple=True)[0]

        for mode, h_q in h_models.items():
            sq = h_q.run_image(t, i).sim
            if len(conf_idx) > 0:
                c_q = sq.argmax(-1)
                fl[mode] += int((c_fp[conf_idx] != c_q[conf_idx]).sum())
                tot[mode] += len(conf_idx)
            for (aidx, cls) in img_targets:
                fp_rank, q_rank, is_lateral = gt_rank_change(sf[aidx], sq[aidx], cls)
                recip[mode].append(1.0 / q_rank)
                if fp_rank == 1 and q_rank != 1:
                    lost[mode] += 1
                if fp_rank != 1 and q_rank == 1:
                    gained[mode] += 1
                if is_lateral:
                    lateral[mode] += 1

        if (i + 1) % 500 == 0:
            print(f"    streaming {i+1}/{len(probe_paths)}장 처리")

    return {mode: dict(top1_flip=fl[mode] / max(tot[mode], 1) * 100, n_flip=tot[mode],
                       **summarize_gt(recip[mode], lost[mode], gained[mode], lateral[mode]))
            for mode in h_models}


def quantized_weight_mib(model_module):
    """09-19 (사용자 지적) 이전엔 total_elements / (1024**2)로 원소당 1바이트
    (=8bit)를 암묵 가정했다 -- --w-bits를 CLI로 조정 가능하게 만든 이상 이제
    실제 버그다(--w-bits 4면 실제 크기는 절반인데 8bit 기준으로 찍힘). 각
    conv가 이미 들고 있는 self.w_bits(QuantConv2d/AdaRoundQuantConv2d 둘 다
    생성자에서 저장)로 비트 수를 직접 계산한다."""
    total_bits = 0
    for m in model_module.modules():
        if isinstance(m, (AdaRoundQuantConv2d, QuantConv2d)):
            if getattr(m, "hi_cols", None) is not None:      # 09-29: 채널 단위 혼합 정밀도
                per_col = m.conv.weight.numel() // m.conv.weight.shape[1]
                n_hi = int(m.hi_cols.sum())
                total_bits += per_col * (n_hi * m.hi_bits + (m.conv.weight.shape[1] - n_hi) * m.w_bits)
            else:
                total_bits += m.conv.weight.numel() * m.w_bits
        elif type(m).__name__ == "QuantLinear":
            # 10-03: --attn-quant로 8bit 양자화되는 attention Linear(실행 중 입력을 받는 것)도 모델 weight다.
            # 텍스트 상수(ConstQ: guide projection 결과, contrastive 텍스트)는 어휘마다 달라지는 값이라 제외한다.
            total_bits += m.lin.weight.numel() * m.bits
    return total_bits / 8 / (1024 * 1024)


MCHECK_RATIO = 1.5   # 10-02 +A: M을 켠 쪽 calib-밖 COCO flip이 이 배수를 넘으면 M을 끈다
PROTECT_BUDGET = 0.015                   # +P/+R/+H: W8로 보호할 양자화 weight 파라미터 비율
PROTECT_IMAGES = 200                     # 보호 진단 이미지 수(train2017 앞쪽)
PROTECT_CACHE = "configs/protect_cache"  # 진단 결과 캐시(모델·프로토콜별, seed·조건과 무관하게 한 번만 계산)
MCHECK_CACHE = "configs/mcheck_cache"    # 10-28 +A 결정 캐시(모델·비트·프로토콜별 한 번). 있으면 두 번 빌드하지 않고 그 결정을 쓴다


def mcheck_cache_path(args):
    """M 사전 검사 결정 파일. 모델·비트 설정마다 한 번 정한다(METHOD §4.5). 10-28 처음에는 비트를 빼고 모델당 한 번으로
    했으나, YOLO-World-M에서 W4A5 검사는 "끔", W4A4 검사는 "켬"으로 갈려(runs/205) 비트를 키에 넣었다. seed와 기반 PTQ는
    넣지 않는다. 빌드 결과를 바꾸는 프로토콜 인자(calibration 수, 입력 크기, 첫/마지막 비트, head 입력 비트, attention 양자화)는 넣는다."""
    stem = os.path.splitext(os.path.basename(args.model))[0]
    return os.path.join(MCHECK_CACHE, f"{stem}_w{args.w_bits}a{args.a_bits}_c{args.calib}_img{args.imgsz}_fl{args.first_last_bits}"
                                      f"_la{args.last_abits}_aq{args.attn_quant}.json")
MIG_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)  # +M: scale 이전의 a 후보('이전 없음'도 항상 후보)
MIG_IMAGES = 8                           # +M: a 탐색에 쓸 calibration 이미지 수
RECON_ITERS_ADA = 1000                   # adaround 재구성 iters
RECON_ITERS = 2000                       # brecq/qdrop 재구성 iters
QDROP_PROB = 0.5
COMBINED_ITERS = 1500                    # combined 2단계(s_mult) iters


def build(model_cls, w, names, device, calib, mode, fp=None, pidx=None, h_eval=None, cal_idx=None,
          w_bits=8, a_bits=8, skip_head=True, first_last_bits=0, last_abits=0, attn_quant="none",
          hi_wbit_convs=(), migrate=False, gate_commute=False,
          adaround_act_observer="minmax", combined_stage1="none", combined_recon_iters=0):
    """mode: "fp" | "naive" | "adaround" | "qdrop" | "pdquant" | "brecq" | "combined".
    fp 외에는 fp(FP 모델)가 필요하고, combined는 pidx(S)/h_eval/cal_idx(H_cal)도 필요하다.
    first_last_bits/last_abits/attn_quant/hi_wbit_convs/migrate는 양자화기 설정이라 calibrate 전에,
    조건과 무관하게 같은 방식으로 적용된다."""
    m = model_cls(w)
    m.set_classes(names)
    if mode == "fp":
        m.fuse(); m.model.to(device).eval()
        return m
    m.fuse()
    if gate_commute:
        # 09-30 설계 방향 1: C2fAttn의 텍스트 게이트를 cv2 뒤로 옮기는 FP 등가 재배치(quant/fusion_quant.py).
        # 양자화 전(wrap_convs 전)에 구조를 바꾼다. 재구성의 FP 기준도 같은 구조여야 하므로 아래 _fpref 참고.
        assert mode in ("naive", "brecq", "qdrop", "pdquant"), f"게이트 교환 미지원 조건: {mode}"
        from quant.fusion_quant import gate_commute_all
        gate_commute_all(m.model)
    # 09-23: CLIP 인코더(clip_model) 제외는 quant_model.ALWAYS_SKIP_NAMES가 담당한다
    # -- set_classes()가 캐싱하는 CLIP vision tower의 patch-embed conv가 양자화 대상에
    # 섞여 모델 크기를 22.8% 과대계상하던 버그(경위는 wrap_convs 주석 참고).
    wrap_convs(m.model, w_bits, a_bits,
               skip_modules=[m.model.model[-1]] if skip_head else None)
    if first_last_bits > 0:
        # 저비트 표준 프로토콜 -- stem과 (head 양자화 시) cv2/cv3 마지막 conv를 고정 비트로.
        set_first_last_bits(m.model, first_last_bits)
    if last_abits > 0:
        # head cv2/cv3 마지막 1x1 입력 activation만 last_abits(16)로.
        set_last_abits(m.model, last_abits)
    if hi_wbit_convs:
        # +P/+R/+H: conv 단위 혼합 정밀도(고른 conv의 weight만 8bit).
        set_conv_wbits(m.model, hi_wbit_convs, 8)
    if attn_quant != "none":
        # attention(Linear/matmul)과 (attn_cls면) head contrastive matmul도 8bit로(quant/attn_quant.py).
        # gate_commute 뒤여야 한다.
        from quant.attn_quant import quantize_attention
        quantize_attention(m.model, attn_quant, 8)
    m.model.to(device).eval()
    if migrate:
        # +M: 입력 채널별 scale 이전. 배포 제약(같은 생산 채널을 받는 소비 conv는 같은 s, residual add로
        # 합쳐지는 생산 채널도 같은 s)을 지키는 버전이다(quant/channel_graph.py). observer/weight scale이
        # 이전된 값 기준으로 잡혀야 하므로 calibrate 전. layer-wise AdaRound(optimize_adaround)는
        # conv.weight를 FP 입력과 직접 곱해 target을 만들어서 이전과 호환되지 않는다 -- combined는
        # 1단계가 brecq일 때만 허용(2단계의 s_mult는 conv당 스칼라라 이전과 독립).
        assert mode in ("naive", "brecq", "qdrop", "pdquant") or \
            (mode == "combined" and combined_stage1 == "brecq"), f"migration 미지원 조건: {mode}"
        # migrate: True/"M" = 우리 M, "S" = SmoothQuant, "W" = AWQ(10-17 기준선, 같은 공유 제약·같은 이미지).
        from quant.channel_graph import search_and_apply_tied
        search_and_apply_tied(m.model, calib[:MIG_IMAGES], device, MIG_ALPHAS,
                              rule={"S": "smooth", "W": "awq"}.get(migrate, "tied"))
    _fpref = fp
    if gate_commute and mode in ("brecq", "qdrop", "pdquant"):
        # BRECQ가 quant/FP conv를 트리 순서로 짝짓고 FP 출력을 목표로 쓰므로, FP 기준 모델도 같은
        # 게이트 교환 구조여야 cv2_main/cv2_side가 올바른 짝을 갖는다(FP 등가).
        from quant.fusion_quant import gate_commute_all as _gca
        _fpref = model_cls(w); _fpref.set_classes(names); _fpref.fuse(); _gca(_fpref.model)
        _fpref.model.to(device).eval()
    # activation scale 초기화는 공식 BRECQ/QDrop 방식(MSE, L_2.4 grid search).
    calibrate(m.model, calib, device=device, act_observer="mse")
    if mode == "naive":
        pass
    elif mode == "adaround":
        # AdaRound 원 논문(Nagel et al. ICML'20)에 충실한 구현: 순수 weight-rounding이라 activation scale은
        # 학습하지 않고, activation calibration은 논문이 명시한 min-max로 재보정한다.
        # adaround_act_observer="mse"면 재보정을 생략해 다른 방법과 같은 관측기를 공유한다
        # (교란변수 격리용 대조, 표 1의 adaround(mse)).
        if adaround_act_observer != "mse":
            calibrate(m.model, calib, device=device, act_observer=adaround_act_observer)
        convert_to_adaround(m.model, channelwise_smult=True, w_quant_mode="adaround")
        optimize_adaround(m.model, fp.model, calib, device, iters=RECON_ITERS_ADA,
                          verbose=False, learn_act_scale=False)
    elif mode == "qdrop":
        # QDrop(Wei et al., ICLR 2022)은 BRECQ 프레임워크를 물려받아 activation scale 학습(LSQ)을 포함한다.
        # brecq와 재구성 단위/iters를 맞추고 drop 유무만 단일 변수로 비교한다. 공식 COCO 설정(batch 2,
        # neck layer-wise)과 달리 batch 1, 전부 block-wise다(docs/BASELINE_STATUS.md).
        convert_to_adaround(m.model, channelwise_smult=False)
        optimize_brecq(m.model, _fpref.model, calib, device, iters=RECON_ITERS,
                       qdrop_prob=QDROP_PROB, verbose=False, learn_act_scale=True,
                       batch=1, neck_layerwise=False)
    elif mode == "pdquant":
        # 10-28 PD-Quant(Liu et al., CVPR 2023) 기준선. 공식 구현이 QDrop 코드 위에 만들어졌으므로 위 qdrop과 같은 설정
        # (재구성 단위, iters, batch, drop 확률, LSQ)에 PD 손실과 DC만 더한다(quant/pdquant.py). BN 통계는 접기 전 모델에서 모은다.
        from quant.pdquant import collect_bn_targets, PDContext
        _unf = model_cls(w); _unf.set_classes(names)
        _tg = collect_bn_targets(_unf.model)
        del _unf
        _named = dict(_fpref.model.named_modules())
        _bn_map = {_named[k]: v for k, v in _tg.items() if isinstance(_named.get(k), torch.nn.Conv2d)}
        print(f"  [pdquant] DC 대상 conv {len(_bn_map)}/{len(_tg)}개 (BN 통계), PD 손실 T={__import__('quant.pdquant', fromlist=['PD_T']).PD_T}", flush=True)
        convert_to_adaround(m.model, channelwise_smult=False)
        _pd = PDContext(_fpref.model, list(calib), device)
        try:
            optimize_brecq(m.model, _fpref.model, calib, device, iters=RECON_ITERS,
                           qdrop_prob=QDROP_PROB, verbose=False, learn_act_scale=True,
                           batch=1, neck_layerwise=False, pd=_pd, dc_bn_map=_bn_map)
        finally:
            _pd.close()
    elif mode == "brecq":
        # weight rounding과 activation scale을 공동 최적화한다(공식의 순차 2단계가 아님, batch 1, 전부 block-wise).
        convert_to_adaround(m.model, channelwise_smult=False)
        optimize_brecq(m.model, _fpref.model, calib, device, iters=RECON_ITERS,
                       verbose=False, learn_act_scale=True, two_stage=False,
                       batch=1, neck_layerwise=False)
    elif mode == "combined":
        # Combined(PromptCal) 기준선: 1단계(weight 재구성, combined_stage1) 뒤에 2단계로 per-tensor
        # s_mult(activation scale)를 margin/neighbor 목적함수로 학습한다. S(pidx)와 H_cal(cal_idx)에
        # margin_loss를 걸고, H_eval은 neighbor 후보 풀에서 제외한다(held-out).
        convert_to_adaround(m.model, channelwise_smult=False)
        if combined_stage1 != "none":
            assert combined_recon_iters > 0, (
                f"--combined-stage1={combined_stage1}인데 --combined-recon-iters="
                f"{combined_recon_iters}입니다 -- 0이면 1단계가 조용히 생략되고 "
                f"'none'과 동일한 결과가 나옵니다. iters를 명시하세요"
                f"(예: --combined-recon-iters 2000).")
        if combined_stage1 == "adaround":
            optimize_adaround(m.model, fp.model, calib, device, iters=combined_recon_iters,
                              verbose=False)
        elif combined_stage1 == "brecq":
            # alpha만 재구성한다(activation scale은 2단계가 전담). 주의: 여기는 neck_layerwise를 넘기지
            # 않아 optimize_brecq의 기본값(neck은 conv 단위)을 받는다 -- 위 brecq 조건(전부 block-wise)과
            # 재구성 단위가 다르다. 표 1의 combined가 이 설정으로 측정됐으므로 바꾸지 말 것.
            optimize_brecq(m.model, fp.model, calib, device, iters=combined_recon_iters,
                           verbose=False)
        optimize_promptcal_scale_neighbor(m.model, fp.model, calib, device, pidx, iters=COMBINED_ITERS,
                                          lr=1e-2, k=5, neighbor_k=5, neighbor_weight=1.0,
                                          asymmetric=True, scale_reg_weight=1.0,
                                          exclude_from_neighbors=h_eval,
                                          cal_idx=cal_idx, cal_weight=1.0,
                                          identity_aware_margin=True, verbose=False)
    else:
        raise ValueError(f"알 수 없는 조건: {mode}")
    return m


def qmodel_path(d, args, cond):
    """--save-model/--load-model 파일 이름: <모델>_w<W>a<A>_s<seed>_<조건>.pt (조건의 '+'는 '-'로)."""
    stem = os.path.splitext(os.path.basename(args.model))[0]
    return os.path.join(d, f"{stem}_w{args.w_bits}a{args.a_bits}_s{args.seed}_{cond.replace('+', '-')}.pt")


def save_qmodel(wrapper, path, meta):
    """10-27: 빌드한 양자화 모델(ultralytics wrapper 안의 nn.Module)을 통째로 저장한다. 커스텀 모듈(QuantConv2d,
    AdaRound, 게이트 교환 구조, 이전 scale mig)이 그대로 들어가므로 코드가 바뀌면 못 읽을 수 있다 -- 그래서 커밋과
    인자를 옆의 .json에 남긴다. CLIP 텍스트 인코더(clip_model)는 원 가중치 그대로라 빼고 저장한다."""
    import json, subprocess
    os.makedirs(os.path.dirname(path), exist_ok=True)
    core = wrapper.model
    clip = getattr(core, "clip_model", None)
    if clip is not None:
        core.clip_model = None
    try:
        from qmodel_io import save_module
        save_module(core, path + ".tmp")
        os.replace(path + ".tmp", path)
    finally:
        if clip is not None:
            core.clip_model = clip
    root = os.path.dirname(os.path.abspath(__file__))
    git = lambda *a: subprocess.run(["git", "-C", root, *a], capture_output=True, text=True).stdout.strip()
    meta = dict(meta, commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
                saved=time.strftime("%Y-%m-%d %H:%M:%S"), torch=torch.__version__)
    json.dump(meta, open(path[:-3] + ".json", "w"), indent=1, ensure_ascii=False)
    print(f"  [save] {path} ({os.path.getsize(path) / 2**20:.1f} MiB)", flush=True)


def update_qmodel_meta(path, **kw):
    import json
    j = path[:-3] + ".json"
    if os.path.exists(j):
        meta = json.load(open(j)); meta.update(kw)
        json.dump(meta, open(j, "w"), indent=1, ensure_ascii=False)


def load_qmodel(model_cls, w, path, device):
    """save_qmodel로 저장한 모델을 ultralytics wrapper에 다시 넣는다. 어휘는 평가 전에 switch_vocab으로 바꾼다."""
    import json
    m = model_cls(w)
    from qmodel_io import load_module
    m.model = load_module(path, map_location=device)    # 일반 텐서 속성(ConstQ 캐시, scale 등)은 .to()로 안 옮겨진다
    m.model.to(device).eval()
    m.predictor = None
    j = path[:-3] + ".json"
    meta = json.load(open(j)) if os.path.exists(j) else {}
    print(f"  [load] {path} (commit {meta.get('commit', '?')[:8]}{', dirty' if meta.get('dirty') else ''}, "
          f"저장 {meta.get('saved', '?')}, 저장 당시 COCO AP {meta.get('coco_ap', '?')})", flush=True)
    return m, meta


def eval_odinw_yolo(fp, models, conditions, args, device):
    """ODinW-13(GLIP 공식 split, core/odinw.py): 데이터셋마다 어휘를 바꿔 COCO AP. 반환 {모델: {데이터셋: AP, mean}}."""
    from odinw import load_odinw13, coco_ap as odinw_ap, report as odinw_report
    odinw_sets = load_odinw13(args.odinw_root, args.eval_cap)
    odinw_res = {k: {} for k in ["FP32"] + conditions}
    for ds in odinw_sets:
        for k in odinw_res:
            m = fp if k == "FP32" else models[k]
            switch_vocab(m, ds["names"], device)
            preds = predict_lvis_results(m, ds["img_paths"], ds["img_ids"], args.imgsz, device, max_det=300)
            for r in preds:                                   # predict_lvis_results는 클래스 순번+1을 넣는다
                r["category_id"] = ds["cat_ids"][r["category_id"] - 1]
            odinw_res[k][ds["name"]] = 100 * odinw_ap(ds["gt"], preds, ds["img_ids"])
        print(f"  [ODinW] {ds['name']} ({len(ds['img_ids'])}장) " +
              " ".join(f"{k}={odinw_res[k][ds['name']]:.2f}" for k in odinw_res), flush=True)
    return odinw_report(odinw_res, odinw_sets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-world.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--data", default="configs/coco_local.yaml")
    ap.add_argument("--gt-ann", default=None)
    ap.add_argument("--lvis-ann",
                    default="/data/taeho/lvis_datasets/labels_dl/extracted/lvis/annotations/lvis_v1_minival.json")
    ap.add_argument("--conditions", default="naive,adaround,qdrop,brecq,combined",
                    help="쉼표로 구분된 조건 목록(콤마 뒤 공백 없이). <기반>+<접미사> 형식, 예: naive,brecq,brecq+GPM")
    ap.add_argument("--w-bits", type=int, default=8)
    ap.add_argument("--a-bits", type=int, default=8)
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--skip-head", action=argparse.BooleanOptionalAction, default=True,
                    help="detection head를 양자화하지 않음(QDrop COCO 프로토콜). 확정 프로토콜은 --no-skip-head")
    ap.add_argument("--first-last-bits", type=int, default=0,
                    help=">0이면 stem 첫 conv와 (head 양자화 시) cv2/cv3 마지막 1x1 conv를 이 비트로 고정. "
                         "모든 조건에 동일 적용. 기본 0=꺼짐")
    ap.add_argument("--last-abits", type=int, default=0,
                    help=">0이면 head cv2/cv3 마지막 1x1 conv의 입력 activation만 이 비트로(16 권장). "
                         "보호 진단에도 같은 값이 쓰이고 진단 캐시 이름에 _la<비트>가 붙는다. 기본 0=꺼짐")
    ap.add_argument("--attn-quant", choices=["none", "attn", "attn_cls"], default="none",
                    help="attention Linear/matmul(attn), + head contrastive matmul(attn_cls)을 8bit로 양자화")
    ap.add_argument("--adaround-act-observer", choices=["minmax", "mse"], default="minmax",
                    help="AdaRound 모드 전용 activation observer. 기본 minmax(원 논문). mse면 다른 방법과 같은 관측기")
    ap.add_argument("--combined-stage1", choices=["none", "adaround", "brecq"], default="none",
                    help="Combined의 1단계(weight 재구성)로 뭘 쓸지. none = round-to-nearest. "
                         "none이 아니면 --combined-recon-iters를 같이 줘야 한다")
    ap.add_argument("--combined-recon-iters", type=int, default=0,
                    help="Combined 1단계의 iters(--combined-stage1이 none이 아닐 때)")
    ap.add_argument("--deterministic", action="store_true",
                    help="torch.use_deterministic_algorithms(warn_only=True) 활성화. 같은 seed 재실행에서 "
                         "Combined까지 bit 단위로 재현되게 한다")
    ap.add_argument("--eval-cap", type=int, default=0,
                    help="스모크 테스트용: flip/GT/UPIR/lost와 LVIS AP를 계산하는 probe 이미지 수 제한. 0이면 제한 없음. "
                         "COCO AP(measure_ap)에는 적용되지 않는다 -- 항상 --data의 전체 val 세트로 측정")
    ap.add_argument("--odinw", action="store_true",
                    help="LVIS 뒤에 ODinW-13(GLIP 공식 split)도 평가한다(core/odinw.py). --eval-cap은 데이터셋마다 적용")
    ap.add_argument("--odinw-root", default="/data/taeho/odinw")
    ap.add_argument("--odinw-only", action="store_true",
                    help="COCO/LVIS flip·AP를 건너뛰고 ODinW-13만 평가한다(--odinw 포함). 조건 모델마다 COCO AP는 "
                         "같은 모델인지 확인하려고 잰다(--no-coco-check로 끔)")
    ap.add_argument("--no-coco-check", action="store_true", help="--odinw-only에서 COCO AP 확인을 건너뛴다")
    ap.add_argument("--save-model", default="",
                    help="빌드한 조건 모델을 이 폴더에 저장(qmodel_path 이름 + 메타 .json). 예: /data/taeho/cvpr2027_qmodels")
    ap.add_argument("--mcheck-refresh", action="store_true",
                    help="+A: 저장된 M 결정(configs/mcheck_cache)이 있어도 두 후보를 다시 빌드해 검사하고 결정을 덮어쓴다")
    ap.add_argument("--load-model", default="",
                    help="빌드 대신 이 폴더에서 같은 이름의 저장 모델을 불러온다(없으면 중단)")
    ap.add_argument("--post-build", default="",
                    help="진단 스크립트 경로. 빌드 직후(평가 전) 실행하고 평가 없이 종료(PTQ_POST_BUILD와 같음)")
    args = ap.parse_args()
    if args.odinw_only:
        args.odinw = True
    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    gt_ann = args.gt_ann or os.path.join(args.coco_root, "annotations", "instances_val2017.json")
    print(f"[args] {vars(args)}")
    # 09-28: --device N이 실제로 어느 물리 GPU에 올라갔는지 로그에 남긴다.
    # CUDA_DEVICE_ORDER를 PCI_BUS_ID로 고정했으므로 여기 찍히는 인덱스는
    # nvidia-smi 인덱스와 같아야 한다 -- 다르면 환경이 바뀐 것이니 멈추고 확인할 것.
    if device != "cpu":
        _p = torch.cuda.get_device_properties(int(args.device))
        print(f"[gpu] --device {args.device} -> nvidia-smi GPU {args.device} "
              f"({_p.name}, {_p.total_memory // 2**20}MiB, uuid={_p.uuid}) "
              f"CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER')}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.deterministic:
        # 09-23: 같은 seed로 두 번 돌렸을 때 naive/AdaRound/QDrop/BRECQ는 모든 지표가
        # bit-identical인데 Combined만 재현되지 않는다(실측: COCO_AP 36.61 vs 36.64,
        # Heval_flip 4.79% vs 5.26%, lost 226 vs 239 -- runs/91 seed0 vs runs/92_env_recheck).
        # cudnn.deterministic은 conv 알고리즘만 고정할 뿐 index/scatter backward의
        # atomicAdd 계열 비결정성은 못 잡는다. 이 플래그를 켜면 결정적 구현이 있는
        # 연산은 그걸 쓰고, 없는 연산은 경고를 띄워 "어디가 비결정적인지"를 특정할 수
        # 있다. 기본 False -- 켜면 일부 연산 구현이 바뀌어 기존 확정 수치와 달라질 수
        # 있으므로 진단/재현성 확보용으로만 쓸 것.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)

    if args.odinw_only:
        lvis_gt, lvis_img_ids = None, set()
    else:
        from lvis import LVIS
        print(f"[lvis] loading GT {args.lvis_ann}")
        lvis_gt = LVIS(args.lvis_ann)
        lvis_img_ids = set(lvis_gt.get_img_ids())

    coco = load_names("coco")
    lvis_names = load_names("lvis")
    from model_zoo import model_factory
    YOLOWorld = model_factory(args.model)        # 10-04: 이름은 그대로 두고 모델 종류(YOLO-World / YOLOE)에 맞는 생성자를 쓴다

    # calib: train2017 (평가 데이터와 완전 분리)
    calib_paths = sorted(glob.glob(os.path.join(args.coco_root, "train2017", "*.jpg")))[:args.calib]
    print(f"[data] calib {len(calib_paths)}장 (train2017)")

    # probe: val2017 전체 -- COCO-80은 5000장 다 쓰고, LVIS는 그중 minival(공식 4809장)과
    # 겹치는 것만 채점
    probe_paths = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))
    if args.eval_cap > 0:
        probe_paths = probe_paths[:args.eval_cap]
    lvis_probe_paths, lvis_probe_ids = [], []
    for p in probe_paths:
        iid = int(os.path.basename(p).split(".")[0])
        if iid in lvis_img_ids:
            lvis_probe_paths.append(p); lvis_probe_ids.append(iid)
    print(f"[data] probe {len(probe_paths)}장(val2017 전체), 그중 LVIS minival과 겹치는 "
          f"{len(lvis_probe_paths)}장으로 LVIS 채점")

    calib = [preprocess(p, args.imgsz) for p in calib_paths]
    # probe(5000장)는 CPU에 미리 다 올려두지 않는다 -- 640x640x3 float32 기준
    # 장당 ~4.9MB, 5000장이면 ~24.6GB인데 이게 프로세스당 고정 비용으로 깔려서
    # 동시 실행 가능한 seed 수를 제한하는 주범이었다. 필요할 때마다
    # preprocess()로 다시 읽는다 -- 디스크 read 속도가 빠르므로 전체 실험
    # 시간(수 시간)에 비해 오버헤드가 무시할 수준.

    if not args.odinw_only:
        coco_gt_by_path = load_coco_gt_by_path(gt_ann, probe_paths)
        lvis_gt_by_path = load_lvis_gt_by_path(lvis_gt, lvis_probe_paths, lvis_probe_ids)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(80)
    S = perm[:40].tolist(); H_cal = perm[40:60].tolist(); H_eval = perm[60:80].tolist()
    H_eval_set = set(H_eval)
    S_set = set(S); H_cal_set = set(H_cal)

    print("[build] FP (COCO-80)")
    fp = build(YOLOWorld, args.model, coco, device, calib, "fp")

    conditions = [c.strip() for c in args.conditions.split(",")]

    _valid = {"naive", "adaround", "qdrop", "pdquant", "brecq", "combined"}
    # 09-29: 조건 접미사 -- "brecq+PM"처럼 기반 조건 뒤에 양자화기 옵션을 조건별로 붙인다(한 run에서 짝비교).
    #   P = 보호(누수 없는 진단, COCO 어휘 top-1 flip 기준, 예산 PROTECT_BUDGET)
    #   R = 같은 예산 무작위 보호(random.Random(1000), runs/134 rand0과 같은 목록)
    #   H = 같은 예산 HAWQ식 보호(출력 MSE = 'cls' 기준)
    #   M = 공유 제약 scale 이전(a 후보 0,0.25,0.5,0.75,1)
    #   G = 텍스트 게이트 교환(09-30 설계 방향 1). G와 P/R/H를 함께 쓰면 C2fAttn cv2(교환으로 사라지는 conv)는
    #       보호 목록에서 빠진다(교환이 그 conv를 대신 다룸 -> 모델이 그만큼 작아짐).
    #   A = M을 사전 검사로 켤지 정한다(아래 빌드 루프 참고).
    #   S = SmoothQuant(a=0.5 고정), W = AWQ(mean|x|^a, weight 오차로 a 탐색) -- M 자리에 넣는 기준선(10-17).
    #       M과 같은 공유 제약·같은 이미지로 적용한다. M/S/W 중 하나만 쓴다.
    _flag_ok = set("PRHMGASW")

    def _split_cond(c):
        base, _, suf = c.partition("+")
        return base, set(suf)

    assert all(_split_cond(c)[0] in _valid and _split_cond(c)[1] <= _flag_ok for c in conditions), \
        f"--conditions에 알 수 없는 값: {[c for c in conditions if _split_cond(c)[0] not in _valid or not _split_cond(c)[1] <= _flag_ok]}"
    assert all(len(_split_cond(c)[1] & set("MSW")) <= 1 and not ({"A"} <= _split_cond(c)[1] and _split_cond(c)[1] & set("SW"))
               for c in conditions), "M/S/W는 한 조건에 하나만, A는 M하고만 쓴다"
    assert len(set(conditions)) == len(conditions), "--conditions에 중복이 있음"
    _all_flags = set().union(*(_split_cond(c)[1] for c in conditions))
    cond_lists = {"P": (), "R": (), "H": ()}
    if (_all_flags & {"P", "R", "H"}) and args.w_bits < 8:
        import json as _json, random as _random
        from diag_w4_sensitivity import rank_convs_leakfree, select_protected
        _budget = PROTECT_BUDGET
        os.makedirs(PROTECT_CACHE, exist_ok=True)
        _cache = os.path.join(PROTECT_CACHE, f"{os.path.splitext(os.path.basename(args.model))[0]}"
                                             f"_n{PROTECT_IMAGES}_c{args.calib}_img{args.imgsz}"
                                                 f"{f'_la{args.last_abits}' if args.last_abits else ''}"
                                                 f"{f'_aq{args.attn_quant}' if args.attn_quant != 'none' else ''}.json")
        if os.path.exists(_cache):
            _rows = _json.load(open(_cache))
        else:
            print(f"[protect] 누수 없는 conv 단위 진단 실행 -> {_cache}")
            _rows = rank_convs_leakfree(args.model, args.coco_root, device, n_eval=PROTECT_IMAGES,
                                        n_calib=args.calib, imgsz=args.imgsz, first_last_bits=args.first_last_bits or 8,
                                        last_abits=args.last_abits, attn_quant=args.attn_quant)
            _json.dump(_rows, open(_cache, "w"), indent=1)
        cond_lists["P"] = tuple(select_protected(_rows, _budget, "coco")[0])
        cond_lists["H"] = tuple(select_protected(_rows, _budget, "cls")[0])
        # R: runs/134 random_sets.json(rand0)과 같은 절차 -- 진단 행 순서(groups_of 순서)로 섞고 예산 ±10%.
        _convs = [(r["name"], r["mparam"]) for r in _rows]
        _target = _budget * sum(p for _, p in _convs)
        _rng = _random.Random(1000)
        while True:
            _order = _convs[:]; _rng.shuffle(_order); _sel = []; _tot = 0.0
            for n_, p_ in _order:
                if _tot + p_ <= _target * 1.1:
                    _sel.append(n_); _tot += p_
                if _tot >= _target * 0.9:
                    break
            if _target * 0.9 <= _tot <= _target * 1.1:
                break
        cond_lists["R"] = tuple(_sel)
        for k_, v_ in cond_lists.items():
            if k_ in _all_flags:
                print(f"[cond+{k_}] 예산 {_budget:.1%} -> {list(v_)}")
    elif _all_flags & {"P", "R", "H"}:
        print(f"[cond] w-bits={args.w_bits} >= 8 -> +P/+R/+H는 규칙상 no-op")
    models, calib_time = {}, {}
    # 09-29: 조건마다 빌드 직전에 RNG 상태를 루프 진입 시점으로 되돌린다. 이전에는 앞 조건이 소비한 난수
    # (MSE observer 표본 추출, BRECQ 배치 샘플링, QDrop 마스크)만큼 뒤 조건의 RNG 위치가 밀려서 조건 순서가
    # 결과를 바꿨다(§5.2). 복원하면 모든 조건이 "첫 번째 조건"과 같은 상태에서 시작한다 -- 첫 조건은 이전과 동일.
    import random as _py_random
    import numpy as _np
    _rng_snapshot = (torch.get_rng_state(),
                     torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                     _np.random.get_state(), _py_random.getstate())
    model_mib_by = {}
    for cond in conditions:
        mode, _flags0 = _split_cond(cond)
        # 10-02 +A: 이전(M)을 누수 없는 사전 검사로 켤지 정한다. M 끔/켬 두 후보를 모두 빌드(각각 같은 RNG 상태에서
        # 출발 -- 단독 조건과 bit-identical)하고, calibration에 쓰지 않은 train2017 이미지 + COCO 어휘에서 FP 대비
        # top-1 flip을 잰다. M을 켠 쪽 flip이 MCHECK_RATIO배를 넘으면 M을 끈 모델을 쓴다(runs/155 검증).
        if args.load_model:
            models[cond], _meta = load_qmodel(YOLOWorld, args.model, qmodel_path(args.load_model, args, cond), device)
            calib_time[cond] = _meta.get("calib_time", float("nan"))
            model_mib_by[cond] = quantized_weight_mib(models[cond].model)
            continue
        _variants = [_flags0] if "A" not in _flags0 else [(_flags0 - {"A", "M"}), (_flags0 - {"A"}) | {"M"}]
        _mc_path = mcheck_cache_path(args)
        if len(_variants) == 2 and os.path.exists(_mc_path) and not args.mcheck_refresh:
            # 10-28: 이 모델·프로토콜의 M 결정이 이미 있으면 그 후보만 빌드한다(두 번 빌드 + 검사 생략). 결과는
            # 검사를 다시 돌렸을 때 고른 모델과 같다 -- 두 후보 모두 같은 RNG 상태에서 출발하므로.
            import json as _json
            _mc = _json.load(open(_mc_path))
            _variants = [_variants[1] if _mc["use_m"] else _variants[0]]
            print(f"  [+A] 저장된 결정 사용: M {'켬' if _mc['use_m'] else '끔'} ({_mc_path}; "
                  f"M 끔 {_mc['flip_off']:.2f}% / 켬 {_mc['flip_on']:.2f}%, {_mc.get('source', '')})", flush=True)
        _built = []
        t_cond = time.perf_counter()
        for flags in _variants:
            torch.set_rng_state(_rng_snapshot[0])
            if _rng_snapshot[1] is not None:
                torch.cuda.set_rng_state_all(_rng_snapshot[1])
            _np.random.set_state(_rng_snapshot[2]); _py_random.setstate(_rng_snapshot[3])
            c_hi_wbit = tuple(dict.fromkeys(sum((cond_lists[f] for f in "PRH" if f in flags), ())))
            if "G" in flags:
                _attn_idx = {str(i) for i, b in enumerate(fp.model.model) if type(b).__name__ == "C2fAttn"}
                c_hi_wbit = tuple(n for n in c_hi_wbit
                                  if not (n.split(".")[0] in _attn_idx and n.split(".")[1:2] == ["cv2"]))
            print(f"[build] {cond}" + (f" -> 후보 {'+'.join(sorted(flags))}" if len(_variants) > 1 else "")
                  + (f"  (W8 보호 {len(c_hi_wbit)}개 {list(c_hi_wbit)}, scale 이전={next((k for f, k in (('M', '켬'), ('S', 'SmoothQuant'), ('W', 'AWQ')) if f in flags), '끔')}, "
                     f"게이트 교환={'G' in flags})" if flags else ""))
            _m = build(YOLOWorld, args.model, coco, device, calib, mode, fp=fp,
                       pidx=S, h_eval=H_eval, cal_idx=H_cal,
                       w_bits=args.w_bits, a_bits=args.a_bits, skip_head=args.skip_head,
                       first_last_bits=args.first_last_bits, last_abits=args.last_abits,
                       attn_quant=args.attn_quant, hi_wbit_convs=c_hi_wbit,
                       migrate=next((f for f in "MSW" if f in flags), False), gate_commute=("G" in flags),
                       adaround_act_observer=args.adaround_act_observer,
                       combined_stage1=args.combined_stage1,
                       combined_recon_iters=args.combined_recon_iters)
            _built.append((flags, _m))
        if len(_built) == 1:
            models[cond] = _built[0][1]
        else:
            from diag_w4_sensitivity import mcheck_flip
            t_chk = time.perf_counter()
            (_f_off, _m_off), (_f_on, _m_on) = _built
            _fl_off, _fl_on = mcheck_flip([_m_off, _m_on], fp, args.model, coco, args.coco_root, args.calib,
                                          args.imgsz, device)
            _use_m = _fl_on <= MCHECK_RATIO * _fl_off
            models[cond] = _m_on if _use_m else _m_off
            print(f"  [+A] calib 밖 COCO flip: M 끔 {_fl_off:.2f}% / M 켬 {_fl_on:.2f}% -> "
                  f"M {'켬' if _use_m else '끔'} (기준 {MCHECK_RATIO}배, 검사 {time.perf_counter() - t_chk:.1f}s)", flush=True)
            import json as _json
            os.makedirs(MCHECK_CACHE, exist_ok=True)
            _json.dump(dict(use_m=bool(_use_m), flip_off=_fl_off, flip_on=_fl_on, ratio=MCHECK_RATIO,
                            source=f"{cond} W{args.w_bits}A{args.a_bits} seed {args.seed}, "
                                   f"{time.strftime('%Y-%m-%d')}"), open(_mc_path, "w"), indent=1, ensure_ascii=False)
            print(f"  [+A] 결정 저장 -> {_mc_path}", flush=True)
            del _m_off, _m_on, _built
            torch.cuda.empty_cache()
        calib_time[cond] = time.perf_counter() - t_cond
        model_mib_by[cond] = quantized_weight_mib(models[cond].model)
        print(f"  {cond} 빌드 {calib_time[cond]:.1f}s  (이론적 크기 {model_mib_by[cond]:.2f} MiB)")
        if args.save_model:
            save_qmodel(models[cond], qmodel_path(args.save_model, args, cond),
                        dict(cond=cond, args=vars(args), calib_time=calib_time[cond], model_mib=model_mib_by[cond],
                             protect=list(c_hi_wbit)))
    _post = args.post_build or os.environ.get("PTQ_POST_BUILD")
    if _post:
        # 10-01: 진단용 훅 -- 빌드 직후(평가 전) 모델을 그대로 넘겨 스크립트를 실행하고 종료(주 실험 경로 영향 없음).
        print(f"[post-build] {_post} 실행 후 평가 없이 종료", flush=True)
        exec(open(_post).read(), {**globals(), **locals()})   # 한 이름 공간: 스크립트 안 함수끼리 서로 보이게
        return

    if args.odinw_only:
        # 10-27: COCO/LVIS는 이미 잰 조건을 ODinW에서만 다시 잴 때. COCO AP는 기존 로그와 같은 모델인지 확인하는 용도.
        if not args.no_coco_check:
            for cond in conditions:
                (_ap, _), _ = measure_ap(models[cond], args.data, args.imgsz, args.device)
                print(f"  [COCO-check][{cond}] AP={_ap:.2f} (기존 로그의 [COCO-80][{cond}] AP와 같아야 한다)", flush=True)
                for d in (args.save_model, args.load_model):
                    if d:
                        update_qmodel_meta(qmodel_path(d, args, cond), coco_ap=round(_ap, 4))
        eval_odinw_yolo(fp, models, conditions, args, device)
        # COCO/LVIS 표가 없으므로 표 생성기(logparse, make_leaderboard)의 완료 문구("이론적 모델 크기")를 쓰지 않는다.
        print("\n[ODinW 전용 실행 완료] 모델 크기: " + ", ".join(f"{k} {v:.2f} MiB" for k, v in model_mib_by.items()))
        return

    # ---------------- COCO-80: FP sim 1회 계산 + GT 매칭 ----------------
    print(f"\n[gt] COCO-80 FP sim 계산 + anchor 매칭 ({len(probe_paths)}장)")
    h_fp = SimilarityHarness(fp.model, device=device)
    grid_specs = get_grid_specs(h_fp, preprocess(probe_paths[0], args.imgsz))
    fp_sims = [h_fp.run_image(preprocess(p, args.imgsz), i).sim
              for i, p in enumerate(probe_paths)]
    h_fp.close()
    coco_gt_targets = build_gt_targets(fp_sims, probe_paths, coco_gt_by_path, grid_specs, args.imgsz)
    n_coco_gt = sum(len(t) for t in coco_gt_targets)
    print(f"  COCO GT anchor 수={n_coco_gt}")

    results = {}
    for mode in conditions:
        h_q = SimilarityHarness(models[mode].model, device=device)
        q_sims = [h_q.run_image(preprocess(p, args.imgsz), i).sim
                 for i, p in enumerate(probe_paths)]
        h_q.close()
        heval_flip, n1 = group_flip(fp_sims, q_sims, H_eval)
        top1_flip, n2 = standard_flip(fp_sims, q_sims)
        coco_gt_res = gt_metrics_for_method(fp_sims, q_sims, coco_gt_targets, H_eval_set,
                                            S_set=S_set, H_cal_set=H_cal_set)
        del q_sims          # measure_ap()이 model.val() 내부에서 DataLoader worker를
        free_cpu_mem()      # fork하므로, 그 전에 이 조건의 큰 q_sims부터 비워둔다
        (coco_ap, coco_ap50), coco_pc = measure_ap(models[mode], args.data, args.imgsz, args.device)
        s_ap = subset_map(coco_pc, S); heval_ap = subset_map(coco_pc, H_eval)
        lrg = coco_gt_res["lost_rate_by_group"]; lbg = coco_gt_res["lost_by_group"]; dbg = coco_gt_res["denom_by_group"]
        print(f"  [COCO-80][{mode}] AP={coco_ap:.2f} S_AP={s_ap:.2f} H_eval_AP={heval_ap:.2f} "
              f"Heval_flip={heval_flip:.2f}% Top1_flip={top1_flip:.2f}% "
              f"GT_MRR={coco_gt_res['mrr']:.4f} GT_R@1={coco_gt_res['r1']:.4f} "
              f"lost={coco_gt_res['lost']} gained={coco_gt_res['gained']} lateral={coco_gt_res['lateral']} "
              f"corrective_rate={coco_gt_res['corrective_rate']:.2f}% UPIR={coco_gt_res['upir']:.2f}%")
        print(f"    [lost by group] S={lbg['S']}/{dbg['S']}({lrg['S']:.2f}%) "
              f"H_cal={lbg['H_cal']}/{dbg['H_cal']}({lrg['H_cal']:.2f}%) "
              f"H_eval={lbg['H_eval']}/{dbg['H_eval']}({lrg['H_eval']:.2f}%)")
        if args.save_model:
            update_qmodel_meta(qmodel_path(args.save_model, args, mode), coco_ap=round(coco_ap, 4))
        results[mode] = dict(coco_ap=coco_ap, s_ap=s_ap, heval_ap=heval_ap, heval_flip=heval_flip,
                             top1_flip=top1_flip, coco_gt=coco_gt_res, calib_time=calib_time[mode],
                             lost_rate_by_group=lrg)

    # FP32 COCO-80 AP -- 반드시 fp의 vocabulary를 LVIS로 바꾸기 전에 재야 함
    # (바꾼 뒤 COCO 80-class validator에 넣으면 confusion matrix index가 깨짐)
    (fp_coco_ap, fp_coco_ap50), _ = measure_ap(fp, args.data, args.imgsz, args.device)

    # ---------------- LVIS-1203(minival): 이식, 스트리밍(이미지 1장씩) ----------------
    print(f"\n[switch] 전부 LVIS-1203 vocab으로 이식 (minival {len(lvis_probe_paths)}장)")
    switch_vocab(fp, lvis_names, device)
    for mode in conditions:
        switch_vocab(models[mode], lvis_names, device)

    h_fp = SimilarityHarness(fp.model, device=device)
    lvis_grid_specs = get_grid_specs(h_fp, preprocess(lvis_probe_paths[0], args.imgsz))
    h_models = {mode: SimilarityHarness(models[mode].model, device=device) for mode in conditions}

    print(f"  [streaming] flip/GT_MRR/R@1/lost/gained 계산 중 ({len(lvis_probe_paths)}장 x {len(conditions)}조건)")
    lvis_stream_res = compute_lvis_flip_gt_streaming(h_fp, h_models, lvis_probe_paths,
                                                     lvis_gt_by_path, lvis_grid_specs, args.imgsz)
    h_fp.close()
    for h_q in h_models.values():
        h_q.close()
    n_lvis_gt = lvis_stream_res[conditions[0]]["n"]
    print(f"  LVIS GT anchor 수={n_lvis_gt}")
    for mode in conditions:
        r = lvis_stream_res[mode]
        print(f"  [LVIS-flip/GT][{mode}] Top1_flip={r['top1_flip']:.2f}%(n={r['n_flip']}) "
              f"GT_MRR={r['mrr']:.4f} GT_R@1={r['r1']:.4f} lost={r['lost']} gained={r['gained']} "
              f"lateral={r['lateral']} corrective_rate={r['corrective_rate']:.2f}%")

    for mode in conditions:
        print(f"[predict+AP] {mode} (minival {len(lvis_probe_paths)}장) ...")
        preds = predict_lvis_results(models[mode], lvis_probe_paths, lvis_probe_ids, args.imgsz, device)
        lvis_ap_res = run_lvis_eval(lvis_gt, preds, lvis_probe_ids)
        print(f"  [LVIS-AP][{mode}] AP={lvis_ap_res.get('AP',0):.4f} AP50={lvis_ap_res.get('AP50',0):.4f} "
              f"APr={lvis_ap_res.get('APr',0):.4f} APc={lvis_ap_res.get('APc',0):.4f} "
              f"APf={lvis_ap_res.get('APf',0):.4f}")
        r = lvis_stream_res[mode]
        results[mode].update(lvis_ap=lvis_ap_res.get("AP", 0.0), lvis_ap50=lvis_ap_res.get("AP50", 0.0),
                             lvis_apr=lvis_ap_res.get("APr", 0.0), lvis_apc=lvis_ap_res.get("APc", 0.0),
                             lvis_apf=lvis_ap_res.get("APf", 0.0),
                             lvis_top1_flip=r["top1_flip"],
                             lvis_gt=dict(mrr=r["mrr"], r1=r["r1"], lost=r["lost"], gained=r["gained"],
                                         lateral=r["lateral"], corrective_rate=r["corrective_rate"]))

    print("\n[predict+AP] FP32 (참고, LVIS) ...")
    preds_fp = predict_lvis_results(fp, lvis_probe_paths, lvis_probe_ids, args.imgsz, device)
    fp_lvis = run_lvis_eval(lvis_gt, preds_fp, lvis_probe_ids)

    # ---------------- ODinW-13 (10-27) ----------------
    if args.odinw:
        odinw_res = eval_odinw_yolo(fp, models, conditions, args, device)
        for k in conditions:
            results[k]["odinw"] = odinw_res[k]

    print("\n" + "=" * 100)
    lsq_bits = []
    if not args.skip_head:
        lsq_bits.append("head 양자화")
    if args.combined_stage1 != "none":
        lsq_bits.append(f"Combined 1단계={args.combined_stage1}")
    if args.first_last_bits > 0:
        lsq_bits.append(f"첫/마지막 레이어 {args.first_last_bits}bit")
    if args.last_abits > 0:
        lsq_bits.append(f"head 마지막 conv 입력 A{args.last_abits}")
    if args.attn_quant != "none":
        lsq_bits.append(f"attention 8bit({args.attn_quant})")
    lsq_tag = f" [{', '.join(lsq_bits)}]" if lsq_bits else ""
    print(f" 공식 데이터 설정(calib=train2017 {len(calib_paths)}장, LVIS=공식 minival) -- seed {args.seed}{lsq_tag}")
    if args.eval_cap > 0:
        print(f" 주의: --eval-cap={args.eval_cap}은 COCO_AP/S_AP/H_eval_AP(measure_ap, --data yaml의 "
              f"고정 val split 사용)에는 적용 안 됨 -- 항상 전체 val 세트로 측정됨. "
              f"LVIS_AP/APr/APc/APf는 probe_paths에서 파생돼 --eval-cap이 적용됨(위 LVIS 채점 "
              f"{len(lvis_probe_paths)}장 참고). flip/GT/UPIR/lost(아래 두 번째 표)도 --eval-cap 적용됨.")
    print("=" * 100)
    print(f"{'':>10} | {'COCO AP':>8} | {'S_AP':>7} | {'H_eval_AP':>9} | {'LVIS AP':>8} | {'APr':>6} | {'APc':>6} | {'APf':>6}")
    print(f"{'FP32':>10} | {fp_coco_ap:>8.2f} | {'-':>7} | {'-':>9} | {fp_lvis.get('AP',0):>8.4f} | "
          f"{fp_lvis.get('APr',0):>6.4f} | {fp_lvis.get('APc',0):>6.4f} | {fp_lvis.get('APf',0):>6.4f}")
    for mode in conditions:
        r = results[mode]
        print(f"{mode:>10} | {r['coco_ap']:>8.2f} | {r['s_ap']:>7.2f} | {r['heval_ap']:>9.2f} | "
              f"{r['lvis_ap']:>8.4f} | {r['lvis_apr']:>6.4f} | {r['lvis_apc']:>6.4f} | {r['lvis_apf']:>6.4f}")

    print("\n" + "=" * 100)
    print(" flip / GT / UPIR / 비용")
    print("=" * 100)
    print(f"{'':>10} | {'Heval_flip':>10} | {'Top1_flip':>9} | {'UPIR':>6} | {'lost':>5} | "
          f"{'CorrRate':>8} | {'LVIS_flip':>9} | {'LVIS_lost':>9} | {'L_CorrR':>8} | {'calib(s)':>9}")
    for mode in conditions:
        r = results[mode]
        print(f"{mode:>10} | {r['heval_flip']:>9.2f}% | {r['top1_flip']:>8.2f}% | "
              f"{r['coco_gt']['upir']:>5.2f}% | {r['coco_gt']['lost']:>5} | "
              f"{r['coco_gt']['corrective_rate']:>7.2f}% | "
              f"{r['lvis_top1_flip']:>8.2f}% | {r['lvis_gt']['lost']:>9} | "
              f"{r['lvis_gt']['corrective_rate']:>7.2f}% | {r['calib_time']:>9.1f}")

    print("\n" + "=" * 100)
    print(" lost 그룹별 분해 (S/H_cal/H_eval) -- \"lost가 H_eval에 몰리는가\" 가설 확인용")
    print("=" * 100)
    print(f"{'':>10} | {'lost_rate_S':>11} | {'lost_rate_H_cal':>15} | {'lost_rate_H_eval':>16}")
    for mode in conditions:
        lrg = results[mode]["lost_rate_by_group"]
        print(f"{mode:>10} | {lrg['S']:>10.2f}% | {lrg['H_cal']:>14.2f}% | {lrg['H_eval']:>15.2f}%")

    # 표 생성기(logparse.py, make_leaderboard.py)가 "이론적 모델 크기" 문구로 실행 완료 여부를 판단한다.
    print("\n이론적 모델 크기: " + ", ".join(f"{k} {v:.2f} MiB" for k, v in model_mib_by.items()))
    update_leaderboard()


def update_leaderboard():
    """환경변수 PTQ_LEADERBOARD=1이면 실험이 끝날 때 LEADERBOARD.html과 leaderboard.csv를 다시 만든다
    (scripts/leaderboard/make_leaderboard.py). 두 파일은 git에 들어 있고 runs/의 로그 전체로 다시 쓰이므로
    기본은 꺼 둔다. 여러 실험이 동시에 끝나도 겹치지 않게 파일 잠금을 쓰고, 실패해도 실험 결과에는 영향이 없다."""
    if os.environ.get("PTQ_LEADERBOARD") != "1":
        return
    sys.stdout.flush()                                    # 이 실험의 결과 표가 로그에 다 써진 뒤 읽게 한다
    try:
        import fcntl, importlib.util
        root = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(root, "runs", ".leaderboard.lock"), "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            spec = importlib.util.spec_from_file_location(
                "make_leaderboard", os.path.join(root, "scripts", "leaderboard", "make_leaderboard.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.main()
    except Exception as e:                                # 리더보드 실패는 실험 실패가 아니다
        print(f"[leaderboard] 갱신 실패(무시): {e}")


if __name__ == "__main__":
    main()
