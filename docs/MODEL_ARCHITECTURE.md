# 최종 모델 구조와 작동 원리 (2026-10-27 기준)

**대상 독자:** 이 방법을 처음 보는 공저자나, 코드를 고치거나 다시 돌릴 사람. 10월 중의 수정(결정적 pooling, OWLv2 지원, SmoothQuant/AWQ 비교, 디렉터리 정리)을 모두 반영한 **지금 코드 그대로의** 구조를 설명한다. 연구 경위는 [`HISTORY.md`](HISTORY.md), 근거 실험은 [`METHOD.md`](METHOD.md), 논문 방향은 [`PAPER_DIRECTION.md`](PAPER_DIRECTION.md)에 있다.

---

## 1. 한눈에 보기

학습 없는 PTQ다. FP 검출기 하나와 calibration 이미지 256장(COCO train2017)을 받아, 아래 순서로 양자화 모델을 만든다.

```
FP 검출기
  │ ⓪ G  텍스트 게이트 교환 (구조만 바꿈, FP 등가)         ← 텍스트 게이트 융합 블록이 있을 때만
  │    확정 프로토콜 적용 (모든 conv·Linear, attention, region–텍스트 내적 양자화)
  │ ①  P  진단이 고른 소수 층을 W8로 보호                    ← 모델당 한 번 만든 진단 캐시
  │ ②  M  공유 제약 scale 이전                              ← 모델당 한 번 사전 검사로 켤지 결정
  │    calibrate (activation scale 초기화, MSE)
  │ ③  BRECQ (또는 QDrop) 재구성: rounding + activation step 학습
  ▼
양자화 모델 (W4A8 / W4A6 / W4A5 / W8A8)
```

| 요소 | 무엇을 고치나 | 언제 켜나 | 추가 비용 |
|---|---|---|---|
| ⓪ G 게이트 교환 | 텍스트 attention 분기의 outlier가 융합 conv의 scale을 오염시키는 것 | 구조에 텍스트 게이트 융합(C2fAttn)이 있으면 | 없음 (FP 등가, 비트·연산량 동일) |
| ① P 보호 | 손상이 몰린 소수 층 | 항상 (예산 1.5%) | 크기 +0.3% (G와 함께) ~ +1.3% |
| ② M 공유 제약 이전 | per-tensor activation의 채널 편차 | 사전 검사가 승인하면 | 채널별 곱 1회(앞 층 requant에 흡수) |
| ③ BRECQ | 나머지 반올림 오차 | 항상 | PTQ 시간만 |

## 2. 대상 모델과 실제로 쓰이는 구성

| 모델 | 구조 | 진입점 | G | P | M (사전 검사 결과) | 최종 구성 |
|---|---|---|---|---|---|---|
| YOLO-World-S (v1) | YOLOv8 CNN + C2fAttn 텍스트 게이트 | `run_comparison.py` | ✅ | ✅ | 켬 | GPM |
| YOLO-World-S v2 | 같음 | `run_comparison.py` | ✅ | ✅ | 켬 | GPM |
| YOLO-World-M | 같음 (큰 모델) | `run_comparison.py` | ✅ | ✅ (G가 대상 대부분을 맡아 실제로는 없음) | **끔** (M을 켜면 flip 2~7배) | GP |
| YOLOE-v8s | YOLOv8 CNN, 텍스트는 head의 contrastive에만 | `run_comparison.py` | 해당 없음 | ✅ | 켬 | PM |
| YOLOE-11s | YOLO11 CNN(C3k2, C2PSA) | `run_comparison.py` | 해당 없음 | ✅ | 켬 | PM |
| OWLv2-B/16 (self-training 체크포인트) | CLIP ViT-B/16 + late fusion head | `run_owl.py` | 해당 없음 | ✅ (효과 없음) | 켬 | PM |

조건 이름은 `<기반>+<접미사>`다. 우리 방법은 모든 모델에 `brecq+GPM`을 넘기고, 적용할 수 없는 요소는 코드가 알아서 건너뛴다(YOLOE·OWLv2에서 G는 아무것도 하지 않는다).

## 3. 양자화 프로토콜 (모든 조건·모델에 같게)

`scripts/protocol.sh`의 인자:

```
--deterministic --calib 256 --no-skip-head --first-last-bits 8
--attn-quant attn_cls --last-abits 16
```

| 항목 | YOLO 계열 | OWLv2 |
|---|---|---|
| 양자화 범위 | 모든 conv(head 포함) + attention Linear/matmul + head의 region–텍스트 내적 | 비전 쪽 모든 Linear(1×1 conv로 바꿔서) + 패치 conv + attention q/k/v + region–텍스트 내적 |
| weight | 출력 채널별 비대칭, MSE로 범위 탐색 | 같음 |
| activation | per-tensor 비대칭, MSE(L2.4)로 범위 탐색 | 같음 |
| 첫 레이어 | stem 첫 conv W8A8 | 패치 conv W8A8 |
| 마지막 레이어 | head cv2/cv3 마지막 1×1: W8, **입력 A16** | class/box/objectness 출력층: W8, 입력 A16 |
| 구조상 예외 | – | **softmax 출력 16bit** (키 3601개라 확률 대부분이 1/3601 근처인데, 8bit 균일 격자는 1/255 간격이라 거의 0이 된다) |
| 텍스트 쪽 | 어휘가 고정된 배포를 가정해 텍스트 임베딩은 미리 계산한 상수. 상수는 행별 8bit | 텍스트 인코더는 FP 상수, 내적에 쓰는 텍스트는 행별 8bit |
| FP로 남는 것 | LayerNorm, softmax, sigmoid, max, residual add, DFL, NMS | LayerNorm, GELU, residual add, sigmoid, elu |

## 4. 빌드 순서 (`run_comparison.py::build`)

1. FP 모델을 불러오고 conv-BN을 합친다(`fuse`). 이때 `core/model_zoo.py`가 ImagePoolingAttn의 pooling을 **결정적 구현**으로 바꾼다(§9).
2. **G**: `quant/fusion_quant.gate_commute_all` — C2fAttn의 cv2를 쪼개고 forward를 교체한다.
3. conv를 `QuantConv2d`로 감싼다(`wrap_convs`). 첫·마지막 비트와 마지막 입력 A16을 지정한다.
4. **P**: 진단 캐시에서 고른 층의 weight 비트를 8로 올린다(`set_conv_wbits`). G를 켰으면 G가 대신 다루는 C2fAttn cv2는 목록에서 뺀다.
5. attention·내적 양자화 지점을 끼운다(`quant/attn_quant.quantize_attention`).
6. **M**: `quant/channel_graph.search_and_apply_tied` — calibration 8장으로 공유 그룹별 s를 고르고 적용한다.
7. calibrate: 256장을 흘려 activation 범위(MSE)와 weight scale을 잡는다.
8. **BRECQ**: backbone·neck은 블록 단위, head는 conv 단위로 2000 iter 재구성한다. rounding(AdaRound α)과 activation step(LSQ, 16bit 미만만)을 함께 학습하고, 목표는 같은 구조로 바꾼 FP 모델의 출력이다.

M을 켤지 정하는 사전 검사(`+A`, §7)는 3~8을 M 끔/켬 두 번 돌려 비교한 결과를 모델당 한 번 쓴다. 모든 조건은 빌드 직전에 RNG를 같은 상태로 되돌려, 조건 순서가 결과를 바꾸지 않는다.

## 5. ⓪ G — 텍스트 게이트 교환

**문제.** YOLO-World의 C2fAttn은 네 분기 `[y0, y1, y2, a]`를 concat해 1×1 conv(cv2) 하나로 합친다. 텍스트 attention 분기 `a = p ⊙ aw`(p = 이미지 특징의 projection, aw = 위치별 텍스트 게이트)는 값과 weight가 다른 분기보다 훨씬 크다(중앙값의 87배). 출력 채널별 weight scale을 이 분기가 차지하면서, 나머지 세 분기의 작은 weight가 4bit 격자에서 0 근처로 뭉개진다. 이 conv 하나만 W4로 내려도 LVIS flip이 6.5%에서 79%가 된다.

**변환.** 1×1 conv는 선형이고 게이트는 위치별 스칼라라서 FP에서 정확히 같다.

```latex
\mathrm{cv2}\big(\mathrm{cat}(y_0, y_1, y_2, \textstyle\sum_h aw_h \odot p_h)\big)
= W_{\text{main}} \cdot \mathrm{cat}(y_0, y_1, y_2) + b + \sum_h aw_h \odot (W_h \cdot p_h)
```

코드(`quant/fusion_quant.gate_commute`): cv2의 weight 열을 앞 분기용 `cv2_main`(bias 포함)과 head별 `cv2_side[h]`로 쪼개고, forward를 "게이트를 곱하기 전의 p를 cv2_side에 넣고, 그 출력에 aw를 곱해 더한 뒤 SiLU"로 바꾼다.

**양자화에서 달라지는 것.**
1. 텍스트 분기가 자기 conv를 가져 다른 분기와 weight·activation scale을 공유하지 않는다.
2. 양자화되는 입력이 게이트 전 값이라, 게이트가 만드는 위치별 크기 편차가 범위에서 빠진다.
3. 게이트가 작은 위치(텍스트와 무관한 영역)에서는 양자화 오차도 게이트만큼 줄어든다.

scale만 분리하면 flip이 68.8% → 18.5%, 게이트까지 뒤로 옮기면 9.7%다. 배포에서는 1×1 conv 두 개 + 위치별 곱 + 덧셈이고, 연산량과 비트는 원래와 같다.

## 6. ① P — 누수 없는 진단과 보호

**진단** (`core/diag_w4_sensitivity.rank_convs_leakfree`, 결과는 `configs/protect_cache/`에 모델당 한 번 저장)
1. naive W8A8 모델(확정 프로토콜)을 기준으로 둔다.
2. 층 하나만 weight를 W4로 내리고 calibration 쪽 이미지(train2017 200장)를 흘린다.
3. FP가 확신하는(sigmoid > 0.25) 위치에서 **COCO top-1이 바뀐 비율(flip)의 증가**를 잰다.
4. 모든 층에 반복한다(첫·마지막 고정 층 제외). val2017과 LVIS는 읽지도 않는다.

**선택** (`select_protected`): 증가가 큰 층부터 담되, 양자화 weight 파라미터의 **1.5%** 안에서(110%를 넘지 않게, 90% 이상 차면 멈춤). 고른 층은 weight만 W8이고 activation 비트는 그대로다.

| 모델 | 보호 대상 (G를 켰을 때) | 예산이 잡아내는 손상량 |
|---|---|---|
| YOLO-World-S | 1, 2.cv2, 4.cv1 (12.cv2는 G가 맡음) | 41.4%p |
| YOLO-World-M | 없음 (12.cv2, 15.cv2 모두 G가 맡음) | 30.1%p |
| YOLOE-v8s | 1, 2.cv2, 4.cv2, 15.cv2 등 초반 backbone 중심 | 21.1%p |
| YOLOE-11s | 4.cv2, 2.cv2, 2.m.0.cv2, 1, 4.cv1, head 1개 | 12.8%p |
| OWLv2 | 3층 k_proj, 5층 out_proj | 4.6%p (잡음 ±1.6%p) |

**작동 원리.** 손상이 소수 층에 몰려 있으면 그 층만 8bit로 두는 것이 비트를 고르게 더 쓰는 것보다 훨씬 효율적이다. 같은 예산의 무작위 보호는 +0.21에 그치고 진단 기반은 +1.05다. 반대로 OWLv2처럼 손상이 퍼져 있으면 효과가 없다. 그래서 진단의 포착량이 P가 들을지를 미리 알려 준다.

## 7. ② M — 공유 제약 scale 이전

**기본 아이디어.** 입력 채널 c마다 양수 s_c를 두면 FP 함수는 그대로다.

```latex
y = \mathrm{conv}(x, W) = \mathrm{conv}\big(x / s,\; W \cdot \mathrm{diag}(s)\big)
```

x의 채널 편차를 s로 평평하게 만들면 per-tensor activation 격자가 덜 낭비되고, 그만큼 weight 열이 커진다.

**배포 제약.** 배포에서는 `x/s`를 앞 층(생산자)이 출력할 때 requant 단계에 곱한다. 그래서
1. 같은 생산 채널을 받는 모든 소비 conv는 같은 s를 써야 하고,
2. residual add로 합쳐지는 생산 채널들도 같은 s여야 한다.

`quant/channel_graph.py`는 이 관계를 모델 구조를 하드코딩하지 않고 **측정으로** 찾는다. 생산 conv 출력을 채널별로 다르게 나눠 두 번 흘리고, 소비 conv 입력의 변화 비율로 어느 생산 채널에서 왔는지 역산한다. 그 결과를 union-find로 묶어 공유 그룹을 만든다(YOLO-World-S: 생산 채널 10,592개 → 9,920그룹). OWLv2는 구조가 고정이라 공유 단위를 직접 적는다(LN 뒤 q/k/v, LN 뒤 fc1, GELU 뒤 fc2, head 입력층).

**s 고르기.** 단위마다 `s = max|x|^α / max|W|^(1−α)`에서 α ∈ {0, 0.25, 0.5, 0.75, 1}과 "이전 없음(s=1)" 중, 영향받는 모든 소비 conv의 **W·A를 함께 양자화한 출력 오차**가 가장 작은 것을 고른다(calibration 8장). s=1이 후보라서 어떤 단위도 이전 때문에 나빠지지 않는다.

**사전 검사 (`+A`, `core/diag_w4_sensitivity.mcheck_flip`).** M은 취약성을 없애지 않고 옮긴다(12.cv2 민감도는 줄고 1.conv 민감도는 3.5~4배 는다). 그래서 G·P가 받쳐 줘도 모델에 따라 해로울 수 있다. calibration에 쓰지 않은 train2017 200장과 COCO 어휘로 M 끔/켬 모델의 flip을 비교해, 켠 쪽이 **1.5배를 넘으면 끈다**. 모델당 한 번 정하고 모든 비트에 쓴다. 문턱을 1.1~2배 어디에 둬도 결정이 같다. 한계: YOLO-World-S v2 W8A8처럼 LVIS에서만 나빠지는 경우는 잡지 못한다.

**대체 규칙.** 같은 공유 제약 아래에서 s 규칙만 바꾼 SmoothQuant(`+S`, α=0.5 고정)와 AWQ(`+W`, `mean|x|^α`, weight만 양자화한 오차로 α 탐색)를 비교용으로 구현했다. G+P 위에서는 어느 규칙이든 비슷하고, OWLv2에서는 M이 AWQ보다 판정 flip이 1.8%p 낮다.

## 8. ③ BRECQ 재구성

`quant/brecq.optimize_brecq`. 앞 블록부터 차례로, 그 블록에 실제로 들어오는 입력(앞 블록들의 양자화 오차가 쌓인 값)과 FP 블록의 출력을 맞추도록 rounding과 activation step을 2000 iter 학습한다. 손실은 BRECQ 공식 구현과 같은 정규화된 L2, warmup 20%, rounding 정규화 가중치 0.01이다. 16bit activation(A16 입력)은 step이 발산하므로 학습하지 않는다(`LSQ_MAX_BITS`).

G를 켠 모델은 FP 기준 모델에도 같은 게이트 교환을 적용해, quant/FP conv가 같은 짝을 이루게 한다(FP 등가라 목표는 같다). QDrop을 기반으로 쓰면 같은 루프에 drop 확률 0.5가 더해진다.

## 9. 재현성: 결정적 pooling (10-16 수정)

YOLO-World의 ImagePoolingAttn에 있는 `AdaptiveMaxPool2d`는 backward에서 겹치는 창의 gradient를 atomicAdd로 더해, `--deterministic`에서도 실행마다 값이 달랐다. BRECQ가 이 경로로 역전파하므로 같은 seed·같은 조건이 다른 모델을 냈다. `core/model_zoo.DetAdaptiveMaxPool2d`가 argmax는 원래 연산으로 구하고 값은 `gather`로 뽑아, forward는 bit 단위로 같고 backward는 결정적이다. 수정 후 전체 표를 다시 돌렸고 448칸 중 440칸이 기존과 bit 단위로 같았다. 이제 같은 GPU 종류에서 대기열을 다시 돌리면 결과가 bit 단위로 재현된다.

## 10. OWLv2를 같은 파이프라인에 올린 방법

`core/owl_zoo.py`, `core/owl_quant.py`, `run_owl.py`.
- **Linear → 1×1 conv** (`LinearConv`): 토큰 `[.., C]`를 `[1, C, M, 1]`로 바꿔 흘린다. 같은 연산이고 양자화 단위(activation per-tensor, weight 출력 채널별)도 같아서 `QuantConv2d`, calibrate, AdaRound/BRECQ, 보호 목록 코드를 그대로 쓴다.
- **블록 리스트로 펼치기** (`OwlDet.model`): [stem(패치 conv + 위치 임베딩 + pre-LN), ViT 층 12개, head]. BRECQ는 YOLO와 같은 방식으로 블록 단위 재구성한다.
- **체크포인트:** `google/owlv2-base-patch16`(self-training만). `-ensemble`/`-finetuned`는 LVIS base로 fine-tuning돼 LVIS가 held-out 어휘가 아니다.
- **평가:** NMS 없이 패치마다 상위 점수를 채점(OWLv2 기본 후처리와 같음). TF32를 꺼야 원 모델과 logits가 맞는다(켜면 최대 2.2 차이, 끄면 0.002).

## 11. 배포 형태

| 구성요소 | 배포에서의 모습 | 확인 상태 |
|---|---|---|
| 기본 양자화 | W4(또는 W8) weight 출력 채널별, activation per-tensor INT | 시뮬레이션(fake-quant) |
| G | 1×1 conv 두 개 + 위치별 곱 + 덧셈 + SiLU | FP 등가 확인, 엔진 fusion 미확인 |
| P | 해당 층만 INT8 weight (층별 정밀도) | 시뮬레이션 |
| M | 생산 층 requant 계수에 1/s, 소비 층 weight에 s를 미리 곱함 | 배포형 FP 등가(상대 오차 5.5e-6) 확인 |
| head 입력 A16 | 해당 1×1 conv 6개만 INT16 activation × INT8 weight | 시뮬레이션 |
| 사전 검사·진단 | PTQ 시점에만. 배포 모델에는 없음 | – |

모델 크기(YOLO-World-S W4): BRECQ 6.58 MiB, 우리 6.60 MiB(+0.3%), FP32 약 49 MiB. 실기기 지연은 아직 재지 않았다.

## 12. 코드 위치

| 무엇 | 파일 |
|---|---|
| 진입점 | `run_comparison.py`(YOLO 계열), `run_owl.py`(OWLv2) |
| G | `quant/fusion_quant.py` |
| P 진단·선택, M 사전 검사 | `core/diag_w4_sensitivity.py`, 캐시 `configs/protect_cache/` |
| M (YOLO) | `quant/channel_graph.py` (`rule="tied"`; 비교용 `"smooth"`, `"awq"`) |
| M (OWLv2), OWLv2 프로토콜 | `core/owl_quant.py` |
| 양자화기 | `quant/fake_quant.py`, `quant/quant_model.py`, `quant/attn_quant.py` |
| BRECQ/QDrop/AdaRound | `quant/brecq.py`, `quant/adaround.py` |
| 모델 생성, 결정적 pooling | `core/model_zoo.py`, `core/owl_zoo.py` |
| 실행 | `scripts/protocol.sh`, `scripts/worker.sh`, `scripts/worker_owl.sh`, `scripts/run_mixed.sh`, `scripts/track.sh` |

주요 상수(`run_comparison.py` 상단): `PROTECT_BUDGET=0.015`, `PROTECT_IMAGES=200`, `MIG_ALPHAS=(0, 0.25, 0.5, 0.75, 1)`, `MIG_IMAGES=8`, `RECON_ITERS=2000`, `MCHECK_RATIO=1.5`.
