# 논문 개요 v2 — 진단이 처방을 고르는 PTQ (2026-10-19 초안)

**대상 독자:** 공저자. 기존 개요([`PAPER_OUTLINE.md`](PAPER_OUTLINE.md), 이하 v1)를 대체하는 문서가 아니라, 같은 실험 결과로 **논문의 중심을 "방법(GPM)"에서 "분석과 진단"으로 옮긴 판**이다. 두 판을 나란히 놓고 어느 쪽으로 쓸지 정하기 위해 따로 둔다.

**관련 자료**
- 수치: [`RESULTS.md`](RESULTS.md)(표 1~5), `CVPR2027/runs/`(10-16 결정적 코드로 다시 낸 로그), `results/LEADERBOARD.html`
- 집중도 표와 그림: [`../figures/src/diag_concentration.py`](../figures/src/diag_concentration.py) → `figures/sensitivity_maps.{pdf,png}`
- 방법의 작동 원리: [`METHOD.md`](METHOD.md)

**상태 표시:** ✅ 근거 있음 · 🟡 일부(seed 부족 등) · ⏳ 진행 중 · ❌ 아직 없음

---

## 0. 왜 다시 짜는가

v1은 GPM을 하나의 방법으로 세운다. 실험이 쌓이면서 리뷰어가 짚을 약점이 분명해졌다.

| 약점 | 근거 |
|---|---|
| P는 혼합 정밀도 계열이다 | 같은 예산에서 HAWQ식(출력 MSE) 기준과 거의 같은 층을 고른다(W4A8 +0.90 vs +1.06, 비유의) |
| M은 SmoothQuant/AWQ 계열이다 | M 자리에 AWQ(GPW)나 SmoothQuant(GPS)를 넣어도 이득이 비슷하다(`runs/170_sw`) |
| G는 적용 범위가 좁다 | 텍스트 게이트가 있는 YOLO-World 계열에만 해당. YOLOE, OWLv2에는 없다. s W4A8에서는 PM이 GPM보다 0.34점 높다 |
| 모델마다 G/P/M 중 쓸 수 없거나 해로운 것이 있다 | m에서 M이 해롭다(검사로 끔), OWLv2에서 P가 효과 없다 |

방법을 전면에 두면 "기존 기법 두 개 + 좁은 트릭 하나"로 읽히기 쉽다. 반면 **문제 발견과 손상 위치 분석은 여섯 모델 모두에서 성립**하고, 위의 "모델마다 다르다"는 사실이 오히려 분석으로 설명된다. 그래서 중심을 옮긴다.

## 1. 한 문장 요약

저비트 PTQ는 open-vocabulary 검출기의 **보지 않은 어휘에 대한 판정을 AP보다 먼저 무너뜨린다.** 손상이 **어디에 어떻게 몰리는지는 구조마다 다르고**, calibration 쪽 데이터만으로 하는 누수 없는 진단이 그것을 미리 보여준다. 진단 결과에 따라 **구조 재배치(G), 보호(P), 공유 제약 이전(M)** 중 무엇이 들을지가 정해지고, 그렇게 고른 처방은 학습 없이 W4에서 여섯 검출기의 판정을 지킨다.

## 2. 제목 후보

1. *Where Low-Bit Quantization Breaks Open-Vocabulary Detectors: Diagnosis-Guided Post-Training Quantization*
2. *Diagnose, Then Quantize: Preserving Held-Out Vocabulary in Low-Bit Open-Vocabulary Detection*
3. *Quantization Damage in Open-Vocabulary Detectors Is Structured — and Diagnosable*

1번을 권한다. "어디서"(분석)와 "진단 기반"(방법)을 함께 담는다. v1의 1번 제목과 거의 같아서 바꾸는 비용도 작다.

## 3. 초록 (초안)

> Open-vocabulary detectors are deployed on edge devices where the vocabulary changes after deployment, yet post-training quantization (PTQ) is tuned and evaluated on the calibration vocabulary. We show that this hides the main failure: at W4A8, BRECQ keeps COCO AP within 3 points of full precision on YOLO-World-S, but flips the top-1 class of 18% of confident predictions on LVIS. Using a leak-free, layer-wise diagnosis that only touches calibration images and vocabulary, we find that this damage is structured, and that its structure differs across architectures: in text-gated detectors (YOLO-World) a single text-fusion convolution alone raises the flip rate by 23–31 points, gate-free CNN detectors (YOLOE) concentrate damage at early backbone split/concat boundaries, and a ViT detector (OWLv2) shows no layer whose damage stands clearly above measurement noise. The diagnosis tells which training-free remedy applies: a bit-free gate commutation that removes the shared-scale contamination of text-gated fusion, protection of the few diagnosed layers, and scale migration under deployment constraints, enabled only when a leak-free check approves it. Across six detectors (YOLO-World S/M/S-v2, YOLOE-v8s/11s, OWLv2) with all convolutions, attention and region–text products quantized, the selected remedies improve held-out LVIS AP over BRECQ by up to +5.8 points at W4 and reduce decision flips by up to 27 points, while the benefit of protection tracks how much damage the diagnosis localizes.

수치는 아래 §6과 맞춰 마지막에 다시 확인한다. 특히 OWLv2 W4A8은 seed가 모이면 넣는다.

## 4. 기여

| # | 기여 | 상태 |
|---|---|---|
| C1 | **문제:** 저비트 PTQ는 held-out 어휘 판정을 AP보다 먼저, 더 크게 무너뜨린다. 판정 충실도(LVIS flip / lost)를 AP와 함께 보고해야 한다. 여섯 모델 모두에서 재현 | ✅ (OWLv2 W4A8 🟡) |
| C2 | **분석:** 누수 없는 층 단위 진단과, 그 결과가 구조마다 다르다는 발견(텍스트 융합 conv 집중 / 초반 backbone 집중 / 분산). 메커니즘: 게이트된 attention 분기의 outlier가 공유 scale을 오염 | ✅ |
| C3 | **진단 기반 처방:** 진단이 무엇을 쓸지 알려준다. G(텍스트 게이트가 있으면, FP 등가·비트 비용 0), P(진단이 잡아내는 손상이 크면), M(누수 없는 사전 검사가 승인하면). P의 효과가 진단이 잡아내는 손상량과 같은 순서로 움직인다는 것을 여섯 모델로 보인다 | ✅ 5모델 · 🟡 OWLv2 |
| C4 | **엄격한 프로토콜과 재현성:** head 포함, attention·region–텍스트 내적까지 8bit. 결정적 실행(재실행 시 bit 단위 재현) | ✅ |

v1과 달라지는 점: G를 "주인공 방법"이 아니라 **C2의 메커니즘 분석에서 나온 구조적 처방**으로 둔다. P와 M이 기존 기법이라는 점은 숨기지 않고, "언제 쓰는가"를 진단이 정한다는 데 기여를 둔다. M을 AWQ/SmoothQuant 규칙으로 바꿔도 된다는 비교는 그대로 ablation으로 싣는다.

## 5. 본문 구성 (8쪽)

### 1. Introduction (1.0쪽)
- 엣지에서 어휘가 바뀌는 검출. calibration 어휘의 AP만 보면 W4A8 BRECQ가 쓸 만해 보인다(**그림 1**: 비트별 COCO AP와 LVIS flip).
- 질문 세 개로 이끈다: 무엇이 망가지는가(C1), 어디서 왜(C2), 그래서 무엇을 써야 하는가(C3).
- 기여 C1~C4.

### 2. Related Work (0.6쪽)
- PTQ(AdaRound, BRECQ, QDrop, PD-Quant), 검출 PTQ(Reg-PTQ), ViT PTQ(PTQ4ViT, RepQ-ViT: softmax 등 특수 양자화기)
- outlier와 scale 이전(SmoothQuant, AWQ, OmniQuant), CNN의 등가 변환(DFQ의 CLE. SiLU에서는 FP 등가가 아니라 비교하지 않는다고 한 줄)
- 혼합 정밀도(HAWQ, HAQ). 차이: 판정 기준의 누수 없는 진단과 "쓸지 말지"의 결정
- OVOD(YOLO-World, YOLOE, OWL-ViT/OWLv2, GLIP, Grounding DINO), OVOD 양자화(QATMA: QAT, 층별 분석 없음)

### 3. 판정 충실도와 누수 없는 진단 (1.0쪽)
- 3.1 지표: LVIS flip(FP가 확신하는 예측의 top-1이 바뀐 비율), lost. COCO = calibration 어휘, LVIS = held-out 어휘. OWLv2는 LVIS로 fine-tuning되지 않은 self-training 체크포인트를 쓴다.
- 3.2 진단 절차: naive W8A8 기준 모델에서 층 하나만 W4로 내리고 calibration 쪽 이미지·COCO 어휘에서 flip 증가를 잰다. val/LVIS는 보지 않는다. 측정 잡음(음수 값들의 RMS)을 함께 보고한다.

### 4. 분석: 손상은 구조를 따른다 (1.5쪽) — 논문의 중심
- **그림 2(핵심): 여섯 모델의 층별 민감도 지도**(`figures/sensitivity_maps.pdf`, 같은 절대 눈금, 잡음 띠 표시).
- **표 A: 집중도와 P의 효과**(§6 참고).
- 4.1 텍스트 게이트 CNN: 융합 conv 하나가 23~31%p. 메커니즘(분기별 activation·weight 통계, 공유 scale 오염) — **그림 3**.
- 4.2 게이트 없는 CNN: 초반 backbone의 split/concat 경계에 작은 봉우리 여러 개(YOLOv8, YOLO11 두 백본에서 같은 위치).
- 4.3 ViT: 3층 하나(4.1%p)를 빼면 잡음(±1.6%p) 안. 손상은 많은 층의 작은 기여가 쌓여 생긴다.
- 4.4 이전은 취약성을 옮긴다(12.cv2 민감도↓, 1.conv 3.5~4배↑) — 왜 M이 G/P와 함께여야 하는지.

### 5. 진단 기반 처방 (1.2쪽)
- 5.1 G: 식과 구조도(**그림 4**). FP 등가, 비트 비용 0. 적용 조건: 텍스트 게이트 융합이 있는 구조.
- 5.2 P: 진단 상위 층을 예산 안에서 W8. 쓸지 말지는 진단이 잡아내는 손상량으로 판단(§7의 규칙 논의).
- 5.3 M: 배포 제약(생산자 공유, residual add)을 지키는 이전. 규칙(M/AWQ/SmoothQuant)은 바꿔 써도 된다. 누수 없는 사전 검사로 모델당 한 번 켤지 정한다.
- 5.4 BRECQ/QDrop 위의 plug-in.

### 6. 실험 (2.3쪽)
- 6.1 설정과 프로토콜(6 seed, 짝 Δ ± 95% CI, Holm). OWLv2 프로토콜의 예외(softmax 출력 16bit, 키 3601개라 8bit 균일 격자로 표현 불가)를 밝힌다.
- 6.2 주 결과(YOLO-World-S, 표 1): v1 표 1 그대로.
- 6.3 여섯 모델 일반화(표 2): 모델마다 진단이 고른 처방과 결과. 회복률(BRECQ가 잃은 LVIS AP 중 되찾은 비율)을 함께.
- 6.4 ablation(표 3): G/P/M 조합, 보호 기준(P/H/R), 이전 규칙(M/S/W).
- 6.5 비용: 크기 +0.3%, 빌드 시간.

### 7. Discussion & Limitations (0.4쪽)
- W4A4 실패(per-tensor A4 절벽), OWLv2 W4A8 절대 성능 낮음(ViT 전용 양자화기 범위 밖), 실제 기기 지연 미측정, M 검사가 놓치는 경우(v2 W8A8).

## 6. 핵심 수치 (10-19 기준, `CVPR2027/runs/`)

**표 A — 진단 집중도와 P의 효과** (`figures/src/diag_concentration.py`)

| 모델 | 구조 | 최민감 층 | 최대 증가 (%p) | 잡음 ±2σ (%p) | 예산 1.5% 포착량 (%p) | P 효과 W4A8 (Δ LVIS) | P 효과 W4A5 |
|---|---|---|---|---|---|---|---|
| YOLO-World-S | CNN, 텍스트 게이트 | 12.cv2 | 30.8 | 0.10 | 41.4 | +1.05 ± 0.20 (6) | +1.37 ± 0.88 (6) |
| YOLO-World-S v2 | CNN, 텍스트 게이트 | 12.cv2 | 24.0 | 0.10 | 35.3 | – | – |
| YOLO-World-M | CNN, 텍스트 게이트 | 12.cv2 | 23.2 | 0.08 | 30.1 | +1.58 ± 0.33 (6) | +1.81 ± 0.35 (6) |
| YOLOE-v8s | CNN, 게이트 없음 | 1 | 6.2 | 0.07 | 21.1 | +0.77 ± 0.27 (6) | +1.13 ± 0.31 (6) |
| YOLOE-11s | CNN(YOLO11), 게이트 없음 | 4.cv2 | 3.2 | 0.23 | 12.8 | +0.16 ± 0.33 (6) | +0.84 ± 0.30 (6) |
| OWLv2-B/16 | ViT, late fusion | 3.k_proj | 4.1 | 1.62 | 4.6 | −0.31 ± 0.20 (3) | – |

읽는 법: 포착량이 클수록 P의 효과가 크다(10-22: m 6 seed, OWLv2 3 seed로 보강). YOLOE-v8s의 P 효과는 GP(G가 아무것도 하지 않아 P와 같다)로 잰 값이다.

**표 2 후보 — 모델별 처방과 결과 (BRECQ 대비 짝 Δ LVIS AP / Δ flip %p)**

| 모델 | 진단이 고른 처방 | W8A8 | W4A8 | W4A5 |
|---|---|---|---|---|
| YOLO-World-S | G + P + M | +0.31 / −0.93 | +0.81 / −5.26 | +2.53 / −8.44 (seed 0 제외) |
| YOLO-World-S v2 | G + P + M | +0.39 / **+2.05** (M이 해침, GP는 −0.24 / −0.65) | +5.82 / −26.81 | +5.64 / −19.36 |
| YOLO-World-M | G + P (M 검사에서 끔) | −0.02 / −1.15 | +1.12 / −6.60 | +2.23 / −10.51 |
| YOLOE-v8s | P + M (G 해당 없음) | −0.02 / −0.54 | +0.74 / −2.41 | +4.40 / −13.55 |
| YOLOE-11s | P + M | +0.11 / −0.67 | +0.73 / −1.32 | +4.56 / −19.25 |
| OWLv2-B/16 | M (P 효과 없음) | +1.07 / −4.00 (3 seed) | +1.66 / −1.70 (PM, 3 seed) | – |

YOLO-World-S v2에서는 PM이 W4에서 GPM보다 LVIS AP가 높다(+6.41 / +7.09). 표 3(ablation)에 함께 싣는다.

## 7. 처방 선택 규칙 — 정해야 할 것

- **G:** 구조 조건(텍스트 게이트 융합이 있는가). 데이터를 보지 않는다. ✅
- **M:** 누수 없는 사전 검사(calibration 밖 COCO flip이 1.5배를 넘으면 끈다). 이미 확정. ✅ (v2 W8A8을 놓치는 한계)
- **P:** 지금은 모든 모델에 켜고 있다. 진단이 잡아내는 손상량으로 켤지 정하는 규칙을 둘지 결정해야 한다.
  - 안 1(권장): **규칙을 두지 않고 관계를 보여준다.** 표 A처럼 "포착량이 클수록 효과가 크다"를 연속적인 관계로 제시하고, OWLv2에서는 효과가 없다고 보고한다. 문턱을 이 여섯 모델을 보고 정하면 사후 튜닝이 되기 때문이다.
  - 안 2: 잡음 대비 기준(예: 잡음 ±2σ의 k배를 넘는 층만 보호 후보)으로 문턱을 정의한다. 데이터 쪽 기준이라 덜 자의적이지만, k 선택이 여전히 남는다.

## 8. 주장과 근거

| 주장 | 근거 | 상태 |
|---|---|---|
| held-out 판정이 AP보다 먼저 무너진다 | 여섯 모델, 모든 비트 설정 | ✅ (OWLv2 W4A8 🟡) |
| 손상 위치는 구조를 따른다 | 진단 지도 6개, 같은 절차 | ✅ |
| G는 비트 비용 0으로 게이트 구조의 손상을 없앤다 | s W4A5 단독 +1.37, m GP W4A5 +2.23 | ✅ |
| P의 효과는 진단이 잡아내는 손상량을 따른다 | 표 A(m 6 seed, OWLv2 3 seed 포함) | ✅ (모델 6개, 경향으로 서술) |
| 이전 규칙은 바꿔 써도 된다 | s 6 seed(M/S/W), OWLv2 3 seed(M/W: W8A8에서 M이 flip 1.8%p 더 낮음) | ✅ s·OWLv2 |
| 엄격한 프로토콜에서도 결론이 유지된다 | attention·contrastive 8bit 6 seed | ✅ |
| 결정적 실행, bit 단위 재현 | 10-16 수정 후 448칸 비교 | ✅ |

## 9. 리뷰어 예상 질문

| 질문 | 답 |
|---|---|
| P, M은 기존 기법 아닌가? | 그렇다. 기여는 기법이 아니라 언제·어디에 쓰는지를 판정 기준의 누수 없는 진단이 정한다는 데 있다. 같은 진단이 구조마다 다른 처방을 고르고(표 A, 그림 2), M 대신 AWQ/SmoothQuant를 써도 결론이 같다(표 3) |
| G는 YOLO-World에만 쓰이지 않나? | 그렇다. YOLO-World 계열(v1/v2, S~X)은 널리 쓰이는 실시간 OVOD다. G는 분석이 찾은 메커니즘(게이트 분기의 공유 scale 오염)에 대한 처방이고, 게이트가 없는 구조에서는 진단이 다른 처방을 고른다는 것이 논지다 |
| OWLv2 W4A8 성능이 낮다 | ViT의 per-tensor W4A8은 원래 어렵고, ViT 전용 양자화기(log2 softmax 등)는 범위 밖이다. 본문은 W8A8을 주로 쓰고 W4A8은 경향만 보고한다 |
| 진단 비용은? | 층 수 × 200장 forward. YOLO s 약 10분, OWLv2 약 1.5시간(L40S). 모델당 한 번 |
| 6개 모델로 "관계"를 말할 수 있나? | 인과가 아니라 일관된 경향으로 쓴다. 각 모델 안에서는 같은 예산의 무작위 보호(R) 대비로 위치 선택의 효과를 따로 보인다(s: R +0.21 vs P +1.06) |

## 10. 남은 일 (11/10 초록 등록, 11/16 제출 기준)

1. ⏳ OWLv2 W4A8 seed 2(10-19 밤 완료 예정). 필요하면 OWLv2 W8A8·W4A8에 GPW 비교.
2. ❌ 처방 선택 규칙에서 P를 어떻게 다룰지 결정(§7).
3. ❌ m의 P 단독 seed 보강(지금 2 seed). 표 A의 CI를 좁힌다. 4줄 × 약 1시간.
4. ❌ 그림 1(동기), 그림 3(메커니즘), 그림 4(G 구조도). 그림 2는 초안 완료.
5. ❌ OWLv2 BRECQ의 COCO AP 하락(box 손상 의심) 확인 — 본문 수치 해석에 필요.
6. ❌ (가능하면) 이전 규칙 비교를 YOLOE-11s W4A5에도.
7. ❌ 엣지 기기에서 "추가 비용 0" 확인.
