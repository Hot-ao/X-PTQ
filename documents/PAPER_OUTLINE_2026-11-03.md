# 논문 개요 (2026-11-03)

**대상 독자:** 공저자. 논문의 이야기 흐름, 절별 내용, 지금 근거가 있는 것과 없는 것을 한 번에 본다.

**이전 판과의 관계**
- [docs/PAPER_OUTLINE.md](../docs/PAPER_OUTLINE.md)(v1, 10-02): 방법(GPM) 중심.
- [docs/PAPER_OUTLINE_v2_diagnosis.md](../docs/PAPER_OUTLINE_v2_diagnosis.md)(v2, 10-19): 진단 중심. 이 판은 v2의 구성을 이어받는다.
- 바뀐 점: 11-02 새 P·M 규칙, 8개 검출기(YOLO-World-L, Grounding DINO 추가), W4A4 열, P 예산 설명.

**관련 문서**
- 방향과 근거: [PAPER_DIRECTION_2026-11-02.md](PAPER_DIRECTION_2026-11-02.md)
- 새 규칙 수치: [docs/RESULTS_NEW_PM_2026-11-02.md](../docs/RESULTS_NEW_PM_2026-11-02.md)
- 최신 결과표: [RESULTS_2026-11-04.md](RESULTS_2026-11-04.md)
- 실시간 표: `results/LEADERBOARD.html` (주요 표 탭)

**수치 기준:** 따로 적지 않으면 **새 규칙, seed 0**(`runs/234_rerun_s0`, `runs/236_fill_s0`)이다. 값은 LVIS minival AP ×100 / LVIS top-1 flip %이고, Δ는 같은 seed BRECQ 대비다. seed 1·2가 들어오면 바뀐다. **11-04: YOLO 6개 모델 W4A8·W4A5의 3 seed 결과는 [RESULTS_2026-11-04.md](RESULTS_2026-11-04.md) §1에 있다.** GDINO flip은 재측정 전이라 쓰지 않는다.

**상태 표시:** ✅ 근거 있음 · 🟡 일부(seed 0만, 또는 일부 모델) · ⏳ 진행 중 · ❌ 아직 없음

---

## 0. 11-02 ~ 11-03에 바뀐 것

| 항목 | 이전 (10-19 v2) | 지금 |
|---|---|---|
| P 선택 | G가 처리하는 C2fAttn cv2를 선택 뒤에 빼고, 빈 자리를 채우지 않음(S 실제 0.33%) | 먼저 빼고 예산을 다시 채움, 분모 = 양자화 weight 전체(S 9개, 1.59%) |
| M xmax | calibration 8장 | calibration 256장 전체 |
| M 사전 검사 | flip_on ≤ 1.5 × flip_off | **ρ = 1.0** (flip_on ≤ flip_off) |
| 표의 "Ours" | 모델당 고정 구성 | seed마다 calibration 밖 flip으로 GP / GPM 선택 |
| 모델 | 6개 | **8개**: YOLO-World-S / S-v2 / M / **L**, YOLOE-v8s / 11s, OWLv2-B/16, **Grounding DINO-T** |
| 비트 | W8A8 ~ W4A5 | **W4A4 열 추가**(S, M) |
| 크기 표기 | "+0.3%" | **+1.6%**(S, 평균 4.06 bit). 모델별 표는 §6.5 |

---

## 1. 한 문장 요약

저비트 PTQ는 open-vocabulary 검출기의 **보지 않은 어휘에 대한 판정을 AP보다 먼저 무너뜨린다.** 그 손상은 **구조가 정하는 소수 지점에 몰려 있어서**(weight 1.5%에 CNN 검출기 손상의 48–72%), calibration 쪽 데이터만 쓰는 누수 없는 진단이 그 지점을 찾는다. 진단 결과로 **구조 재배치(G), 보호(P), 공유 제약 이전(M)** 중 무엇을 쓸지 정한다. 이렇게 고른 처방은 학습 없이 8개 검출기에서 판정을 지키고, 기존 PTQ가 무너지는 W4A4에서도 동작한다.

## 2. 제목 후보

1. *Where Low-Bit Quantization Breaks Open-Vocabulary Detectors: Diagnosis-Guided Post-Training Quantization* (권장, v2와 같음)
2. *Diagnose, Then Quantize: Preserving Held-Out Vocabulary in Low-Bit Open-Vocabulary Detection*

## 3. 초록 (초안, 수치는 seed 0 — seed 1·2 후 다시 맞춘다)

> Open-vocabulary detectors are deployed on edge devices where the vocabulary changes after deployment, yet post-training quantization (PTQ) is tuned and evaluated on the calibration vocabulary. We show that this hides the main failure: at W4A8, BRECQ keeps COCO AP within 3 points of full precision on YOLO-World-S but flips the top-1 class of 17% of confident LVIS predictions. A leak-free, layer-wise diagnosis that uses only calibration images and vocabulary shows that this damage is concentrated and follows the architecture: 1.5% of the weights account for 48–72% of the diagnosed damage in five CNN detectors, located at the text-fusion convolution in text-gated detectors and at early backbone split/concat boundaries in gate-free ones, while a ViT detector spreads it across many layers. The diagnosis selects among three training-free remedies: a bit-free gate commutation that removes the shared-scale contamination of text-gated fusion, 8-bit protection of the diagnosed layers (+1.6% weight memory), and scale migration under deployment constraints, enabled only when a leak-free check approves it. With all convolutions, attention and region–text products quantized, the selected remedies improve held-out LVIS AP over BRECQ on eight detectors (YOLO-World S/S-v2/M/L, YOLOE-v8s/11s, OWLv2, Grounding DINO) by up to +6.3 points at W4A5 where BRECQ stays usable, and keep YOLO-World-S at 18.6 LVIS AP at W4A4 where BRECQ and QDrop collapse below 1. It works as a plug-in on both BRECQ and QDrop.

- "+6.3"은 BRECQ가 무너지지 않은 칸 중 최대(YOLOE-11s W4A5)다. L W4A5 +14.0은 BRECQ seed 0이 무너진 값이라 초록에 쓰지 않는다.
- GDINO는 LVIS 2000장 subset, 새 규칙 PM 값이다(11-04). 초록에서 GDINO는 AP로만 언급한다. **GDINO의 LVIS flip은 다른 모델과 측정 방법이 달라서 재실험한다**(§6.3 각주). 재측정 전에는 GDINO flip을 판정 충실도 주장에 쓰지 않는다.

## 4. 기여

| # | 기여 | 상태 |
|---|---|---|
| C1 | **문제:** 저비트 PTQ는 held-out 어휘 판정을 AP보다 먼저, 더 크게 무너뜨린다. 판정 충실도(LVIS flip / lost)를 AP와 함께 보고해야 한다 | ✅ 8개 모델 |
| C2 | **분석:** 손상은 소수 지점에 몰리고(집중도), 위치는 구조를 따른다(텍스트 융합 conv / 초반 backbone / ViT 분산). 메커니즘: 게이트된 attention 분기의 outlier가 공유 scale을 오염 | ✅ (메커니즘 그림 ❌) |
| C3 | **진단 기반 처방:** G(텍스트 게이트가 있으면, FP 등가·비트 비용 0), P(집중된 곳만 W8, 고정 예산 1.5%), M(누수 없는 검사가 승인하면). BRECQ·QDrop 위 plug-in | 🟡 seed 0 |
| C4 | **엄격한 프로토콜과 재현성:** head 포함, attention·region–텍스트 내적까지 8bit. 결정적 실행 | ✅ |

## 5. 본문 구성 (8쪽)

### 1. Introduction (1.0쪽)
- 엣지에서 어휘가 바뀌는 검출. calibration 어휘의 AP만 보면 W4A8 BRECQ가 쓸 만해 보인다(**그림 1**: 비트별 COCO AP와 LVIS flip).
- 질문 세 개로 이끈다: 무엇이 망가지는가(C1), 어디서 왜(C2), 그래서 무엇을 써야 하는가(C3).

### 2. Related Work (0.6쪽)
- PTQ: AdaRound, BRECQ, QDrop. 검출 PTQ(Reg-PTQ)는 인용만. PD-Quant는 부록에서만 언급.
- outlier와 scale 이전: SmoothQuant, AWQ. CLE(DFQ)는 SiLU에서 FP 등가가 아니라 비교하지 않는다고 한 줄.
- 혼합 정밀도: HAWQ, HAQ, BRECQ의 MP. 차이: 외부 제약 아래 비트 배분이 아니라, 손상 집중 때문에 작은 고정 예산이면 된다는 것(§5.2).
- OVOD: YOLO-World, YOLOE, OWL-ViT/OWLv2, GLIP, Grounding DINO. OVOD 양자화는 거의 없다(QATMA는 학회 논문이 아니므로 비교하지 않는다).

### 3. 판정 충실도와 누수 없는 진단 (1.0쪽)
- 3.1 지표: LVIS flip(FP가 확신하는 예측의 top-1 변화율), lost. COCO = calibration 어휘, LVIS = held-out 어휘. OWLv2는 LVIS로 fine-tuning되지 않은 self-training 체크포인트.
- 3.2 진단: naive W8A8 기준에서 층 하나만 W4로 내리고, calibration 밖 COCO 이미지 200장·COCO 어휘에서 flip 증가를 잰다. val/LVIS는 보지 않는다. 측정 잡음을 함께 보고한다.

### 4. 분석: 손상은 몰려 있고, 구조를 따른다 (1.5쪽) — 논문의 중심
- **그림 2(핵심):** 모델별 층 민감도 지도(`figures/sensitivity_maps.pdf`).
- **표 A: 손상 집중도**(§6.1). 고르게 퍼져 있다면 weight 1.5%는 손상의 1.5%만 담는다. 실제로는 CNN 다섯 개에서 48–72%, OWLv2는 35%.
- 4.1 텍스트 게이트 CNN: 융합 conv 하나가 23–31%p. 메커니즘(**그림 3**).
- 4.2 게이트 없는 CNN: 초반 backbone split/concat 경계(YOLOv8, YOLO11 두 백본에서 같은 위치).
- 4.3 ViT·DETR: 분산. P 효과도 작다.
- 4.4 이전은 취약성을 옮긴다(12.cv2 민감도↓, 1.conv↑). M이 G·P와 함께여야 하는 이유.

### 5. 진단 기반 처방 (1.2쪽)
- 5.1 **G:** 식과 구조도(**그림 4**). FP 등가, 비트 비용 0. 적용 조건은 구조(텍스트 게이트 융합)이고 데이터를 보지 않는다.
- 5.2 **P:** 진단 상위 층을 예산 1.5% 안에서 W8로 둔다(G가 처리하는 층은 빼고 다시 채운다).
  - 예산은 혼합 정밀도처럼 외부 제약에서 정한 값이 아니라, **모든 모델·비트에 같은 고정 배포 비용**(W4 weight의 b%를 W8로 두면 weight 메모리 +b%)으로 쓴다.
  - 작은 예산으로 충분한 근거는 집중도(표 A)다. 민감도(0.5–3%)는 부록.
  - **쓰지 말 것:** "1.5%를 집중도를 보고 정했다"(예산은 09-29에 정했다), "새 혼합 정밀도 기준".
- 5.3 **M:** 배포 제약(생산자 공유, residual add)을 지키는 이전. xmax는 calibration 전체에서 잰다. 규칙(M / AWQ / SmoothQuant)은 바꿔 써도 된다.
  - **사전 검사:** calibration 밖 이미지·COCO 어휘에서 M을 켠 쪽 flip이 끈 쪽보다 크지 않을 때만 켠다(ρ = 1.0). seed마다 한 번.
- 5.4 BRECQ/QDrop 위 plug-in. 빌드 순서와 FP 기준 모델의 구조 정합.

### 6. 실험 (2.3쪽)
- 6.1 설정: 프로토콜(head 포함, 첫·마지막 8bit, attention·contrastive 8bit, head 마지막 conv 입력 A16), calibration train2017 256장, 짝 Δ ± 95% CI.
  - 재구성 iters는 모든 방법 2k(원 논문 20k)이고 학습률은 각 공식값이라고 각주로 밝힌다. BRECQ에서 2k와 20k가 W8A8에서 같다는 근거([docs/BASELINE_STATUS.md](../docs/BASELINE_STATUS.md) §4.1).
  - OWLv2·GDINO 프로토콜의 예외(softmax 16bit 등)를 밝힌다.
- 6.2 **주 결과(표 1, YOLO-World-S):** W8A8 / W4A8 / W4A6 / W4A5 / **W4A4**.
- 6.3 **8개 모델 일반화(표 2):** 모델마다 진단이 고른 처방과 결과.
- 6.4 **ablation(표 3):** G/P/M 조합, 보호 기준(P / H / R), 이전 규칙(M / S / W).
- 6.5 비용: 크기(+1.4–1.6%), 빌드 시간, 진단 비용.
- (부록) P 예산 민감도, xmax 8 vs 256, 규칙 변경 경위, calibration 출처(O365), ODinW-13, GDINO 구현 차이.

### 7. Discussion & Limitations (0.4쪽)
- **W4A4:** 우리 방법은 동작하지만 FP 대비 −7점이다. per-tensor A4가 여전히 절벽이다.
- **M 검사가 놓치는 경우:** S W4A8(GP가 0.60 높음), v2 W8A8(flip 4.79 → 8.09).
- **v2에서 G 위 새 규칙이 PM보다 나쁨**(§8 미결).
- OWLv2 W4A8 절대 성능이 낮다(ViT 전용 양자화기는 범위 밖). 실기기 지연은 미측정.

---

## 6. 핵심 수치

### 6.1 표 A — 손상 집중도 (예산 안에 들어가는 진단 손상 비중, 실제 P 규칙)

| 모델 | 구조 | 최민감 층 | 1.5% 예산이 담는 손상 | 실제 보호 |
|---|---|---|---|---|
| YOLO-World-S | CNN, 텍스트 게이트 | 12.cv2 | 72% | 9개, 1.59% |
| YOLO-World-S v2 | CNN, 텍스트 게이트 | 12.cv2 | 53% | 8개, 1.36% |
| YOLO-World-M | CNN, 텍스트 게이트 | 12.cv2 | 55% | 8개, 1.48% |
| YOLOE-v8s | CNN, 게이트 없음 | 1 | 67% | 9개, 1.42% |
| YOLOE-11s | CNN(YOLO11), 게이트 없음 | 4.cv2 | 48% | 6개, 1.42% |
| OWLv2-B/16 | ViT, late fusion | 3.k_proj | 35% | 2개, 1.35% |

- 고르게 퍼져 있다면 1.5%. 즉 CNN에서 30–50배 몰려 있다.
- L·GDINO의 집중도는 아직 계산하지 않았다(❌).

### 6.2 표 1 후보 — YOLO-World-S (seed 0)

| 방법 | W8A8 | W4A8 | W4A6 | W4A5 | W4A4 |
|---|---|---|---|---|---|
| FP32 | 25.89 | | | | |
| RTN | 25.44 / 4.97 | 1.30 / 85.2 | 0.57 / 92.4 | 0.12 / 99.3 | 0.00 / 99.8 |
| BRECQ | 25.54 / 3.82 | 22.75 / 17.49 | 21.33 / 20.65 | 11.29 / 62.27 † | 0.45 / 97.21 |
| QDrop | 25.37 / 3.92 | 22.63 / 18.55 | 20.10 / 30.09 | 5.74 / 81.25 | 0.27 / 97.93 |
| BRECQ + SmoothQuant | 25.54 / 4.06 | 22.16 / 19.98 | 21.47 / 22.15 | 19.14 / 31.76 | 9.25 / 66.05 |
| BRECQ + AWQ | 25.53 / 3.42 | 21.80 / 19.80 | 20.38 / 26.92 | 18.95 / 32.54 | 0.33 / 98.07 |
| **Ours (BRECQ)** | **25.78 / 2.71** (GPM) | **23.97 / 10.69** (GPM) | **23.66 / 13.67** (GP) | **22.94 / 16.20** (GPM) | **18.60 / 32.20** (GPM) |
| Ours (QDrop) | 25.80 / 2.69 | 24.40 / 11.12 | 23.17 / 14.03 | 22.39 / 17.50 | 12.71 / 46.16 |

- † BRECQ seed 0이 W4A5에서 무너진다. 이 칸은 seed 1·2로 읽는다(새 GPM Δ +3.10, +3.28. 새 GPM 3 seed 평균 22.98).
- AdaRound는 W8A8 25.34, W4A8 7.79만 있다. 표에 넣을지 정해야 한다.
- W4A4는 BRECQ·QDrop이 모두 1 미만이다. 이전 권고대로 **표 1에 W4A4 열 하나**, 나머지 모델의 W4A4(M: GPM 15.59 vs BRECQ 0.54)는 부록.
- 크기 열: W4 6.23 MB → +P 6.33 MB(S).

### 6.3 표 2 후보 — 8개 모델 (seed 0, BRECQ 대비 Δ LVIS AP / Δ flip %p)

| 모델 | 처방 (선택) | W8A8 | W4A8 | W4A5 |
|---|---|---|---|---|
| YOLO-World-S | G+P+M | +0.24 / −1.11 | +1.22 / −6.80 | † (s1·s2 +3.10, +3.28) |
| YOLO-World-S v2 | G+P+M | +0.07 / **+3.30** | +3.46 / −18.49 | +3.89 / −16.67 |
| YOLO-World-M | G+P (M 검사에서 끔) | −0.03 / −1.40 | +1.32 / −8.21 | +4.21 / −14.86 |
| YOLO-World-L | G+P+M | ⏳ | +3.64 / −12.47 | +14.00 / −36.71 † |
| YOLOE-v8s | P (+M) | +0.07 / −0.49 | +0.86 / −3.73 (P) | +6.01 / −17.44 |
| YOLOE-11s | P (+M) | +0.19 / −0.78 | +0.28 / −2.05 (P) | +6.32 / −23.30 |
| OWLv2-B/16 | M / P+M | +1.67 / −4.70 (M) | +2.06 / −2.32 (PM) | – |
| Grounding DINO-T | P+M | – | – | +2.97 / flip 재측정 ⏳ (LVIS 2000장, 새 규칙) |

- QDrop 열: S, L만 있다. v2·m·YOLOE 두 모델은 `runs/236` 대기열에 있다.
- L W4A5는 BRECQ 15.37, QDrop 1.39로 기준선이 약하다. seed가 모일 때까지 Δ를 강조하지 않는다.
- **GDINO flip은 재측정한다.** 지금까지의 GDINO LVIS flip은 FP가 확신하는 query에서 **어휘 chunk 안의** top-1이 바뀐 비율이었다. 다른 모델(YOLO, YOLOE, OWLv2)은 1203 어휘 전체의 top-1로 잰다. 정의가 달라서 다른 모델과 비교할 수 없다. 기존 값(PM 25.0% vs BRECQ 24.1%)은 쓰지 않는다. 1203 어휘 전체 top-1 기준 flip(`run_gdino.py`의 `lvis_gtflip`)으로 다시 잰다. AP는 영향이 없다.
- 기존 6 seed(구 규칙)와 비교: S, M, YOLOE는 같은 양상이다. **v2는 W4에서 이득이 줄었다**(구 +5.82 → +3.46, W4A8). §8 참고.

### 6.4 표 3 후보 — S ablation (seed 0, Δ vs BRECQ)

| 조건 | W4A8 | W4A5 | W4A4 |
|---|---|---|---|
| +G | ⏳ | +9.91 | +1.79 |
| +P | +0.96 | +3.01 | +0.36 |
| +M | ⏳ | +9.37 | +8.60 |
| +GP | +1.82 | +11.04 | +2.09 |
| +GM | ⏳ | +10.19 | +16.74 |
| +PM | +1.07 | +10.82 | +16.14 |
| +GPM | +1.22 | +11.65 | +18.15 |
| +R (무작위, 같은 예산) | +0.25 | ⏳ | – |
| +H (HAWQ식) | +0.87 | +3.18 | – |
| +GPS / +GPW (이전 규칙 교체) | +0.94 / +1.22 | +10.64 / +10.61 | – |

- W4A5 Δ가 큰 것은 BRECQ seed 0 붕괴 때문이다. 조건 사이 차이로 읽는다: GPM − GP = +0.61, GP − P = +8.03.
- W4A4에서는 M이 있어야 열린다(GP 2.54 vs GPM 18.60).
- 예산 민감도(부록): S W4A5, xmax 8장, 3 seed. GPM 0.5% 22.55 / 1.5% 22.79 / 3% 22.93, GP는 늘지 않음.

### 6.5 크기 (새 P 규칙)

| 모델 | 보호 비율 | W4 MB | W4+P MB | 평균 bit |
|---|---|---|---|---|
| YOLO-World-S | 1.59% | 6.23 | 6.33 | 4.06 |
| YOLO-World-S v2 | 1.36% | 6.11 | 6.20 | 4.05 |
| YOLO-World-M | 1.48% | 13.96 | 14.17 | 4.06 |
| YOLOE-v8s | 1.42% | 5.55 | 5.63 | 4.06 |
| YOLOE-11s | 1.42% | 4.69 | 4.76 | 4.06 |
| OWLv2-B/16 | 1.35% | 43.65 | 44.24 | 4.05 |

---

## 7. 주장과 근거

| # | 주장 | 근거 | 상태 |
|---|---|---|---|
| R1 | held-out 판정이 AP보다 먼저 무너진다 | 8개 모델, 모든 W4 설정 | ✅ |
| R2 | 손상은 소수 지점에 몰리고 위치는 구조를 따른다 | 표 A, 진단 지도 | ✅ (L, GDINO ❌) |
| R3 | 집중도가 P의 효과를 예측한다 | 모델 6개의 경향 | 🟡 새 규칙으로 P 효과 재계산 필요 |
| R4 | G는 비트 비용 0으로 게이트 손상을 없앤다 | S W4A5 단일 요소 중 최대 | ✅ |
| R5 | M은 받쳐 주는 요소가 있을 때 이득이다 | W4A5 GPM − GP +0.61, W4A4 GP 2.54 → GPM 18.60 | ✅ |
| R6 | P 예산은 M과 함께일 때 성능으로 이어진다 | 예산 sweep 3 seed | 🟡 |
| R7 | 학습 없는 plug-in이다(BRECQ, QDrop) | S 5개 비트 모두 QDrop + GPM > QDrop | ✅ seed 0 |
| R8 | DETR·ViT 계열에서도 처방 선택이 맞게 작동한다 | OWLv2 W8A8 M +1.67, GDINO PM +2.97 (AP만, flip 재측정 중) | 🟡 |
| R9 | 기존 PTQ가 무너지는 W4A4에서 동작한다 | S 18.60 vs BRECQ 0.45 / QDrop 0.27, M 15.59 vs 0.54 | 🟡 seed 0 |

## 8. 미결 사항 (교수님과 정할 것)

1. **v2에서 새 규칙 GPM이 PM보다 나쁘다.** seed 0에서 W4A8 GPM 19.44 vs PM 22.45, W4A5 17.19 vs 19.72.
   - `runs/235`: P를 다시 채우지 않아도 18.88로 같다. P를 다시 채우지 않고 xmax를 8장으로 되돌리면 21.69가 재현된다. 원인은 **v2에서 G와 함께 쓰는 xmax 256장 통계**로 보인다.
   - 선택지: (a) 그대로 보고(진단 선택의 한계), (b) v2 표에 PM을 함께 싣기, (c) M 검사를 GP/GPM/PM 셋 중 고르는 규칙으로 넓히기(규칙 변경이라 전체 재선택 필요).
2. **S W4A8 M 검사가 놓친다**(GP 24.57 > GPM 23.97). GP와 GPM을 둘 다 실을지, 각주로 둘지.
3. **seed 수:** S 주 결과 3 seed 또는 6 seed, 나머지 모델 3 seed.
4. **GDINO 평가 규모:** LVIS 2000장 / 전체, seed 수. flip은 측정 방법을 맞춰 재실험한다(§6.3 각주). 재실험할 때 규모도 함께 정한다.
5. **AdaRound 행:** W4A6 이하가 비어 있다. 채울지(약 1.5h × 3칸), 표에서 뺄지.

## 9. 리뷰어 예상 질문

| 질문 | 답 |
|---|---|
| P는 혼합 정밀도인데 예산은 왜 1.5%인가? | 외부 제약에서 정한 값이 아니라 모든 모델에 같은 고정 비용(+1.6%)이다. 손상이 몰려 있어서 충분하다: 고르게 퍼져 있다면 1.5%만 담겠지만 CNN에서 48–72%를 담는다. 0.5–3%에서 결론이 같다(부록) |
| P, M은 기존 기법 아닌가? | 그렇다. 기여는 어디에·언제 쓰는지를 판정 기준의 누수 없는 진단이 정한다는 데 있다. 같은 예산의 무작위 보호는 +0.25, 진단 기반은 +0.96(W4A8). M 대신 AWQ/SmoothQuant를 써도 비슷하다 |
| QDrop이 원 논문과 달리 BRECQ보다 나쁘다 | 모든 방법을 2k iters로 맞추고 학습률은 공식값을 썼다(각주). 주 비교는 가장 강한 기준선인 BRECQ와의 짝 비교이고, QDrop 위에서도 GPM이 같은 이득을 낸다 |
| W4A4는? | 기존 PTQ는 1 미만으로 무너지고, 우리 방법은 18.6(S)이다. FP와의 차이(−7)는 한계로 적는다 |
| 진단 비용은? | 층 수 × 200장 forward. YOLO s 약 10분, OWLv2 약 1.5시간. 모델당 한 번 |
| M을 seed마다 고르는 건 튜닝 아닌가? | 평가를 보지 않는 검사(calibration 밖 이미지, COCO 어휘)다. 다만 xmax 256장 선택 때 LVIS를 본 것은 사실이므로, "calibration 이미지 전체를 쓰는 것이 기본"으로 설명한다 |

## 10. 일정 (초록 11-10, 본문 11-16, 부록 11-23 AoE)

| 날짜 | 할 일 |
|---|---|
| 11-03 | 234 마무리(GDINO PM), 236 빈칸 채우기(QDrop 열, ablation W4A8, R, SW) |
| 11-03 ~ 11-06 | seed 1·2. §8의 1번(v2) 결정 후 시작 |
| 11-04 ~ 11-09 | 본문 초안(4절 분석, 6절 실험 먼저), 그림 1·3·4. L·GDINO 집중도 계산 |
| 11-10 | 초록 등록(진단 중심 제목·초록) |
| 11-10 ~ 11-16 | 본문 완성, 짝 통계 재계산, 부록 |
