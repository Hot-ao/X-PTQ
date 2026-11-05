# 논문 구성안: 교수님 제안 짜임새에 맞춘 절별 내용 (2026-11-05)

**대상 독자:** 공저자. 교수님이 제안하신 섹션 구성에 지금까지의 내용(논문 방향, 개요, 결과)을 배치한다. 집필할 때 절마다 무엇을 쓰고, 어떤 그림·표를 쓰는지 보는 문서다.

**바탕 문서**
- 이야기와 주장: [PAPER_DIRECTION_2026-11-02.md](PAPER_DIRECTION_2026-11-02.md), [PAPER_OUTLINE_2026-11-03.md](PAPER_OUTLINE_2026-11-03.md)
- 수치: [RESULTS_2026-11-04.md](RESULTS_2026-11-04.md) (3 seed 메인 표), [GENERALIZATION_2026-11-01.md](GENERALIZATION_2026-11-01.md)
- 방법 그림 배치: [FIGURE_METHOD_PLACEMENT_2026-11-04.md](FIGURE_METHOD_PLACEMENT_2026-11-04.md)
- 시나리오(식): [METHOD_SCENARIO_2026-11-01.md](METHOD_SCENARIO_2026-11-01.md)

**수치 기준:** 따로 적지 않으면 [RESULTS_2026-11-04.md](RESULTS_2026-11-04.md) §1의 새 규칙, 3 seed 평균이다. 값은 LVIS AP / LVIS flip(%)이다. 제출 전에 최종 표로 다시 맞춘다.

---

## 교수님 제안 구성과 분량 (CVPR 본문 8쪽)

| 절 | 분량 | 핵심 메시지 | 그림·표 |
|---|---|---|---|
| Abstract | – | 문제 → 원인 → 방법 → 결과를 5~6문장으로 | – |
| 1 Introduction | 1.0 | 5단락, 단락마다 주제 하나, 두괄식 | **Fig. 1** (teaser) |
| 2 Related Work | 0.6 | 기존 연구와 우리의 차이 | – |
| 3 Motivation | 1.0 | 기존 PTQ의 한계를 **그림으로** 보여 준다 | **Fig. 2** (한계 관찰) |
| 4 Method | 1.8 | 4.0 개요(이름, 진단, 전체 그림) → 4.1 G → 4.2 P → 4.3 M | **Fig. 3** (파이프라인), Fig. 4 (G 상세, 선택) |
| 5 Experiments | 3.2 | 5.1 데이터셋 → 5.2 평가 → 5.3 주 결과(정량·정성) → 5.4 ablation → 5.5 discussion | **Table 1~3**, **Fig. 5** (정성) |
| 6 Conclusion | 0.3 | 요약, 한계, 향후 과제 | – |

### 이전 개요(11-03)에서 바뀌는 점
- 11-03 개요는 **분석 절(4절, 1.5쪽)**이 중심이었다. 교수님 구성에서는 이 내용이 **3 Motivation(1.0쪽)**으로 줄어들고, 분량이 Method와 Experiments로 간다.
- 그래도 논문의 중심 메시지("진단이 처방을 고른다")는 유지한다. Motivation의 관찰 하나하나가 처방 하나로 이어지게 쓰고(3절), Method 개요에서 진단을 먼저 소개한다(4.0).
- 지표·진단 절(11-03의 3절)은 나뉜다. flip 정의는 **3 Motivation**으로, 진단 절차는 **4 Method 개요**로 간다.

---

## Abstract

다섯 문장 구조를 권한다(초안 문장은 [PAPER_OUTLINE_2026-11-03.md](PAPER_OUTLINE_2026-11-03.md) §3).

| 순서 | 내용 | 근거 수치 |
|---|---|---|
| ① 배경 | OVOD는 배포 후 어휘가 바뀌는 엣지 환경에서 쓰이고, 저비트 PTQ가 필요하다 | – |
| ② 문제 | 기존 PTQ는 calibration 어휘의 AP만 보면 쓸 만해 보이지만, 처음 보는 어휘의 판정을 먼저 무너뜨린다 | S W4A8 BRECQ: LVIS flip 17.7% |
| ③ 원인 | 누수 없는 진단으로 보면 손상은 구조가 정하는 소수 지점에 몰린다 | weight 1.5%에 CNN 손상의 48–72% |
| ④ 방법 | 진단에 따라 G(게이트 교환), P(보호), M(공유 제약 이전 + 사전 검사)을 고르는 학습 없는 plug-in | +1.6% 크기 |
| ⑤ 결과 | 8개 검출기, BRECQ·QDrop 위에서 개선. 기존 PTQ가 무너지는 W4A4에서도 동작 | L W4A8 +4.13, S W4A4 0.43 → 18.60 |

---

## 1. Introduction (5단락)

**작성 원칙** (교수님 제안)
- 한 단락에 주제 하나만 쓴다.
- 두괄식: 단락 첫 문장이 그 단락의 주장이다.
- 주요 단락(②~④)은 마지막 문장에서 주장을 한 번 더 말한다. 앞 문장과 같은 말을 반복하지 말고, 단락의 근거를 거친 뒤의 결론으로 다시 쓴다.

| 단락 | 첫 문장 (주장) | 내용 | 마지막 문장 (재주장) | 근거·그림 |
|---|---|---|---|---|
| ① 배경 | OVOD는 배포 후 어휘가 바뀌는 엣지 기기에서 가장 필요하다 | 로봇·드론·카메라에서 재학습 없이 대상을 바꾼다. 기기 안에서 돌리려면 4bit PTQ가 필요하다 | 그런데 양자화 시점에는 배포 후 어휘를 알 수 없다 | 시나리오 1절 |
| ② 문제 | 기존 PTQ는 calibration 어휘의 AP를 지키는 동안 처음 보는 어휘의 판정을 무너뜨린다 | BRECQ·QDrop은 출력 오차를 줄인다. AP는 몇 점 떨어지는데 확신 예측의 6분의 1이 다른 클래스로 바뀐다. 그래서 AP와 함께 판정 flip을 봐야 한다 | AP만으로는 OVOD 양자화의 실패가 보이지 않는다 | **Fig. 1**: 비트별 COCO AP vs LVIS flip (`figures/teaser_flip.pdf`) |
| ③ 원인 | 이 손상은 모델 전체에 퍼져 있지 않고, 구조가 정하는 소수 지점에 몰려 있다 | 누수 없는 층별 진단. 텍스트 게이트 융합 conv / 초반 backbone / ViT는 분산. weight 1.5%가 손상의 48–72% | 따라서 어디가 무너지는지 알면 그곳만 고치면 된다 | 표 A 수치를 문장으로 |
| ④ 방법 | 우리는 진단이 처방을 고르는 학습 없는 PTQ를 제안한다 | G(FP 등가, 비트 0), P(1.5% W8), M(배포 가능한 이전 + 누수 없는 검사). BRECQ·QDrop 위 plug-in, 표준 정수 커널 | 모델마다 진단이 다른 처방을 고르고, 같은 규칙이 모든 모델에 적용된다 | Fig. 3 참조 |
| ⑤ 결과·기여 | 8개 검출기에서 판정을 지키고, 기존 PTQ가 무너지는 W4A4에서도 동작한다 | 핵심 수치 2~3개 + **기여 목록(3~4개 bullet)** | – (기여 목록으로 끝냄) | Table 1·2 |

**기여 목록 (⑤ 끝)**
1. 저비트 OVOD의 숨은 실패를 판정 flip으로 드러낸다.
2. 손상이 소수 지점에 몰리고 위치가 구조를 따른다는 분석과, 그것을 찾는 누수 없는 진단.
3. 진단 기반 처방 G·P·M. 학습 없이 BRECQ·QDrop 위에 얹고 표준 정수 커널로 배포된다.
4. 엄격한 프로토콜(attention·region–텍스트 내적까지 양자화)로 8개 검출기와 W4A4까지 평가한다.

---

## 2. Related Work (0.6쪽)

| 소절 | 다룰 연구 | 우리와의 차이 (한 줄) |
|---|---|---|
| OVOD | YOLO-World, YOLOE, OWL-ViT/OWLv2, GLIP, Grounding DINO | 이 모델들을 저비트로 내리는 연구가 거의 없다 |
| PTQ | AdaRound, BRECQ, QDrop (검출 PTQ: Reg-PTQ, InlierQ) | 출력 오차를 줄이고 어휘를 고려하지 않는다. 우리는 그 위에 얹는다 |
| Outlier·scale 이전 | SmoothQuant, AWQ, OS+, OmniQuant | 통계 방식은 따르되(SmoothQuant), 배포 공유 제약과 사전 검사를 더한다 |
| 혼합 정밀도 | HAWQ-V2/V3, HAQ, BRECQ의 MP, OMPQ | 외부 제약 아래 비트 배분이 아니라, 손상 집중 때문에 작은 고정 예산(1.5%)으로 충분하다 |
| OVOD 양자화 | QATMA(재학습) | 우리는 학습 없는 PTQ다. 수치 비교는 중심에 두지 않는다 |

---

## 3. Motivation: 기존 기법의 한계 (1.0쪽)

**작성 방향:** 관찰 세 개를 **Fig. 2의 패널 세 개**로 보여 준다. 관찰 하나가 처방 하나로 이어지게 쓴다. 각 관찰의 마지막 문장은 "→ 그래서 X가 필요하다"로 끝낸다.

**3.1 판정 flip 정의** (짧게, 식 1개): FP가 확신하는 위치에서 top-1 프롬프트가 바뀐 비율. 역전 조건($\delta_k-\delta_{k^*}>m_k$)과 "큰 어휘일수록 margin이 작아 쉽게 뒤집힌다"를 한 문단으로.

| 관찰 | 기존 기법의 한계 | Fig. 2 패널 | 이어지는 처방 |
|---|---|---|---|
| **관찰 1: 손상은 소수 지점에 몰린다** | 기존 PTQ는 모든 층을 같은 비트로 다룬다. 실제로는 weight 1.5%에 손상의 48–72%가 몰려 있다 | (a) 층별 민감도 지도 (`figures/sensitivity_maps.pdf`) 또는 집중도 막대 | **P**: 몰린 곳만 W8로 |
| **관찰 2: 텍스트 게이트 분기가 공유 scale을 오염시킨다** | YOLO-World는 텍스트 분기를 다른 분기와 이어 붙여 conv 하나로 합친다. 텍스트 분기 값이 4배 커서 scale을 차지한다. 이 conv 하나만 W4로 내려도 LVIS flip 3.6 → 68.8% | (b) C2fAttn 분기별 크기 통계 + 이 conv만 W4일 때의 flip | **G**: 게이트를 conv 뒤로 |
| **관찰 3: activation 채널 편차가 저비트 activation을 막는다** | per-tensor activation 때문에 A4~A5에서 무너진다. 기존 scale 이전(SmoothQuant, AWQ)은 배포 공유 제약을 고려하지 않고, 단독으로 쓰면 오히려 나빠지기도 한다(M 단독 W4A5 −0.75, YOLO-World-M에서 M이 해로움) | (c) 비트별 BRECQ vs +이전, 또는 W4A4에서 M 유무 | **M**: 공유 제약 + 사전 검사 |

- Fig. 1(teaser)은 "문제"(AP vs flip)를, Fig. 2는 "원인"(왜 기존 기법이 실패하나)을 맡는다. 둘이 겹치지 않게 한다.
- **준비 상태:** (a) 그림 있음. (b) 수치 있음, 그림 필요(11-03 개요의 "그림 3"). (c) 수치 있음, 그림 필요.

---

## 4. Method (1.8쪽)

### 4.0 개요: 이름, 진단, 전체 그림 (0.5쪽)

**① 이름 정하기 (교수님 제안).** 아래는 후보이고, 기존 기법과 이름이 겹치지 않는지 확인해야 한다.

| 대상 | 지금 부르는 이름 | 후보 |
|---|---|---|
| 전체 방법 | GPM, Ours | **DGQ** (Diagnosis-Guided Quantization) / **OVQ** (Open-Vocabulary Quantization) |
| D 진단 | 누수 없는 진단 | Leak-free Flip Diagnosis |
| G | 게이트 교환 | **Gate Commutation (GC)** |
| P | 보호 | **Diagnosis-guided Protection (DP)** |
| M | 공유 제약 scale 이전 | **Tied Scale Migration** 또는 **Deployable Scale Migration** + Migration Check |

- 약어를 정하면 본문, 표, 그림(현재 D/G/P/M/R 배지)을 한 번에 바꾼다.
- 표의 "+Ours"도 정한 이름으로 바꾼다(예: "+DGQ").

**② 진단(D)을 개요에서 먼저 소개한다.** naive W8A8에서 층 하나만 W4로 내리고, calibration 쪽 이미지와 COCO 어휘로 flip 증가를 잰다. 진단 결과가 G·P·M 적용 위치를 정한다. val·LVIS는 보지 않는다.

**③ 전체 그림 (Fig. 3, 교수님 제안).** input → OVOD → output 파이프라인 위에 각 기법이 붙는 모양으로 그린다.

```
 [Text prompts] ─ Text Encoder (FP) ───────────────┐
                                                    ▼
 [Image] → Backbone ─────────→ Vision-Language PAN ──────→ Head → [Boxes + classes]
           ▲ P: 앞쪽 conv W8     ▲ G: T-CSPLayer 4곳의
           (진단 상위 1.5%)        cv2를 게이트 뒤로
           ◀───── M: conv → conv 연결마다 공유 s (생산 ÷s, 소비 ×s) ─────▶
                         R: BRECQ / QDrop (전체, 바꾸지 않음)
 (모델 밖, 점선) D: 누수 없는 진단 → G·P·M 위치 결정 / M 사전 검사 → 켬·끔
```

- 모듈 대응과 지킬 점은 [FIGURE_METHOD_PLACEMENT_2026-11-04.md](FIGURE_METHOD_PLACEMENT_2026-11-04.md)에 있다(G는 T-CSPLayer에만, P는 백본 앞쪽에만, D는 모델 밖).
- 지금 그림(흐름도 형태, D → G·P → M check)은 "절차"를 보여 준다. 교수님 제안대로 바꾸면 "기법이 모델의 어디에 붙는지"가 먼저 보인다. 절차(빌드 순서)는 그림 아래쪽 작은 띠로 남기는 것을 권한다.

### 4.1 G: Gate Commutation (0.4쪽)
- 식: 등가식과 양자화 후 비교식(시나리오 5.1).
- 적용 조건: 텍스트 게이트 융합이 있는 구조. 데이터를 보지 않고 구조로 정한다.
- 성질: FP 등가, 비트 비용 0, 연산 형태 동일.
- 근거(짧게): 교환 전후 비교표(시나리오 5.1, **확정 프로토콜로 재측정 필요**). 자세한 구조는 Fig. 4(선택) 또는 Fig. 3 확대 패널.

### 4.2 P: Diagnosis-guided Protection (0.4쪽)
- 식: $\Delta_\ell$ 정의와 예산 1.5% 안 greedy.
- G와의 관계: G가 맡는 conv는 빼고 예산을 다시 채운다.
- 예산 설명: 외부 제약이 아니라 모든 모델에 같은 고정 비용(+1.6%, 평균 4.06bit). 손상이 몰려 있어서 작은 예산으로 충분하다(관찰 1). 예산 스윕은 ablation(5.4)에.
- 쓰지 않을 말: "새 혼합 정밀도 기준", "1.5%를 집중도를 보고 정했다".

### 4.3 M: Tied Scale Migration + Check (0.5쪽)
- 배경 2~3줄: activation per-tensor 제약 → weight로 이전 → 생산 층이 $x/s$를 만들어 공유해야 한다(시나리오 5.3 ①~③).
- 식: $s$ 공식, 공유 제약, 하한, α 선택(이전 없음 포함).
- 사전 검사: calibration 밖 이미지·COCO 어휘에서 $\text{flip}_{\text{on}}\le\text{flip}_{\text{off}}$일 때만 켠다(ρ = 1.0).
- 선행 연구 대응 한 줄: 통계는 SmoothQuant처럼 calibration 전체, α는 AWQ처럼 출력 오차 탐색.

(BRECQ·QDrop 위 plug-in 순서는 4 개요 끝이나 4.3 끝에 2~3줄.)

---

## 5. Experiments (3.2쪽)

### 5.1 Datasets & Models
- calibration: COCO train2017 256장, COCO 80 어휘. 진단·검사 이미지 구분(1–200 진단, 257–456 검사).
- 평가: COCO val2017(calibration 어휘), **LVIS minival 1203 어휘(처음 보는 어휘, 주 지표)**, ODinW-13(도메인 밖).
- 모델 8개: YOLO-World S/S-v2/M/L, YOLOE-v8s/11s, OWLv2-B/16, Grounding DINO-T.

### 5.2 Evaluation & Implementation
- 지표: LVIS AP, APr, LVIS flip, lost. 짝 Δ ± 95% CI, seed 수 명시.
- 프로토콜: head 포함, 첫·마지막 8bit, attention·region–텍스트 내적 8bit, head 마지막 1×1 입력 A16. OWLv2·GDINO 예외(softmax 16bit 등).
- 기준선: RTN, AdaRound, BRECQ, QDrop, BRECQ+SmoothQuant, BRECQ+AWQ. 재구성 2k iters 각주.

### 5.3 Main Results
**정량적**
- **Table 1 (YOLO-World-S, 비교 방법 전체 × W8A8/W4A8/W4A6/W4A5/W4A4)**: 기존 PTQ 대비. 결과 문서 §2.
- **Table 2 (모델 일반화, 메인 표)**: BRECQ / +Ours / QDrop / +Ours × W4A8 / W4A5 (+W4A4). 칸은 "LVIS AP (flip)". 모델 이름 아래 진단이 고른 구성(G+P+M 등). 결과 문서 §1 (3 seed).
  - W8A8과 APr은 부록으로 보내 표 폭을 줄인다(앞선 표 검토 의견).
  - OWLv2, GDINO는 같은 표의 열 또는 별도 소표.
- 일반화 요약(본문 2~3줄 + 부록): ODinW-13, calibration 출처(Objects365), W4A4 첫·마지막 FP.

**정성적**
- **Fig. 5**: 같은 이미지에서 FP / BRECQ / Ours의 검출 결과. 처음 보는(LVIS) 클래스가 BRECQ에서 다른 클래스로 바뀌고 Ours에서 유지되는 예. `figures/qual/`(W4A8·W4A5·W4A4 wins)에서 고른다.
- 고를 때 주의: 이긴 사례만 고르면 편향으로 보인다. 부록에 무작위 사례나 실패 사례도 함께 둔다.

### 5.4 Ablation
| 항목 | 질문 | 근거 |
|---|---|---|
| G/P/M 조합 | 세 요소가 각각 기여하는가 | S, W4A8·W4A5·W4A4 (결과 문서 §4) |
| 보호 기준 (P / H / R) | 위치 선택이 중요한가 | 같은 예산 무작위 +0.25 vs P +0.96 (W4A8) |
| 이전 규칙 (M / SmoothQuant / AWQ) | 이전 규칙을 바꿔도 되는가 | GPS, GPW |
| P 예산 (0~5%) | 1.5%가 적절한가 | 예산 스윕 그림(`figures/src/budget_sweep.py`) |
| M 통계·검사 | xmax 8 vs 256, ρ | 부록 |
| 기반 PTQ | BRECQ와 QDrop 모두에서 되는가 | Table 2의 QDrop 열 |

### 5.5 Discussion: Findings & Observations
- **W4A4는 M이 있어야 열린다.** GP 0.08–5.03 vs GPM 10.86–22.60.
- **진단이 고른 처방이 모델마다 다르다.** YOLOE는 G 없음, M은 W4A5에서 M 끔.
- **M 사전 검사의 정확도와 놓치는 경우**(S W4A8, v2 W8A8). 결과 문서 §6.
- **W8A8에서는 AP 이득이 작고 flip 감소가 주 효과다**(v2는 예외).
- **plug-in:** QDrop이 약한 칸일수록 Ours의 이득이 크다.
- 비용: 크기 +1.4–1.6%, 진단·검사 시간.

---

## 6. Conclusion (0.3쪽)
- 요약 3문장: 문제(판정 flip) → 원인(집중·구조) → 방법·결과.
- 한계 2~3개: W4A4에서 FP와의 차이, M 검사가 놓치는 경우, 실기기 지연 미측정.
- 향후 과제 1~2개: 다른 OVOD 구조로의 확장, 실기기 배포.

---

## 정해야 할 것

1. **방법과 기법 이름** (4.0 개요 ①).
2. **분석의 비중**: 교수님 구성에서는 Motivation 1쪽으로 줄어든다. 11-03 개요의 분석 내용(집중도 표, 구조별 위치, 메커니즘) 중 무엇을 본문 Motivation에, 무엇을 부록에 둘지.
3. **Fig. 1과 Fig. 2의 역할 분담**: Fig. 1 = 문제(AP vs flip), Fig. 2 = 기존 기법의 한계(관찰 3개).
4. **Table 2의 범위**: W8A8 포함 여부, OWLv2·GDINO를 같은 표에 둘지.
5. **정성 결과 선정 기준**.

## 그림·표 준비 상태

| 번호 | 내용 | 상태 |
|---|---|---|
| Fig. 1 | 비트별 COCO AP vs LVIS flip | ✅ `figures/teaser_flip.pdf` |
| Fig. 2 | 기존 기법의 한계 관찰 3개 | (a) ✅ 민감도 지도, (b)(c) ❌ |
| Fig. 3 | 파이프라인 위 기법 배치 | 🟡 흐름도 판 있음, 파이프라인 형태로 다시 그리기 |
| Fig. 4 | G 상세 (선택) | 🟡 Fig. 3 안 패널로 대체 가능 |
| Fig. 5 | 정성 결과 | 🟡 `figures/qual/` 후보 있음 |
| Table 1 | S × 비교 방법 × 비트 | 🟡 seed 0 |
| Table 2 | 모델 일반화 메인 표 | 🟡 YOLO 6개 3 seed, OWLv2·GDINO 정리 필요 |
| Table 3 | ablation | 🟡 일부 seed 0, 빈칸 있음 |
| 부록 | 예산 스윕, ODinW, calibration 출처, W8A8, APr | 🟡 |
