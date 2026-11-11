# 5. Experiments 글머리 (2026-11-08)

**대상 독자:** 실험 절을 쓰는 사람. 단락마다 주장과 근거를 글머리로 잡아 둔다. 구조를 확정한 뒤 문장으로 옮긴다.

**수치 출처:** [RESULTS_2026-11-04.md](RESULTS_2026-11-04.md)(11-07 19:30판)와 [results_final.csv](results_final.csv)다. [RESULTS_FINAL.md](RESULTS_FINAL.md)는 11-06판이라 W4A4 Ours가 seed 1개로 남아 있다(S W4A4 18.60 ≠ 앞 절의 18.09). **다시 생성하기 전에는 거기서 수치를 가져오지 않는다.**

**표기**
- 기본은 seed 0–2의 3 seed 평균이다. 위첨자 ⁽¹⁾은 seed 0 하나뿐인 값이다.
- Δ는 같은 seed끼리 짝지은 차이의 평균 ± 95% 신뢰구간이다. 평균끼리 빼지 않는다.
- 설정마다 "LVIS AP a → b, flip c% → d%" 형식으로 쓴다. 본문은 소수 첫째 자리, 표는 둘째 자리다.
- 그림과 표는 [Fig. n], [표 n]으로 참조한다. 확인이 필요하거나 실험이 남은 곳은 [TODO]로 표시한다.
- 수식은 노션 규칙을 따른다. 본문 수식은 $$...$$, 블록 수식은 `equation` 칸의 내용을 /equation 블록에 그대로 넣는다.

**분량:** 2.9쪽(5.1 설정 0.4쪽, 5.2 주 결과 1.3쪽, 5.3 ablation 0.7쪽, 5.4 discussion 0.4쪽). 5.1과 5.2를 합쳐 이전 0.5쪽에서 약 0.1쪽을 줄였다.

---

## 5. Experiments (도입 단락)

**주장:** 실험은 세 질문에 답한다.
- Q1. 기존 PTQ 위에 얹었을 때, 실시간 OVOD 6개의 W4 설정에서 LVIS AP를 높이고 판정 손상(flip)을 줄이는가? (5.2)
- Q2. 세 처방은 각각 무엇을 기여하고, 언제 필요한가? (5.3)
- Q3. 다른 구조(ViT·DETR), 다른 도메인, 다른 calibration 데이터에서도 같은 양상인가? (5.2 끝, 5.4, 부록)

---

## 5.1 Experimental Setup

한 절로 묶고, 굵은 머리말 단락(`\paragraph{}`) 다섯 개로 나눈다: **Models / Datasets and metrics / Quantization protocol / Baselines / Our configuration.** 절 제목과 도입 문장이 하나로 줄어 약 0.1쪽이 준다.

**Models. 주장: 실시간 OVOD 6개를 주 대상으로, ViT·DETR 2개를 확장 대상으로 둔다.**
- 주 대상: YOLO-World S / S-v2 / M / L (텍스트 게이트 있음), YOLOE-v8s / 11s (텍스트 게이트 없음).
- 확장 대상: OWLv2-B/16, Grounding DINO-T.
  - OWLv2는 self-training 체크포인트 `google/owlv2-base-patch16`를 쓴다. LVIS로 fine-tuning된 체크포인트는 LVIS base 클래스를 봤으므로 쓰지 않는다.
  - 확장 대상은 seed 0 하나이고, 사전 검사 대신 고정 구성을 쓴다. OWLv2는 W8A8 M, W4A8 PM이고, GDINO는 PM이다.

**Datasets and metrics. 주장: calibration 어휘와 배포 어휘를 나눠 평가하고, AP와 함께 판정 손상(flip)을 보고한다.**
- calibration: COCO train2017 앞 256장, COCO 80 어휘.
- 평가:
  - calibration 어휘: COCO val2017 AP.
  - 배포 어휘: LVIS minival AP (1203 클래스). LVIS 어휘에는 COCO 80개 개념이 대부분 들어 있다. 그래서 "처음 보는 어휘"가 아니라 **"calibration에 쓰지 않은 더 큰 어휘"**라고 쓴다.
- LVIS flip: FP 모델이 확신하는(점수 > 0.25) 위치 중, 양자화 뒤 1203 어휘 전체의 top-1 클래스가 바뀐 비율이다. 정의는 3.1을 참조한다. 낮을수록 좋다.
- 같은 지표끼리 비교한다. COCO와 LVIS를 나란히 놓을 때는 AP는 AP끼리, flip은 flip끼리 놓는다.
- 개선폭은 seed 0–2의 짝 차이 평균 ± 95% t 신뢰구간이다. 지면을 아끼려고 블록 수식 대신 한 문장으로 쓴다. 예: "we report the mean of per-seed paired differences over three seeds with a 95% t-interval."
  - 짝 차이를 쓰는 이유(각주 후보): seed 하나가 크게 흔들리면 평균끼리의 차이가 부풀려진다. OWLv2 W8A8은 BRECQ 3 seed 평균(26.62)으로 빼면 +3.18이지만, seed 0 짝 차이는 +1.67이다.
- ODinW-13(다른 도메인)과 LVIS mask AP(segmentation)는 부록에서 다룬다고 한 줄 적는다.

**Quantization protocol. 주장: 검출기 전체를 정수 연산 형태로 양자화한다.**
- weight는 per-channel, activation은 per-tensor다. 설정은 W8A8, W4A8, W4A6, W4A5, W4A4다.
- 첫 conv와 head 마지막 1×1은 W8A8이다(관례). head 마지막 1×1의 입력만 A16이다.
- attention의 Linear·matmul·softmax 출력과 region–텍스트 내적은 8bit다.
- 텍스트 인코더는 FP이고 오프라인에서 돈다. 어휘 임베딩은 INT8 상수로 넣는다.
- GDINO는 softmax 16bit, box head 출력 W8A16, 패치 conv W8A8이다. softmax를 8bit로 두면 AP가 1.2 떨어진다([docs/GDINO_CHANGES.md](../docs/GDINO_CHANGES.md)). 본문에는 "모델별 예외는 부록"으로 한 줄만 쓴다.
- [TODO] OWLv2 프로토콜의 예외를 `run_owl.py` 기준으로 적는다.

**Baselines. 주장: 기준선은 재구성 계열의 대표인 BRECQ와 QDrop이고, 같은 조건으로 돌린다.**
- 기준선: RTN, AdaRound, BRECQ, QDrop, BRECQ + SmoothQuant, BRECQ + AWQ.
  - SmoothQuant와 AWQ는 우리 M과 같은 xmax 규칙(calibration 256장)으로 돌렸다.
- 재구성은 모든 방법이 2,000 iters다(원 논문 20,000). batch는 1이고, 학습률은 각 방법의 공식값이다. QDrop의 drop 확률은 0.5다.
  - 근거: BRECQ에서 2k와 20k는 W8A8 최종 지표가 같다([docs/BASELINE_STATUS.md](../docs/BASELINE_STATUS.md) §4.1).
  - 단, rounding이 수렴했다고 쓰지 않는다. "수렴하지 않았지만 지표가 같았다"로 쓴다.
  - QDrop은 M W4A5에서 20,000 iters로 늘려도 BRECQ보다 낮았다(seed 1: 12.40 vs BRECQ 19.94, 2k). 반복 수 부족 탓이 아니다. 각주로 둔다.
- PD-Quant는 공식 설정에서 QDrop보다 낮아 본 실험에서 뺐다. 부록에만 언급한다.
- [TODO] AdaRound는 W8A8과 W4A8만 있다. W4A6 이하를 채울지(약 1.5시간 × 3칸), 표에서 뺄지 정한다.

**Our configuration. 주장: 우리 구성은 calibration 쪽 데이터만으로 정해진다.**
- G는 구조가 정한다. 텍스트 게이트가 있는 YOLO-World에만 적용한다.
- P는 진단이 위치를 정한다. 예산은 모든 모델에 1.5%로 고정한다.
- M은 사전 검사가 사용 여부를 정한다. 검사 지표는 $$\mathrm{Flip}_{\mathrm{chk}}$$다. calibration 다음 순서의 train2017 200장과 COCO 어휘로 잰다.
  - seed 0에서 한 번 고르고, 모든 seed와 QDrop 위 실행에 같은 구성을 쓴다.
- 결정적 실행이고, seed는 0–2다.
- [TODO] GPU 종류(RTX 4000 Ada)와 모델당 빌드 시간을 적는다.

---

## 5.2 Main Results

### 단락 1. YOLO-World-S의 비트별 결과 [표 1]

**주장: W4 전 비트에서 기존 PTQ보다 LVIS AP가 높고 flip이 낮으며, 비트가 낮을수록 차이가 커진다.**
- W4A8: LVIS AP 22.7 → 24.2, flip 17.7% → 10.7% (+1.49 ± 0.74).
- W4A5: LVIS AP 17.0 → 23.0, flip 39.4% → 16.3% (+6.01 ± 12.14).
  - BRECQ seed 0이 무너져서(11.29) 신뢰구간이 0을 포함한다. "유의하게"는 쓰지 않는다.
  - seed 1·2만 보면 BRECQ 19.8, Ours 23.0이다.
- W4A4: LVIS AP 0.4 → 18.1, flip 97.1% → 32.9% (+17.66 ± 1.06). QDrop은 0.2, flip 98.0%다.
- QDrop 위에서도 같다.
  - W4A8: LVIS AP 22.5 → 24.5, flip 18.5% → 11.1% (+2.03 ± 0.60).
  - W4A4: LVIS AP 0.2 → 14.0, flip 98.0% → 42.7% (+13.72 ± 2.75).
- W8A8은 유지 수준이다: LVIS AP 25.5 → 25.7, flip 3.8% → 2.7% (+0.22 ± 0.12).
- scale 이전 기준선과 비교한다. BRECQ + SmoothQuant와 BRECQ + AWQ는 W4A8에서 BRECQ보다 낮고(21.7–21.8, seed 0), W4A4에서 9.25와 0.33이다(seed 0).
  - [TODO] SmoothQuant / AWQ / RTN 행의 3 seed 값. RESULTS_FINAL 재생성 후 채운다.
- 끝 문장: 우리 처방은 기존 PTQ가 무너지는 비트에서 판정 손상을 가장 크게 줄인다.

**[표 1] 구성 (YOLO-World-S, LVIS AP / flip %)**

| 방법 | W8A8 | W4A8 | W4A6 ⁽¹⁾ | W4A5 | W4A4 |
|---|---|---|---|---|---|
| FP32 | 25.89 | | | | |
| RTN | [TODO] | [TODO] | 0.57 / 92.4 | [TODO] | [TODO] |
| AdaRound | [TODO] | [TODO] | – | – | – |
| BRECQ | 25.49 / 3.8 | 22.71 / 17.7 | 21.33 / 20.6 | 16.97 / 39.4 † | 0.43 / 97.1 |
| QDrop | 25.38 / 4.0 | 22.45 / 18.5 | 20.10 / 30.1 | 12.69 / 55.6 | 0.23 / 98.0 |
| BRECQ + SmoothQuant | [TODO] | [TODO] | 21.10 / 24.9 | [TODO] | [TODO] |
| BRECQ + AWQ | [TODO] | [TODO] | 20.38 / 26.9 | [TODO] | [TODO] |
| **Ours (BRECQ)** | **25.71 / 2.7** | **24.20 / 10.7** | **23.66 / 13.7** (GP) | **22.98 / 16.3** | **18.09 / 32.9** |
| └ GP | [TODO] | [TODO] | 23.66 / 13.7 | 22.07 / 19.9 | [TODO] |
| └ GPM | 25.71 / 2.7 | 24.20 / 10.7 | 23.34 / 12.6 | 22.98 / 16.3 | 18.09 / 32.9 |
| **Ours (QDrop)** | **25.79 / 2.7** | **24.48 / 11.1** | 23.17 / 14.0 | **22.35 / 17.8** | **13.95 / 42.7** |

- † BRECQ seed 0 붕괴.
- W4A6은 seed 0뿐이다. [TODO] 비트 곡선에 오차 막대를 넣을 거면 W4A6 seed 1·2를 돌린다(약 24건, 4–5시간).
- S W4A8은 사전 검사가 GPM을 골랐지만 GP가 더 높다(3 seed 모두, 0.26–0.60). 그래서 GP와 GPM 행을 둘 다 싣는다.

### 단락 2. 6개 실시간 OVOD [표 2]

**주장: 6개 모델의 W4A8·W4A5·W4A4 18칸 모두에서 BRECQ보다 LVIS AP가 높고 flip이 낮다.**
- 신뢰구간이 0을 포함하는 칸은 두 곳이다: S W4A5(†), YOLOE-11s W4A8(+0.18 ± 0.91). 그래서 "모든 칸에서 유의하게"라고 쓰지 않는다.
- 예:
  - YOLO-World-L W4A8: LVIS AP 27.9 → 32.0, flip 23.5% → 9.7% (+4.13 ± 1.24).
  - YOLOE-v8s W4A5: LVIS AP 17.8 → 23.2, flip 32.7% → 16.2% (+5.40 ± 2.62).
- QDrop 위에서도 W4 18칸 모두 이득이다. QDrop이 약한 칸(W4A5, W4A4)일수록 Δ가 크다.
  - 예: L W4A5는 LVIS AP 1.4 → 26.9, flip 87.8% → 21.8% (+25.44 ± 3.68)다.
- 끝 문장: 이득은 텍스트 게이트 유무와 모델 크기에 관계없이 나타난다.

**보조 주장: W4A4에서는 기존 PTQ가 무너지고, 우리 처방은 모델을 되살린다.**
- BRECQ는 0.04–3.44, QDrop은 0.01–0.23이다. Ours는 10.90–23.43이고, QDrop + Ours는 6.12–19.24다. 신뢰구간이 모두 0을 넘는다.
- 다만 FP까지는 못 간다. v2는 10.9(FP 22.2), M은 15.0(FP 29.8)으로 절반 정도다. 한계로 적는다.

**보조 주장: 텍스트 게이트가 없는 YOLOE는 W4A8 이득이 작다.**
- YOLOE-v8s W4A8: LVIS AP 23.7 → 24.6, flip 14.0% → 10.4% (+0.87 ± 0.08).
- YOLOE-11s W4A8: LVIS AP 22.8 → 23.0, flip 20.0% → 17.9% (+0.18 ± 0.91).
- G가 적용되지 않아 P만 남기 때문이다. W4A5와 W4A4에서는 이득이 크다.

**[표 2] 구성:** 행은 6개 모델 × 비트(W4A8, W4A5, W4A4)이고, 열은 FP32 / BRECQ / + Ours / Δ / 선택 / QDrop / + Ours / Δ다. 수치는 [RESULTS_2026-11-04.md](RESULTS_2026-11-04.md) §1을 그대로 옮긴다. W8A8은 부록 표로 뺀다.

### 단락 3. W8A8 (부록으로)

**주장: W8A8에서는 이득이 없거나 작아서 주 주장에서 뺀다.**
- 신뢰구간이 0을 넘는 이득은 S(+0.22 ± 0.12)뿐이다. 나머지는 −0.31 ~ +0.17이다.
- v2 W8A8은 M 때문에 flip이 5.0% → 7.9%로 나빠진다. AP는 22.7 → 22.7(−0.04 ± 0.29)이다.
- L W8A8은 LVIS AP 33.3 → 33.0, flip 3.6% → 3.8% (−0.31 ± 0.72)다.
- 한계(5.4)에 적는다.

### 단락 4. ViT·DETR 계열로의 확장 [표 3]

**주장: 같은 절차를 ViT(OWLv2)와 DETR(Grounding DINO)에 적용해도 BRECQ보다 LVIS AP가 오른다.**
- 단, 둘 다 seed 0 하나이고, 사전 검사 대신 고정 구성을 썼다. 확장 결과로만 쓴다.
- OWLv2-B/16 (짝 차이, seed 0):
  - W8A8, M: LVIS AP 28.1 → 29.8, flip 17.1% → 12.4% (+1.67).
  - W4A8, PM: LVIS AP 19.7 → 21.8, flip 28.2% → 25.9% (+2.06).
  - 손상이 퍼져 있어(집중도 35%) P 단독은 효과가 없다.
- Grounding DINO-T, W4A5, PM:
  - LVIS AP 27.3 → 30.2 (FP 30.1, QDrop 29.2).
  - LVIS minival 앞 2,000장으로 평가했다.
  - flip은 다른 모델과 정의가 달라 이 표에서 빼고 부록에 둔다. 이 설정만 AP만 쓰는 예외라 각주로 밝힌다.
- [TODO] OWLv2 Ours seed 1·2 (새 규칙). GDINO W4A8 / W4A4와 seed 1·2. 시간이 없으면 "seed 1개"를 표 각주에 적는다.
- [TODO] OWLv2 표의 BRECQ 값을 seed 0(28.14)으로 둘지, 3 seed 평균(26.62)과 따로 적을지 정한다. 결과 문서 §3은 28.14를 "3 seed 평균"이라고 적었는데, csv로 보면 seed 0 값이다. 결과 문서도 고친다.

### 단락 5. 정성 결과 [Fig. 5]

**주장: BRECQ는 박스 위치를 지킨 채 클래스를 비슷한 다른 클래스로 바꾸고, 우리 처방은 FP의 판정을 유지한다.**
- 예: helmet → cap (headwear), tennis ball → baseball, vase / bowl → bottle / cup ([figures/qual/qual_grid.png](../figures/qual/qual_grid.png)).
- 사례는 손으로 고르지 않고 규칙으로 골랐다.
  - FP가 맞힌 판정(점수 ≥ 0.3, GT와 클래스 일치, IoU ≥ 0.5) 중, 세 비트 모두에서 Ours는 FP와 같은 클래스이고 BRECQ는 다르거나 놓친 이미지다.
  - 후보는 88장이다. 희귀 > 흔함 > 잦음 클래스 순, 그다음 win 수 순으로 정렬한다.
- seed는 W4A8 s0, W4A5 s1, W4A4 s0이다. W4A5는 BRECQ seed 0이 무너져서 seed 1을 썼다. 캡션에 밝힌다.
- [TODO] 전체 win / loss 수를 캡션이나 부록에 적는다. `figures/src/qual_figure.py`가 출력한다.
- [TODO] 11-05 선택 규칙 변경 뒤에 그림을 다시 생성해 같은지 확인한다. S는 세 비트 모두 GPM이라 바뀌지 않을 것으로 본다.
- [TODO] (선택) 부록에 Ours가 틀린 사례 1장을 넣는다.

---

## 5.3 Ablation (YOLO-World-S)

**[표 4] 기준 seed 결정 [TODO]**
- 지금 모든 행이 있는 것은 seed 0 표다([RESULTS_2026-11-04.md](RESULTS_2026-11-04.md) §4). 그런데 W4A5는 BRECQ seed 0 붕괴 때문에 BRECQ 대비 Δ가 부풀려진다.
- 3 seed가 있는 행: BRECQ, +G, +P, +GP, +GPM, +R, +H.
- seed 0만 있는 행: +M, +GM, +PM, +GPS, +GPW.
- 선택지:
  - (a) seed 0 표로 두고, W4A5는 BRECQ 대비가 아니라 조건끼리 비교한다.
  - (b) 빠진 행의 seed 1·2를 W4A8 / W4A5에서 돌린다(5조건 × 2비트 × 2 seed = 20건).
- 아래 수치는 seed 0이다.

### 단락 1. 처방별 기여

**주장: 세 처방은 서로 다른 비트에서 필요하다. G는 W4A5에서, M은 W4A4에서 가장 크게 기여한다.**
- W4A5에서 단일 처방 중 G가 가장 크다: +G 21.20 vs +P 14.30 vs +M 20.66 (BRECQ 11.29, seed 0 붕괴).
  - 조건끼리 비교한다: GP − P = +8.03, GPM − GP = +0.61.
- W4A4는 M이 있어야 열린다: GP 2.54 vs GPM 18.60 (seed 0).
- M 단독은 W4A8에서 해롭다: LVIS AP 22.8 → 22.0, flip 17.5% → 20.4% (−0.78). G나 P가 받쳐 줄 때 이득이 된다.
- 끝 문장: 그래서 M을 항상 켜지 않고 사전 검사로 켤지 정한다.

### 단락 2. 보호 위치

**주장: 같은 예산에서는 어디를 보호하느냐가 중요하다. 진단 기반 위치는 무작위 위치보다 낫고, HAWQ식 기준과는 비슷하다.**
- W4A8 (seed 0, BRECQ 대비): 진단 P +0.96, HAWQ식 H +0.87, 무작위 R +0.25.
- **진단 기준이 새롭다고 주장하지 않는다.** H와 차이가 유의하지 않다.
- 우리 진단의 장점은 기준값이 아니라 데이터 조건이다. calibration 이미지와 calibration 어휘만 쓰고, 정답 라벨이 필요 없다.
- [TODO] 3 seed 값(P, H, R)을 결과 문서에서 확인해 옮긴다. 결과 문서 §4.2에는 P 단독 W4A8 +1.05 ± 0.20(6 seed)만 있다.

### 단락 3. 이전 규칙 교체

**주장: M의 이전 규칙은 SmoothQuant나 AWQ로 바꿔도 결과가 비슷하다. M의 새로운 점은 묶음 제약과 사전 검사다.**
- W4A8 (seed 0, BRECQ 대비): GPM +1.22, GPS +1.17, GPW +1.24.
- W4A5 (seed 0): GPM 22.94, GPS 21.93, GPW 21.90.
- **M의 규칙이 더 낫다고 주장하지 않는다.**

### 단락 4. 보호 예산 [Fig. 부록] 또는 [표 5]

**주장: 1.5% 예산은 성능이 꺾이는 지점에 있고, 예산은 M과 함께일 때만 성능으로 이어진다.**
- S W4A5, 3 seed: GPM은 1% → 1.5%에서 +0.41, 1.5% → 3%에서 +0.02다.
- GP는 예산과 무관하게 21.7–22.1로 평평하다.
- 예산은 이 sweep을 보기 전(09-29)에 정했다. "꺾이는 지점에서 골랐다"가 아니라 "고정 예산이 꺾이는 지점에 있었다"로 쓴다.

### 단락 5. 사전 검사의 정확도 [Fig. 6]

**주장: 사전 검사는 AP 차이가 큰 경우를 모두 맞히고, 틀려도 손해가 작다.**
- 73건(BRECQ 62 + QDrop 11) 중 60건에서 LVIS AP가 더 높은 쪽을 골랐다.
- |ΔAP| ≥ 1인 35건은 모두 맞혔다.
- 틀린 13건은 모두 BRECQ 기반이고, 손해는 최대 0.81 AP다.
- 구성을 하나로 고정하면 최대 10.51(항상 GPM)–22.52(항상 GP)를 잃는다.
- 놓친 경우를 숨기지 않는다: S W4A8(3 seed, 0.26–0.60), YOLOE-11s W4A8(0.17–0.77).
- [TODO] `figures/mcheck_scatter`는 42건 기준이다. 73건으로 다시 그린다(`figures/src/mcheck_scatter.py`가 `runs/234`·`236`만 읽는다).

---

## 5.4 Discussion

### 단락 1. Findings

**주장: 손상 집중도는 작은 예산으로 충분한 이유를 설명하지만, P의 효과 크기를 예측하지는 않는다.**
- 텍스트 게이트 모델은 손상의 57–83%가 G 대상 cv2 4개에 있다.
- 게이트 없는 모델은 1.5% 예산이 손상의 48–67%를 담는다.
- 손상이 퍼진 모델(OWLv2 35%, GDINO 5%)은 P 효과도 작다.
- 그러나 6개 모델에서 1.5% 비중과 P 효과의 순위 상관은 −0.17이다. **"집중도가 효과를 예측한다"고 쓰지 않는다.**

**주장: 보호 예산은 M과 함께일 때 성능으로 이어진다** (5.3 단락 4와 연결).

[TODO] "M이 취약성을 옮긴다(12.cv2 ↓, 1.conv ↑)"는 기존 개요에 있던 관찰이다. 새 규칙에서 다시 쟀는지 확인하고, 확인 전에는 쓰지 않는다.

### 단락 2. 일반화 (부록 표를 요약)

**주장: 이득은 다른 도메인, 다른 calibration 데이터, segmentation에서도 유지된다.**
- **ODinW-13 (seed 0):** W4A5는 6개 모델 모두 +8.8 ~ +26.0이다. W4A8은 YOLO-World 4종이 +2.1 ~ +6.5이고, YOLOE 두 모델은 −1.20 / −0.12다.
- **calibration 데이터 (S, seed 0):** COCO 대신 Objects365(+GQA)로 calibration해도 이득이 남는다. W4A8은 +1.03 ~ +1.73, W4A5는 +3.68 ~ +4.31이다.
- **segmentation (LVIS mask AP):** W4A8은 +0.6 ~ +1.2(3 seed, 같은 방향), W4A5는 +2.7 ~ +5.1(seed 0)이다.
- **첫·마지막 레이어 FP (W4A4):** BRECQ는 회복되지 않는다(S 0.42, M 0.76). W4A4 붕괴는 첫·마지막 레이어 때문이 아니다.

### 단락 3. 비용

**주장: 추가 비용은 크기 +1.3–1.6%와 빌드 시간 증가뿐이다. 연산 형태는 기준선과 같다.**
- 크기: P만 늘린다(1.35–1.59% 보호, 평균 4.05–4.06 bit). G와 M은 늘리지 않는다.
- 진단: 모델당 1회다. YOLO s급 약 10분, OWLv2 약 1.5시간이다.
- 사전 검사: 검사 자체는 약 20초다. 두 후보를 만드는 비용은 BRECQ 단독의 약 2.4–2.9배다.

### 단락 4. 한계

**주장: 우리 처방은 판정 손상을 줄이지만 없애지는 않는다.**
- W4A8에서도 flip이 10.7% 남는다. **"판정을 지킨다"가 아니라 "판정 손상을 줄인다"로 쓴다.**
- W4A4는 FP의 절반 정도에 그치는 모델이 있다(v2, M).
- 사전 검사가 놓치는 경우가 있다(S W4A8, YOLOE-11s W4A8, v2 W8A8).
- [TODO] v2에서 GPM이 PM보다 나쁘다: W4A8 GPM 19.05(3 seed) vs PM 22.45(seed 0). 그대로 보고할지, 표에 PM을 함께 실을지 교수님과 정한다.
- 실기기 지연은 측정하지 않았다.
- OWLv2와 GDINO는 seed 1개이고 고정 구성이다.

---

## 표와 그림 목록

| 번호 | 내용 | 위치 | 상태 |
|---|---|---|---|
| [표 1] | YOLO-World-S 비트별, 비교 방법 전체 | 5.2 | 🟡 RTN·SQ·AWQ·GP 3 seed 칸 [TODO] |
| [표 2] | 6개 모델 × W4A8/W4A5/W4A4, BRECQ·QDrop 짝 | 5.2 | ✅ 결과 문서 §1 |
| [표 3] | OWLv2, GDINO (확장) | 5.2 | 🟡 seed 0 |
| [표 4] | S ablation | 5.3 | 🟡 seed 기준 결정 [TODO] |
| [표 5] 또는 부록 | P 예산 sweep | 5.3 | ✅ 3 seed |
| [Fig. 5] | 정성 결과 | 5.2 | 🟡 win/loss 수, 재생성 확인 |
| [Fig. 6] | 사전 검사 산점도 | 5.3 | 🟡 73건으로 다시 그리기 |
| 부록 | W8A8, ODinW, calibration 출처, segmentation, 첫·마지막 FP, QDrop 20k | 5.4 | ✅ |

## 앞 절과의 수치 일관성

| 수치 | 앞 절 | 실험 절 | 확인 |
|---|---|---|---|
| S W4A8 BRECQ: COCO AP 36.80 → 33.75, LVIS AP 25.89 → 22.71, flip 17.7% | 서론 ②, 3.1 | 표 1 | ✅ csv 3 seed: COCO 33.75, LVIS 22.71 |
| COCO 어휘 flip 3.9% | 서론 ②, 3.1 | – | ⚠️ 6 seed 값(3.92)이다. 3 seed 값으로 바꿀 예정 [TODO] |
| S W4A8 Ours: 22.7 → 24.2, flip 17.7% → 10.7% | 초록, 서론 ⑤ | 5.2 단락 1 | ✅ |
| S W4A4 Ours: 0.4 → 18.1, flip 97.1% → 32.9% | 초록, 서론 ⑤ | 5.2 단락 1, 표 1 | ✅ 결과 문서 3 seed. ⚠️ RESULTS_FINAL은 18.60(seed 0)이라 재생성 필요 |
| 크기 증가 2% 미만 (+1.3–1.6%) | 초록, 서론, 4.2 | 5.4 단락 3 | ✅ |
| 집중도 57–83% / 48–67% | 서론 ③, 3.3 | 5.4 단락 1 | ✅ 결과 문서 §4.2 |
| W8A8 BRECQ LVIS AP −0.4, flip 3.8% | 3.1 | 5.2 단락 3 | ✅ 25.89 → 25.49 |

## [TODO] 모음

1. **RESULTS_FINAL 재생성.** 리더보드의 `pick_ours`가 GP·GPM이 둘 다 있는 seed만 세서, W4A4 Ours가 seed 0만 잡힌다. 내보내기 규칙을 결과 문서와 맞춘 뒤 다시 뽑는다.
2. 표 1의 RTN / AdaRound / SmoothQuant / AWQ / GP 3 seed 칸 채우기 (재생성 후).
3. 표 4의 seed 기준 결정: (a) seed 0 표, 또는 (b) +M / +GM / +PM / +GPS / +GPW의 seed 1·2를 W4A8·W4A5에서 돌림(20건).
4. OWLv2 BRECQ 표기(seed 0 28.14 vs 3 seed 26.62) 정리. 결과 문서 §3 각주도 고친다.
5. OWLv2 프로토콜 예외 확인 (`run_owl.py`).
6. GPU 종류와 빌드 시간 적기.
7. [Fig. 5] win/loss 수, 11-05 규칙으로 다시 생성해 확인.
8. [Fig. 6] 73건으로 다시 그리기.
9. "M이 취약성을 옮긴다" 관찰을 새 규칙으로 다시 쟀는지 확인.
10. v2 GPM < PM 처리 (교수님).
11. AdaRound 행 유지 여부 (교수님).
12. COCO 어휘 flip을 3 seed 값으로 교체.
