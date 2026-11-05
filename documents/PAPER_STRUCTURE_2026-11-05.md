# 논문 구성: 교수님 틀에 맞춘 절별 내용 (2026-11-05)

**대상 독자:** 본문을 쓰는 사람. 교수님이 제안한 절 구성(Abstract → Intro → Related Works → Motivation → Methods → Experiments → Conclusion)에 기존 개요의 내용을 어디에 넣을지 정리한다.

**관련 문서**
- 기존 개요(진단 중심): [PAPER_OUTLINE_2026-11-03.md](PAPER_OUTLINE_2026-11-03.md)
- 방법 그림 배치: [FIGURE_METHOD_PLACEMENT_2026-11-04.md](FIGURE_METHOD_PLACEMENT_2026-11-04.md)
- 방향과 근거: [PAPER_DIRECTION_2026-11-02.md](PAPER_DIRECTION_2026-11-02.md)

**수치 기준:** 대부분 seed 0이다. seed 1·2가 들어오면 다시 맞춘다.

---

## 0. 기존 개요와 달라지는 점

기존 개요는 진단 중심이었다(3 지표·진단 → 4 분석 → 5 처방). 교수님 틀은 Motivation → Method 구조라서 다음 두 가지를 정한다.

1. **진단(D)과 분석을 Motivation(3절)에 넣는다.** "기존 PTQ의 한계를 Figure로 보인다"는 요구에 기존 4절(손상 집중 + 구조 따라감 + 게이트 메커니즘)이 그대로 맞는다. 진단 절차도 3절에서 정의해 두면, 4.2 P와 4.3 M에서 "3.2의 진단을 쓴다"고만 쓰면 된다.
2. **Motivation을 너무 줄이지 않는다.** 10-18에 정한 novelty가 "G + 손상 위치 분석"이라서, 분석이 Motivation 안에 들어가더라도 기여 C1·C2의 근거가 여기에 있다. 1.3쪽 정도는 지킨다.

## 페이지 배분 (본문 8쪽)

| 절 | 분량 |
|---|---|
| Abstract + 1 Introduction | 1.2쪽 |
| 2 Related Work | 0.6쪽 |
| 3 Motivation | 1.3쪽 |
| 4 Method | 1.6쪽 |
| 5 Experiments | 2.9쪽 |
| 6 Conclusion | 0.3쪽 |
| LLM Usage Statement | 쪽수 제외, **필수** |

---

## Abstract

기존 초안(개요 §3)을 그대로 쓰되, 흐름을 **문제 → 원인(Motivation) → 방법 이름과 세 기법 → 결과** 순서로 맞춘다. 방법 이름이 정해지면 초록에 넣는다.

## 1. Introduction (5단락, 두괄식)

각 단락의 첫 문장이 주장이고, 주요 단락은 끝에서 그 주장을 다시 쓴다.

| 단락 | 첫 문장 (주장) | 내용 | 끝 문장 (다시 주장) |
|---|---|---|---|
| ① 배경 | 엣지의 OVOD는 배포 후에 어휘가 바뀌는데, PTQ는 calibration 어휘로만 조정하고 평가한다 | OVOD(YOLO-World 등)의 엣지 수요, 저비트 PTQ의 필요성 | – |
| ② 문제 | 이 평가 방식은 저비트 PTQ의 주된 실패를 가린다 | W4A8 BRECQ: COCO AP는 −3.0으로 버티지만 LVIS flip은 17.6%. **Fig. 1** (비트별 AP와 flip) | 그래서 AP만으로는 OVOD 양자화를 판단할 수 없다 |
| ③ 원인 | 이 손상은 고르게 퍼지지 않고, 구조가 정하는 소수 지점에 몰려 있다 | weight 1.5%가 손상의 48–72%. 텍스트 게이트 융합 conv / 초반 backbone / ViT는 분산 | 그러므로 몰린 지점만 다루면 된다 |
| ④ 방법 | 우리는 [방법 이름]을 제안한다. 진단이 G·P·M 중 무엇을 쓸지 정한다 | G·P·M을 한 줄씩. 학습 없음, BRECQ/QDrop plug-in | – |
| ⑤ 결과와 기여 | 8개 검출기에서 held-out 판정을 지키고, W4A4에서도 동작한다 | 대표 수치 1–2개 + 기여 목록 C1–C4 (bullet) | – |

## 2. Related Works

네 단락으로 쓰고, 각 단락 끝에 "우리와의 차이"를 한 문장씩 둔다.
- **PTQ:** AdaRound, BRECQ, QDrop. 검출 PTQ(Reg-PTQ)는 인용만 한다.
- **Outlier와 scale 이전:** SmoothQuant, AWQ. CLE는 SiLU에서 FP 등가가 성립하지 않는다고 한 줄 쓴다.
- **혼합 정밀도:** HAWQ, HAQ. 차이는 외부 제약 아래에서 비트를 배분하는 게 아니라, 손상이 몰려 있어서 작은 고정 예산이면 된다는 점이다.
- **OVOD:** YOLO-World, YOLOE, OWLv2, GLIP, Grounding DINO. OVOD 양자화 연구는 거의 없다는 점을 쓴다.

## 3. Motivation: 기존 PTQ의 한계

| 절 | 내용 | Figure/표 |
|---|---|---|
| 3.1 판정 충실도 | AP 대신 flip/lost를 정의한다. COCO = calibration 어휘, LVIS = held-out 어휘 | Fig. 1을 여기서 다시 참조 |
| 3.2 누수 없는 진단 | 층 하나만 W4로 내리고, calibration 쪽 이미지 200장과 COCO 어휘로 flip 증가를 잰다. val과 LVIS는 보지 않는다 | – |
| 3.3 관찰 1: 손상은 몰려 있다 | 집중도 48–72% (OWLv2는 35%) | **Fig. 2** 민감도 지도 ([figures/sensitivity_maps.pdf](../figures/sensitivity_maps.pdf)), 표 A |
| 3.4 관찰 2: 위치는 구조를 따른다 + 메커니즘 | 게이트 분기 y₃의 범위 20.4 vs 다른 분기 7–9. 텍스트 분기 하나가 공유 scale을 오염시킨다 | **Fig. 3** 분기별 범위 (❌ 아직 없음) |

마지막 단락에서 관찰 하나하나를 4절의 기법으로 연결한다: 게이트 오염 → G, 소수 지점 집중 → P, 채널 편차 → M.

## 4. Method

**이름 정하기.** 논문용 이름 후보다. 교수님과 상의해서 정한다.
- 전체 프레임워크: 제목 1안에 맞추면 *Diagnosis-Guided Quantization (DGQ)* 같은 형태.
- 기법: **G** = Gate Commutation, **P** = Diagnosis-guided Protection, **M** = Constrained Scale Migration.

**4절 도입 단락 + 전체 그림 (Fig. 4).** input → OVOD(Text Encoder / Backbone / RepVL-PAN / Head) → output 파이프라인 위에 기법을 배지처럼 붙이는 형태다. 지금 [figures/method_arch.png](../figures/method_arch.png)가 이 형태이고, 고칠 곳은 [placement 문서](FIGURE_METHOD_PLACEMENT_2026-11-04.md) §8에 있다. 핵심은 다음과 같다.
- G는 T-CSPLayer 4곳에만 붙인다.
- P는 백본 앞쪽 1·2·4·6에만 붙인다.
- M은 conv → conv 연결 전체에 걸리고, T-CSPLayer 출력은 제외한다.
- D(진단)는 모델 밖 점선 영역에 둔다.
- 빌드 순서는 배지 번호(①–⑤)로 보인다.

| 절 | 들어갈 내용 |
|---|---|
| 4.1 G | 식 `cv2(concat(y0,y1,y2,p⊙a)) = W_main·[...] + Σ a_h⊙(W_h·p_h)`. FP 등가이고 비트 비용은 0이다. 적용 조건은 구조(텍스트 게이트)라서 데이터를 보지 않는다. 원래 구조와 G 적용 후를 나란히 그린 펼친 그림을 넣는다 |
| 4.2 P | 3.2 진단의 상위 층을 고정 예산 1.5% 안에서 W8로 둔다. G 대상은 먼저 빼고 예산을 다시 채운다. 근거는 집중도이고, "1.5%를 집중도를 보고 정했다"고는 쓰지 않는다. 크기는 +1.6% |
| 4.3 M | 생산 conv가 공유하는 s와 residual 제약을 설명한다. 채널 계보는 측정으로 찾고, xmax는 calibration 전체에서 잰다. **사전 검사**(calibration 밖 flip, ρ=1.0)로 M을 켤지 정한다 |
| (4.4 또는 4.3 끝) | BRECQ/QDrop 위 plug-in이라는 점과 빌드 순서를 한 단락으로 쓴다 |

## 5. Experiments

| 절 | 내용 |
|---|---|
| 5.1 Datasets & Evaluation | calibration은 train2017 256장. 평가는 COCO val(calibration 어휘)과 LVIS minival(held-out), 지표는 AP, flip, lost. 짝 Δ ± 95% CI로 보고한다 |
| 5.2 Implementation | 모델 8개, 기준선(RTN/BRECQ/QDrop/+SQ/+AWQ), 양자화 범위 프로토콜. 각주에 2k iters와 OWLv2/GDINO의 예외를 밝힌다 |
| 5.3 Main Results (정량) | **표 1** YOLO-World-S 비트별 (W8A8–W4A4), **표 2** 8개 모델 일반화 |
| 5.3 Main Results (정성) | **Fig. 5** 같은 LVIS 이미지에서 FP / BRECQ / Ours 검출 결과. BRECQ의 라벨이 바뀌는(flip) 사례를 보인다. **❌ 아직 없음, 새로 만든다** |
| 5.4 Ablation | **표 3** G/P/M 조합, 보호 기준(진단 / 무작위 / HAWQ), 이전 규칙 교체(M / SQ / AWQ) |
| 5.5 Discussion | 아래 목록 |

**5.5 Discussion에 넣을 관찰**
- **Findings:** 집중도가 P의 효과를 예측한다(OWLv2는 집중도 35%라 P 효과가 없음). M은 취약성을 옮긴다(12.cv2 → 1.conv). P 예산은 M과 함께일 때만 성능으로 이어진다.
- **비용:** 크기 +1.4–1.6%, 진단 시간(S 약 10분).
- **한계:** W4A4는 FP 대비 −7. M 검사가 놓치는 경우가 있다(S W4A8, v2 W8A8). 실기기 지연은 측정하지 않았다.

## 6. Conclusion

3–4문장으로 쓴다: 문제, 원인(집중 + 구조), 진단 기반 처방, 결과. 바로 뒤에 **LLM Usage Statement**를 둔다.

---

## 남은 일

**아직 없는 그림**
- Fig. 3: 게이트 분기 범위 메커니즘
- Fig. 5: 정성 결과

**교수님과 정할 것** (기존 개요 §8의 미결 사항)
- 방법 이름
- v2의 GPM < PM 처리
- seed 수
