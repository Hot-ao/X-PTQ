# 연구 시나리오: 개방 어휘 객체 검출기의 저비트 사후 양자화

작성: 하태호 · 2026-10-30

---

## 1. 배경: 개방 어휘 객체 검출(OVOD)의 확산

- 개방 어휘 객체 검출(open-vocabulary object detection, OVOD)은 학습 때 정해 둔 클래스가 아니라 실행 시점에 텍스트로 준 어휘를 검출하는 방식이다. 재학습 없이 새로운 대상을 찾을 수 있어 활용 범위가 넓다.
- 주요 학회에서 실시간 모델이 잇따라 나왔다. YOLO-World(CVPR 2024), YOLOE(ICCV 2025), Grounding DINO, OWLv2가 대표적이다.
- 실제 사용도 늘고 있다.
  - 공개 저장소 GitHub star: Grounding DINO 약 10.7k, YOLO-World 약 6.6k, YOLOE 약 2.3k (2026-10-30 기준)
  - YOLO-World와 YOLOE는 Ultralytics 패키지에 통합되어 있다.
  - 임베디드 카메라에서 YOLO-World로 어휘를 실행 중에 바꾸는 사례도 있다(YOLO-Jevois, 2025).
- 이런 모델을 엣지 기기에 올리려면 저비트 양자화가 필요하다.

## 2. 공백: OVOD 양자화 연구가 부족하다

- 직접 관련된 선행 연구는 **QATMA**(arXiv 2603.05964, 2026)다.
  - 재학습이 필요한 양자화 인지 학습(QAT) 방법이다.
  - YOLO-World M/L/X를 W4A4·W4A5에서 다루며, 기존 사후 양자화(PTQ)는 W4A4에서 완전히 실패한다고 보고했다.
  - 무엇이 왜 망가지는지에 대한 층 단위 분석은 없다.
- **재학습 없는 PTQ 관점에서 OVOD의 손상 원인을 분석하고 대응한 연구는 찾지 못했다.**
- 최근 검출 PTQ 연구도 모두 고정 클래스(closed-set) 검출기를 대상으로 한다. 두 연구 모두 BRECQ(2021) 또는 QDrop(2022)을 기준선으로 비교한다.
  - Reg-PTQ(CVPR 2024): 박스 회귀를 다룬다.
  - InlierQ(ICLR 2026): activation 양자화 범위를 다룬다.

## 3. 문제: 기존 검출기 양자화를 그대로 쓰면 무엇이 망가지는가

- 기존 PTQ(BRECQ, QDrop)는 층 출력 오차를 줄이도록 양자화 파라미터를 맞춘다. 어떤 어휘로 쓰일지는 고려하지 않는다.
- OVOD에서는 양자화할 때 본 어휘와 배포 후에 쓸 어휘가 다르다. 실험에서는 양자화 때 COCO 80 클래스를 쓰고, 평가는 처음 보는 LVIS 1203 클래스로 했다.
- 실제로 양자화할 때 본 어휘의 AP는 크게 떨어지지 않는데, 처음 보는 어휘의 판정은 크게 무너진다.

| YOLO-World-S, 6 seed 평균 | COCO AP | LVIS AP | LVIS 판정 flip |
|---|---|---|---|
| FP32 | 36.80 | 25.89 | – |
| BRECQ W8A8 | 36.75 | 25.46 | 3.8% |
| BRECQ W4A8 | 33.77 (−3.0) | 22.74 (−3.2) | **17.6%** |

- 판정 flip은 FP 모델이 확신하는(점수 > 0.25) 위치 중에서 양자화 뒤 top-1 클래스가 바뀐 비율이다.
- W4A8에서 FP가 확신하던 예측의 약 6분의 1이 다른 클래스로 바뀐다. AP만 보면 이 손상이 잘 드러나지 않는다.

## 4. 기준: 텍스트 프롬프트 점수의 역전(flip)

- OVOD의 판정은 한 영역에 대해 텍스트 프롬프트별 점수를 매기고 가장 높은 것을 고르는 일이다.
- 양자화 오차는 이 순위를 뒤집는다(prompt score 역전). 박스 위치가 거의 같아도 클래스가 바뀐다.

**판정 flip의 정의.** 영역 $i$와 프롬프트 $k$의 점수를 $s_{i,k}$, 양자화 후 점수를 $\hat s_{i,k}=s_{i,k}+\delta_{i,k}$로 둔다. FP가 확신하는 영역 집합을 $\mathcal C=\{\,i:\max_k s_{i,k}>\tau\,\}$라 하면 어휘 $\mathcal V$에서의 flip은 다음과 같다.

$$
\mathrm{Flip}(\mathcal V)=\frac{1}{|\mathcal C|}\sum_{i\in\mathcal C}\mathbb 1\!\left[\arg\max_{k\in\mathcal V}\hat s_{i,k}\neq\arg\max_{k\in\mathcal V}s_{i,k}\right]
$$

**역전 조건.** FP의 top-1을 $k^{*}$, 다른 프롬프트 $k$와의 점수 차(margin)를 $m_{i,k}=s_{i,k^{*}}-s_{i,k}$로 두면 다음이 성립한다.

$$
\text{flip}_i \iff \exists\,k\neq k^{*}:\ \delta_{i,k}-\delta_{i,k^{*}}>m_{i,k}
$$

같은 크기의 양자화 오차 $\delta$라도 어휘가 크고 비슷한 클래스가 많으면 작은 margin이 많아 역전이 쉽게 일어난다. 실측으로 LVIS에서 margin이 0.5보다 작은 위치는 COCO의 약 18배다. 처음 보는 큰 어휘에서 판정이 더 크게 무너지는 이유가 이것이다.

- 그래서 AP와 함께 **판정 flip**을 기준으로 삼는다. flip은 기준이 되는 동시에 진단 도구가 된다.
  - **평가 지표:** AP와 함께 보고한다.
  - **라벨 없이 계산된다:** flip은 FP 모델 출력과만 비교하므로 정답 라벨이 필요 없다. 그래서 라벨 없는 calibration 이미지만으로 손상이 어디서 생기는지 잴 수 있다.
  - **누수 없는 진단:** 진단과 결정에 평가 데이터를 쓰지 않는다. calibration 이미지와 calibration 어휘만 쓴다. 이렇게 고른 처방이 처음 보는 어휘(LVIS)에서도 통하는지를 본다.
- 즉, flip을 손실로 직접 최적화하는 것이 아니다. **flip으로 역전이 생기는 위치와 원인을 찾고, 그 원인을 구조적으로 막는다.**

## 5. 원인과 처방: G · P · M

flip 진단으로 찾은 손상 원인은 세 가지다. 각 원인에 처방을 하나씩 대응시킨다.

| | 원인(진단 결과) | 처방 | 적용 여부 결정 |
|---|---|---|---|
| **G** 게이트 교환 | 텍스트 게이트가 곱해진 분기가 다른 분기와 양자화 scale을 공유해서 오염된다 | 게이트 곱셈을 1×1 conv 뒤로 옮긴다. FP 결과는 수학적으로 같다 | 입력 의존 게이트 뒤에 Linear가 오는 구조에만 적용 |
| **P** 손상 위치 보호 | 판정 손상이 소수 층에 몰려 있다 | 층마다 W4로 내렸을 때 flip이 얼마나 느는지 calibration 이미지로 재고, 상위 층을 파라미터 1.5%만큼 W8로 둔다 | 고정 예산, 자동 선택 |
| **M** 공유 제약 scale 이전 | activation 채널 outlier | 같은 입력을 받는 층들이 하나의 scale을 공유하도록 묶어서 이전한다. 표준 정수 커널로 배포할 수 있다 | calibration 밖 이미지에서 flip을 비교해 켜고 끈다 |

### 5.1 G: 게이트 교환 (FP 등가 변환)

텍스트 융합 블록의 마지막 1×1 conv는 여러 분기를 이어 붙인 입력을 받는다. 그중 텍스트 분기는 head $h$의 위치별 게이트 $a_h$가 곱해진 $a_h\odot p_h$다. 1×1 conv는 선형이므로 다음이 정확히 성립한다.

$$
W\big[\,y\,;\,a_1\odot p_1;\dots;a_H\odot p_H\big]=W_{\text{main}}\,y+\sum_{h}a_h\odot\big(W_h\,p_h\big)
$$

양자화하면 두 식이 달라진다.

$$
\underbrace{Q_w(W)\,Q_a\big([\,y;\,a\odot p\,]\big)}_{\text{기존: scale 공유, 게이트 편차 포함}}
\;\longrightarrow\;
Q_w(W_{\text{main}})\,Q_a(y)+\sum_h a_h\odot Q_w(W_h)\,Q_a(p_h)
$$

- 텍스트 분기가 자기 전용 weight·activation scale을 가진다.
- 양자화되는 값이 게이트를 곱하기 전의 $p_h$라서, 게이트가 만드는 위치별 크기 편차가 양자화 범위에서 빠진다.
- 게이트가 작은 위치에서는 양자화 오차도 $a_h$배로 줄어든다.

### 5.2 P: 예산 아래 손상 위치 보호

naive W8A8 모델 $Q^{8}$에서 층 $\ell$ 하나만 W4로 내린 모델을 $Q^{8}_{\ell\to4}$로 둔다. calibration 이미지와 calibration 어휘(cal)로 판정 손상 증가량을 잰다.

$$
\Delta_\ell=\mathrm{Flip}_{\text{cal}}\big(Q^{8}_{\ell\to4}\big)-\mathrm{Flip}_{\text{cal}}\big(Q^{8}\big)
$$

$$
\mathcal S^{*}=\arg\max_{\mathcal S}\sum_{\ell\in\mathcal S}\Delta_\ell
\quad\text{s.t.}\quad\sum_{\ell\in\mathcal S}|\theta_\ell|\le B\,|\theta|,\qquad B=0.015
$$

$\Delta_\ell$이 큰 층부터 담는 greedy로 풀고, $\mathcal S^{*}$의 weight를 8bit로 둔다. 평가 이미지와 평가 어휘는 쓰지 않는다.

### 5.3 M: 공유 제약 scale 이전과 사전 검사

기본 등가식과 $s$ 공식은 SmoothQuant 계열과 같다.

$$
Wx=\big(W\,\mathrm{diag}(s)\big)\big(\mathrm{diag}(s)^{-1}x\big),\qquad
s_c(\alpha)=\frac{\max|x_c|^{\alpha}}{\max|W_{:,c}|^{1-\alpha}}
$$

우리가 더한 것은 세 가지다.

**(1) 공유 제약.** 배포에서 $x/s$는 생산 층의 requant 단계에서 만들어진다. 따라서 같은 생산 채널 그룹 $g$를 받는 모든 소비 층 $\mathcal C(g)$와, residual add로 합쳐지는 채널들은 하나의 $s$를 공유해야 한다.

$$
s_c=s_{g(c)},\qquad \max|x_c|\leftarrow\max_{j\in\mathcal C(g)}\max\big|x^{(j)}_c\big|
$$

**(2) "이전 없음"을 포함한 α 선택.** 단위 $U$마다 출력 오차를 최소화하는 $\alpha$를 고른다. $\varnothing$은 $s=1$(이전 없음)이라 어떤 단위도 이전 때문에 나빠지지 않는다.

$$
\alpha^{*}_U=\arg\min_{\alpha\in\{0,\,0.25,\,0.5,\,0.75,\,1\}\cup\{\varnothing\}}
\sum_{j\in U}\frac{\big\|Q_w\big(W_j\,\mathrm{diag}(s)\big)\,Q_a\big(x_j/s\big)-W_jx_j\big\|^2}{\|W_jx_j\|^2}
$$

**(3) 사전 검사.** calibration에 쓰지 않은 이미지(held-cal)에서 M을 켠 모델과 끈 모델의 판정 손상을 비교한다.

$$
\text{M 사용}\iff\mathrm{Flip}_{\text{held-cal}}(\text{M 켬})\le\rho\cdot\mathrm{Flip}_{\text{held-cal}}(\text{M 끔}),\qquad\rho=1.5
$$

### 5.4 전체 파이프라인

$$
\hat\theta=\mathrm{Recon}\Big(\mathrm{M}_{\alpha^{*}}\big(\mathrm{P}_{\mathcal S^{*}}(\mathrm{G}(\theta))\big)\Big),\qquad
\mathrm{Recon}\in\{\text{BRECQ},\ \text{QDrop}\}
$$

- 기존 재구성 PTQ(BRECQ, QDrop) 위에 얹는 plug-in이다.
- 모든 상수(예산 1.5%, M 검사 1.5배 규칙, 이전 강도 α 후보)는 모든 모델에 고정이다. 모델마다 다른 것은 진단 결과뿐이다.
- 추가 비용은 모델 크기 약 +0.3%이다. 연산 형태는 기준선과 같다.

## 6. 사용 시나리오

| 항목 | 내용 |
|---|---|
| 상황 | 사전학습된 OVOD 모델을 엣지 기기에 4bit로 배포 |
| 가진 것 | 모델, 라벨 없는 이미지 수백 장(도메인 무관), 예시 어휘 |
| 모르는 것 | 배포 후 사용할 어휘 |
| 제약 | 재학습 불가, 표준 per-tensor/per-channel 정수 커널 |
| 절차 | ① G 구조 변환 → ② flip 진단으로 P 보호 목록 결정 → ③ M 사전 검사로 켬/끔 결정 → ④ BRECQ·QDrop 등 기존 재구성 PTQ 실행 |
| 결과물 | 처음 보는 어휘에서도 판정이 덜 무너지는 4bit 모델 |

## 7. 결과 요약

**주 결과 (YOLO-World-S, 6 seed, BRECQ 대비)**

| 설정 | BRECQ LVIS AP / flip | GPM LVIS AP / flip |
|---|---|---|
| W4A8 | 22.74 / 17.6% | **23.55 / 12.3%** |
| W4A5 | 18.35 / 33.8% | **22.31 / 19.6%** |

- W4A5의 BRECQ는 seed 0에서 붕괴했다(COCO 19.3). 이 seed를 빼도 LVIS AP +2.5, flip −8.4%p로 Holm 보정 후 유의하다.

**일반화**
- **calibration 데이터 교체:** 양자화 과정 전체에서 COCO를 빼고 Objects365(+GQA)로 바꿨다. GPM 결과는 거의 같았다(W4A8 LVIS 23.6, W4A5 21.9~22.0). 보호 대상과 M 결정도 같았다. 3 seed × 두 비트 × 두 데이터, 12개 짝 모두에서 BRECQ보다 좋았다.
- **모델 규모:** YOLO-World-L W4A5에서 BRECQ 16.3 → GPM 29.5. QDrop 위에서도 W4A5는 1.4(붕괴) → 26.9로 개선됐다.
- **다른 도메인:** ODinW-13 평균 AP에서 YOLO-World-S는 W4A8 +1.5, W4A5 +3.3이다. YOLOE W4A5는 +8.0~9.7이다.
- **극저비트:** W4A4에서 BRECQ는 붕괴하고(LVIS 0.3) GPM은 유지된다(약 18).
- **다른 구조:** CNN(YOLO-World, YOLOE)과 ViT(OWLv2)에 같은 규칙을 적용했다. DETR 계열(Grounding DINO)은 실험을 진행하고 있다.

## 8. 한계와 남은 일

- **M:** YOLO-World-M에서는 AWQ가 우리 M보다 나았다. M의 이점 주장은 S 모델로 한정한다.
- **최신 PTQ 비교:** PD-Quant(CVPR 2023)를 공식 설정 그대로 옮겼으나, YOLO-World에서는 PD 손실과 분포 보정(DC) 모두 QDrop보다 성능을 떨어뜨렸다. 가중치를 바꿔도 QDrop을 넘는 설정이 없어 본 실험에서 제외했다(그 위에서도 GPM은 LVIS AP를 +2.0~16.8 올렸다). 최신 검출 PTQ(Reg-PTQ, InlierQ)는 closed-set 검출기용이고, 모두 BRECQ·QDrop을 기준선으로 쓴다. 그래서 재구성 계열의 대표인 BRECQ·QDrop 위에서 GPM이 독립적으로 동작함을 보이는 것으로 갈음한다.
- **ODinW:** YOLOE-v8s W4A8에서는 BRECQ보다 낮았다(−1.2). 이 하락은 thermalDogs 한 데이터셋에서 비롯됐고, 이 데이터셋을 빼면 차이가 없다.
- **실기기 지연:** 아직 측정하지 않았다.
- **진행 중:** Grounding DINO W4A5에서 P·M 적용 결과를 보고 있다. naive, BRECQ, QDrop 기준선은 완료했다.

---

### 참고
- QATMA: https://arxiv.org/abs/2603.05964
- YOLO-World: https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html
- Reg-PTQ: https://openaccess.thecvf.com/content/CVPR2024/html/Ding_Reg-PTQ_Regression-specialized_Post-training_Quantization_for_Fully_Quantized_Object_Detector_CVPR_2024_paper.html
- InlierQ: https://arxiv.org/abs/2602.03472
- YOLO-Jevois: https://cnx-software.com/2025/02/06/yolo-jevois-leverages-yolo-world-to-enable-open-vocabulary-object-detection-at-runtime-no-dataset-or-training-needed/
