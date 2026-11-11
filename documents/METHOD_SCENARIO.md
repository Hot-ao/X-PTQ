# 연구 시나리오: 개방 어휘 객체 검출기의 저비트 사후 양자화

> **11-11 seed 0 전환:** 논문 결과는 이제 seed 0 한 번 실행 값으로 쓴다(정수 가속기 실험과 같은 기준). 이 문서의 seed 평균·신뢰구간 수치는 작성 당시 기록으로 남긴다. 논문에 옮길 수치는 [RESULTS_FINAL_1seed.md](RESULTS_FINAL_1seed.md)(seed 0)와 [DRAFT_PATCH_2026-11-10.md](DRAFT_PATCH_2026-11-10.md)를 쓴다. 3 seed 평균은 [RESULTS_FINAL_3seed.md](RESULTS_FINAL_3seed.md)(부록용).

작성: 하태호 · 2026-10-30 · **갱신 2026-11-08**

**11-08 갱신 요약**
- **규칙 (11-02):**
  - P는 G 대상 conv를 먼저 빼고 예산을 다시 채운다.
  - M의 채널 통계는 calibration 전체(256장)에서 잰다.
  - M 사전 검사는 ρ = 1.0이다.
  - 검사는 seed 0에서 한 번 하고, 그 결정을 모든 seed에 쓴다(11-05).
- **이름:** 처방 이름을 초록과 맞췄다. Text-Gate Commutation(G), Diagnosis-guided Protection(P), Tied Scale Migration(M).
- **결과(§7):** 11-07에 끝난 새 규칙 3 seed 결과로 바꿨다. 8개 검출기, W4A4, segmentation, M 검사 정확도를 넣었다.
- **비용:** 크기 증가는 "+0.3%"가 아니라 **+1.3–1.6%**다(새 P 규칙).
- **한계(§8):** 새 규칙에서 드러난 점으로 바꿨다.
- 수치 출처: [RESULTS_2026-11-04.md](RESULTS_2026-11-04.md)(11-07 19:30판).

---

## 1. 배경: 개방 어휘 객체 검출(OVOD)의 확산

- 개방 어휘 객체 검출(open-vocabulary object detection, OVOD)은 학습 때 정해 둔 클래스가 아니라 실행 시점에 텍스트로 준 어휘를 검출하는 방식이다. 재학습 없이 새로운 대상을 찾을 수 있어 활용 범위가 넓다.
- 주요 학회에서 실시간 모델이 잇따라 나왔다. YOLO-World(CVPR 2024), YOLOE(ICCV 2025), Grounding DINO, OWLv2가 대표적이다.
- 실제 사용도 늘고 있다.
  - 공개 저장소 GitHub star: Grounding DINO 약 10.7k, YOLO-World 약 6.6k, YOLOE 약 2.3k (2026-10-30 기준)
  - YOLO-World와 YOLOE는 Ultralytics 패키지에 통합되어 있다.
  - 임베디드 카메라에서 YOLO-World로 어휘를 실행 중에 바꾸는 사례도 있다(YOLO-Jevois, 2025).
- 이런 모델을 엣지 기기에 올리려면 저비트 양자화가 필요하다. 재학습 없이 소량의 calibration 이미지로 양자화하는 사후 양자화(PTQ)가 현실적인 선택이다.

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

- 기존 PTQ(BRECQ, QDrop)는 calibration 데이터에서 층 출력의 재구성 오차를 줄이도록 양자화 파라미터를 맞춘다. 어떤 어휘로 쓰일지는 고려하지 않는다.
- OVOD에서는 양자화할 때 본 어휘와 배포 후에 쓸 어휘가 다르다. 실험에서는 양자화 때 COCO 80 클래스를 쓰고, 평가는 처음 보는 LVIS 1203 클래스로 했다.
- 실제로 양자화할 때 본 어휘의 AP는 크게 떨어지지 않는데, 처음 보는 어휘의 판정은 크게 무너진다.

| YOLO-World-S, 6 seed 평균 | COCO AP | LVIS AP | LVIS 판정 flip |
|---|---|---|---|
| FP32 | 36.80 | 25.89 | – |
| BRECQ W8A8 | 36.75 | 25.46 | 3.8% |
| BRECQ W4A8 | 33.77 (−3.0) | 22.74 (−3.2) | **17.6%** |

- 판정 flip은 FP 모델이 확신하는(점수 > 0.25) 위치 중에서 양자화 뒤 top-1 클래스가 바뀐 비율이다.
- W4A8에서 FP가 확신하던 예측의 약 6분의 1이 다른 클래스로 바뀐다. AP만 보면 이 손상이 잘 드러나지 않는다.
- 비트를 더 낮추면 손상이 커지고 AP까지 무너진다. W4A4에서는 6개 YOLO 계열 모델 모두 BRECQ가 LVIS AP 0.04–3.44, QDrop이 0.01–0.23이다(3 seed).
- BRECQ는 규칙과 무관한 기준선이라 6 seed 값을 그대로 쓴다. 새 결과표의 3 seed 값은 W4A8 22.71 / 17.7%로 같은 양상이다.

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
| **G** Text-Gate Commutation | 텍스트 게이트가 곱해진 분기가 다른 분기와 양자화 scale을 공유해서 오염된다 | 게이트 곱셈을 융합 1×1 conv 뒤로 옮긴다. FP 결과는 수학적으로 같고 추가 비트가 없다 | 구조로 정한다. 텍스트 게이트 분기를 1×1 conv가 받는 구조(YOLO-World의 T-CSPLayer)에만 적용한다. YOLOE, OWLv2, Grounding DINO에는 없다 |
| **P** Diagnosis-guided Protection | 판정 손상이 소수 층에 몰려 있다 | 층마다 W4로 내렸을 때 flip이 얼마나 느는지 calibration 이미지로 재고, 상위 층을 양자화 weight의 1.5%만큼 W8로 둔다 | 고정 예산, 자동 선택 |
| **M** Tied Scale Migration | activation 채널 사이의 범위 편차(outlier) | 같은 텐서를 받는 층들이 하나의 scale을 공유하도록 묶어서 이전한다. 표준 정수 커널로 배포할 수 있다 | M을 켠 모델과 끈 모델의 calibration 밖 flip을 비교해 고른다 |

### 5.1 G: Text-Gate Commutation (FP 등가 변환)

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
- 근거: YOLO-World-S 블록 12의 cv2 입력에서 분기별 activation 최대값은 $y_0$ 7.7, $y_1$ 7.0, $y_2$ 9.3, **$y_3$(게이트 분기) 20.4**다. weight 열 최대값도 게이트 분기가 3.32로 다른 분기(≤ 0.89)보다 크다.
- 텍스트 게이트 모델(YOLO-World 4종)에서는 진단 손상의 57–83%가 G 대상 cv2 4개에 있다.

### 5.2 P: 예산 아래 손상 위치 보호

naive W8A8 모델 $Q^{8}$에서 층 $\ell$ 하나만 W4로 내린 모델을 $Q^{8}_{\ell\to4}$로 둔다. calibration 이미지와 calibration 어휘(cal)로 판정 손상 증가량을 잰다(train2017 앞 200장, COCO 어휘).

$$
\Delta_\ell=\mathrm{Flip}_{\text{cal}}\big(Q^{8}_{\ell\to4}\big)-\mathrm{Flip}_{\text{cal}}\big(Q^{8}\big)
$$

G를 쓰는 모델에서는 G가 이미 다루는 층 집합 $\mathcal G$(T-CSPLayer의 cv2)를 후보에서 먼저 뺀다. 예산의 분모는 그대로 양자화 weight 전체 $|\theta|$다(11-02: G 뒤 다시 채움).

$$
\mathcal S^{*}=\arg\max_{\mathcal S\subseteq\mathcal L\setminus\mathcal G}\sum_{\ell\in\mathcal S}\Delta_\ell
\quad\text{s.t.}\quad\sum_{\ell\in\mathcal S}|\theta_\ell|\le B\,|\theta|,\qquad B=0.015
$$

- $\Delta_\ell$이 큰 층부터 담는 greedy로 푼다. 담은 양이 예산의 110%를 넘지 않게 하고, 90% 이상 차면 멈춘다.
- $\mathcal S^{*}$는 weight만 8bit로 둔다. activation 비트는 그대로다.
- 평가 이미지와 평가 어휘는 쓰지 않는다.
- 실제 보호량은 1.35–1.59%이고, 크기는 +1.3–1.6%다. YOLO-World-S는 9개 conv(Backbone 1·2·4·6 블록)를 보호한다.

**예산의 근거: 손상 집중도.** 손상이 고르게 퍼져 있다면 weight 1.5%는 손상의 1.5%만 담는다. 실제로는 다음과 같다.

| 모델 | 1.5% 예산이 담는 손상 |
|---|---|
| YOLO-World-S | 72% |
| YOLOE-v8s | 67% |
| YOLO-World-M | 55% |
| YOLO-World-S v2 | 53% |
| YOLOE-11s | 48% |
| YOLO-World-L | 38% (최민감 층 `12.cv2` 하나가 55%이고 채널이 넓어 예산에 들지 않는다. 대신 G가 다룬다) |
| OWLv2-B/16 | 35% (P 효과 없음) |
| Grounding DINO-T | 5% (P가 더하는 몫 +0.23) |

- 예산 sweep(S W4A5, 3 seed): GPM은 1% → 1.5%에서 +0.41, 1.5% → 3%에서 +0.02로 1.5%에서 꺾인다. GP는 예산과 무관하게 평평하다(21.7–22.1).
- 예산 1.5%는 09-29에 이 sweep을 보기 전에 정했다. "꺾이는 지점에서 골랐다"가 아니라 "고정 예산이 꺾이는 지점에 있었다"로 쓴다.
- 같은 예산에서 위치 선택이 중요하다. S W4A8에서 진단 기반 P는 +0.96, HAWQ식 보호는 +0.87, 무작위 보호는 +0.25다(seed 0).
- **집중도는 P 효과의 크기를 예측하지 못한다**(6개 모델 순위 상관 −0.17). 집중도는 "작은 예산으로 충분한 이유"와 "손상 위치가 구조를 따른다"의 근거로만 쓴다.

### 5.3 M: Tied Scale Migration과 사전 검사

기본 등가식과 $s$ 공식은 SmoothQuant 계열과 같다.

$$
Wx=\big(W\,\mathrm{diag}(s)\big)\big(\mathrm{diag}(s)^{-1}x\big),\qquad
s_c(\alpha)=\frac{\max|x_c|^{\alpha}}{\max|W_{:,c}|^{1-\alpha}}
$$

우리가 더한 것은 네 가지다.
- 채널 통계 $\max|x_c|$는 **calibration 이미지 전체(256장)**에서 잰다(11-02 규칙. 11-01까지는 α 탐색용 8장).
- α 탐색의 출력 오차는 calibration 앞 $N_m=8$장으로 잰다.

**(1) 공유 제약.** 배포에서 $x/s$는 생산 층의 requant 단계에서 만들어진다. 따라서 같은 생산 채널 그룹 $g$를 받는 모든 소비 층 $\mathcal C(g)$와, residual add로 합쳐지는 채널들은 하나의 $s$를 공유해야 한다. $s$ 공식의 $\max|x_c|$, $\max|W_{:,c}|$는 그룹의 모든 소비 층에 대한 최댓값으로 바꾼다.

$$
s_c=s_{g(c)},\qquad \max|x_c|\leftarrow\max_{j\in\mathcal C(g)}\widetilde x^{(j)}_c,\qquad \max|W_{:,c}|\leftarrow\max_{j\in\mathcal C(g)}\max\big|W^{(j)}_{:,c}\big|
$$

**(2) 채널 통계 하한.** 통계를 잰 이미지에서 거의 0이던 채널은 다른 이미지에서 켜질 수 있다. 그 채널은 $s$가 극단적으로 작아져 $x/s$가 폭주한다(ReLU 뒤 희소 채널, `runs/217_mig_rule`). 그래서 소비 층 $j$마다 채널 통계를 그 입력의 최대 채널값 대비 $\tau$ 아래로 내리지 않는다.

$$
\widetilde x^{(j)}_c=\max\!\Big(\max\big|x^{(j)}_c\big|,\ \tau\max_{c'}\max\big|x^{(j)}_{c'}\big|\Big),\qquad\tau=0.003
$$

규칙은 모든 모델에 같다.
- 트랜스포머(OWLv2, Grounding DINO)는 한 단위의 소비 Linear들이 같은 입력 텐서를 받는다. 그래서 "소비 층 입력 최댓값"이 곧 단위 최댓값이다.
- YOLO 계열은 소비 conv가 여러 생산자 출력의 concat을 받는다. 그래서 그 concat 전체의 최댓값이 기준이 된다.

**(3) "이전 없음"을 포함한 α 선택.** 단위 $U$(add로 묶인 생산자 집합)마다, $U$의 채널을 받는 모든 소비 층 $\mathcal C(U)$의 상대 출력 오차 합을 최소화하는 $\alpha$를 고른다. $\varnothing$은 $s=1$(이전 없음)이다. 따라서 이 대리 오차 기준으로 $s=1$보다 나쁜 $\alpha$는 고르지 않는다(최종 지표까지 보장하지는 않는다).

$$
\alpha^{*}_U=\arg\min_{\alpha\in\{0,\,0.25,\,0.5,\,0.75,\,1\}\cup\{\varnothing\}}
\sum_{j\in\mathcal C(U)}\frac{\big\|Q_w\big(W_j\,\mathrm{diag}(s)\big)\,Q_a\big(x_j/s\big)-W_jx_j\big\|^2}{\|W_jx_j\|^2}
$$

단위는 순서대로 하나씩 정한다. 한 소비 층이 여러 단위의 채널을 받으면(concat), 앞에서 정한 단위의 $s$는 고정하고 아직 정하지 않은 단위의 채널은 $s=1$로 둔다.

**(4) 사전 검사.** M을 켠 모델과 끈 모델을 둘 다 재구성(BRECQ 또는 QDrop)까지 만든 뒤, calibration에 쓰지 않은 이미지(held-cal: train2017에서 calibration 다음 200장, COCO 어휘)에서 판정 손상을 비교한다.

$$
\text{M 사용}\iff\mathrm{Flip}_{\text{held-cal}}(\text{M 켬})\le\rho\cdot\mathrm{Flip}_{\text{held-cal}}(\text{M 끔}),\qquad\rho=1.0
$$

- ρ는 11-02에 1.5에서 1.0으로 바꿨다. 1.1–1.4배 구간에서 M이 손해인 경우를 놓쳤기 때문이다(YOLOE-v8s W4A8, S W4A8).
- 검사는 모델 × 비트 설정마다 seed 0에서 한 번 하고, 그 결정을 모든 seed와 QDrop 위의 실행에 같이 쓴다(`configs/mcheck_cache`, 11-05 규칙).
- **정확도 (73건, BRECQ 62 + QDrop 11):**
  - LVIS AP가 더 높은 쪽을 고른 경우가 60건이다.
  - |ΔAP| ≥ 1인 35건은 모두 맞혔다.
  - 틀린 13건의 손해는 최대 0.81 AP다.
  - 반대로 구성을 하나로 고정하면 최대 10.51(항상 GPM)–22.52(항상 GP)를 잃는다.
- **비용:** 검사 자체는 약 20초다. 두 후보를 모두 만드는 비용은 BRECQ 단독의 약 2.4–2.9배다.

**구현 세부(부록).**
- $s$는 단위 안 기하평균으로 나눠 정규화한다. per-channel weight 양자화와 per-tensor activation 양자화에서는 결과가 같고, 수치 안정용이다.
- 계보가 없는 입력 채널은 $s=1$이다. stem의 이미지 입력, T-CSPLayer 출력처럼 사용자 정의 forward나 비선형 경로로만 이어진 채널이 여기에 해당한다.
- C2fAttn의 attention은 입력 $x$를 그대로 쓴다. 그래서 배포 시 그 앞에서 $x\cdot s$로 되돌린다(`quant/channel_graph.py::check_deploy_equivalence`).

### 5.4 전체 파이프라인

M을 켠 후보와 끈 후보를 둘 다 재구성한 뒤, §5.3 (4)의 사전 검사로 하나를 고른다.

$$
\hat\theta_{\text{on}}=\mathrm{Recon}\Big(\mathrm{M}_{\alpha^{*}}\big(\mathrm{P}_{\mathcal S^{*}}(\mathrm{G}(\theta))\big)\Big),\qquad
\hat\theta_{\text{off}}=\mathrm{Recon}\Big(\mathrm{P}_{\mathcal S^{*}}\big(\mathrm{G}(\theta)\big)\Big),\qquad
\mathrm{Recon}\in\{\text{BRECQ},\ \text{QDrop}\}
$$

$$
\hat\theta=\begin{cases}\hat\theta_{\text{on}} & \mathrm{Flip}_{\text{held-cal}}(\hat\theta_{\text{on}})\le\rho\,\mathrm{Flip}_{\text{held-cal}}(\hat\theta_{\text{off}})\\[2pt] \hat\theta_{\text{off}} & \text{otherwise}\end{cases},\qquad\rho=1.0
$$

- G가 적용되지 않는 모델(YOLOE, OWLv2, Grounding DINO)에서는 $\mathrm{G}$가 항등 변환이다.
- 검사는 모델 × 비트 설정마다 seed 0에서 한 번 하고, 그 결정을 모든 seed와 QDrop 위의 실행에 같이 쓴다.
- 이 선택은 YOLO 계열 6개 모델에만 썼다. OWLv2와 Grounding DINO는 처방을 고정했다(OWLv2: W8A8 M, W4A8 PM / GDINO: PM). 검사를 쓰지 않았으므로 본문에 그렇게 밝힌다.

- 기존 재구성 PTQ(BRECQ, QDrop) 위에 얹는 plug-in이다. 추가 학습은 없다.
- 모든 상수는 모든 모델에 고정이다: 예산 1.5%, M 검사 ρ = 1.0, 이전 강도 α 후보, 통계 하한 0.3%. 모델마다 다른 것은 진단 결과와 검사 결과뿐이다.
- **추가 비용:**
  - 모델 크기 +1.3–1.6%다(평균 4.05–4.06 bit). P만 크기를 늘리고 G와 M은 늘리지 않는다.
  - 연산 형태는 기준선과 같다. G는 1×1 conv 두 개, 위치별 곱, 덧셈이 된다. M은 생산 쪽 채널별 곱 1회다.

## 6. 사용 시나리오

| 항목 | 내용 |
|---|---|
| 상황 | 사전학습된 OVOD 모델을 엣지 기기에 4bit로 배포 |
| 가진 것 | 모델, 라벨 없는 이미지 수백 장(도메인 무관), 예시 어휘 |
| 모르는 것 | 배포 후 사용할 어휘 |
| 제약 | 재학습 불가, 표준 per-tensor/per-channel 정수 커널 |
| 절차 | ① flip 진단(모델당 1회, YOLO s 약 10분) → ② G 구조 변환(해당 구조일 때) → ③ 진단으로 P 보호 목록 결정 → ④ M을 켠 후보와 끈 후보를 BRECQ·QDrop 등 기존 재구성 PTQ로 만듦 → ⑤ calibration 밖 flip 검사로 둘 중 하나 선택 |
| 결과물 | 처음 보는 어휘에서도 판정이 덜 무너지는 4bit 모델 (크기 +1.3–1.6%) |

## 7. 결과 요약 (새 규칙, 11-07)

**주 결과 (YOLO-World-S, 3 seed, LVIS AP / flip)**

| 설정 | BRECQ | Ours | Δ AP (95% CI) | QDrop | QDrop + Ours |
|---|---|---|---|---|---|
| W8A8 | 25.49 / 3.8% | **25.71 / 2.7%** (GPM) | +0.22 ± 0.12 | 25.38 / 4.0% | **25.79 / 2.7%** |
| W4A8 | 22.71 / 17.7% | **24.20 / 10.7%** (GPM) | +1.49 ± 0.74 | 22.45 / 18.5% | **24.48 / 11.1%** |
| W4A5 | 16.97 / 39.4% † | **22.98 / 16.3%** (GPM) | +6.01 ± 12.14 | 12.69 / 55.6% | **22.35 / 17.8%** |
| W4A4 | 0.43 / 97.1% | **18.09 / 32.9%** (GPM) | +17.66 ± 1.06 | 0.23 / 98.0% | **13.95 / 42.7%** |

- † W4A5의 BRECQ는 seed 0에서 붕괴했다(11.29). seed 1·2만 보면 BRECQ 19.81, Ours 23.00이다.

**6개 실시간 모델 (YOLO-World S/S-v2/M/L, YOLOE-v8s/11s, 3 seed)**
- **W4 전 칸(W4A8·W4A5·W4A4, 18칸)에서 BRECQ보다 AP가 높고 flip이 낮다.** QDrop 위에서도 같다.
  - 신뢰구간이 0을 포함하는 칸은 S W4A5(†)와 YOLOE-11s W4A8(+0.18 ± 0.91) 두 곳이다.
  - flip 예: L W4A8은 23.5% → 9.7%, YOLOE-v8s W4A5는 32.7% → 16.2%다.
- **W4A4가 가장 강한 결과다.**
  - BRECQ(0.04–3.44)와 QDrop(0.01–0.23)이 무너질 때 Ours는 10.90–23.43이고, 신뢰구간이 모두 0을 넘는다.
  - 검사는 6개 모델 모두 GPM(YOLOE는 PM)을 골랐다. GP만으로는 0.08–5.03에 그친다. **W4A4는 M이 있어야 열린다.**
- **모델 규모:** YOLO-World-L W4A5에서 BRECQ 16.25 → Ours 29.89(+13.65 ± 0.91)다. QDrop 위에서는 1.44 → 26.87이다.
- **W8A8은 "유지" 수준이다.** 신뢰구간이 0을 넘는 이득은 S(+0.22)뿐이고, 나머지는 −0.31 ~ +0.17이다.

**ViT·DETR 계열 (seed 0, 처방 고정)**
- **OWLv2-B/16:** W8A8 M +1.67, W4A8 PM +2.06이다. P 단독은 효과가 없다(손상 집중도 35%).
- **Grounding DINO-T:** W4A5, LVIS 앞 2,000장에서 BRECQ 27.26, QDrop 29.23, PM 30.23(FP 30.06)이다. flip은 다른 모델과 정의가 다른 측정으로만 있어 보조 근거로만 쓴다(§8).

**일반화 (seed 0)**
- **다른 도메인:** ODinW-13 평균 AP에서 W4A5는 6개 모델 모두 +8.8 ~ +26.0이다. W4A8은 YOLO-World 4종이 +2.1 ~ +6.5이고, YOLOE 두 모델은 −1.20 / −0.12로 BRECQ와 같거나 약간 낮다.
- **calibration 데이터 교체:** COCO 대신 Objects365(+GQA)로 calibration해도 이득이 유지된다(W4A8 +1.03 ~ +1.73, W4A5 +3.68 ~ +4.31). 검사 어휘는 COCO-80이다.
- **Instance segmentation:** mask 분기까지 양자화한 LVIS mask AP에서 이득이 유지된다. W4A8은 +0.6 ~ +1.2(3 seed 같은 방향), W4A5는 +2.7 ~ +5.1이다(YOLOE-v8s/11s-seg, YOLO-World-Seg-M).
- **첫·마지막 레이어 FP:** W4A4에서 첫·마지막 레이어를 FP로 두어도 BRECQ는 회복되지 않는다(S 0.42, M 0.76). 따라서 W4A4 붕괴는 첫·마지막 레이어 때문이 아니다.

## 8. 한계와 남은 일

**한계 (본문에 적는다)**
- **M 검사가 놓치는 경우:**
  - S W4A8은 3 seed 모두 GPM을 골랐지만 GP가 0.26–0.60 높다. 표 1에 GP와 GPM을 둘 다 싣는다.
  - YOLOE-11s W4A8도 검사와 AP가 엇갈린다(0.17–0.77).
  - 틀린 경우의 손해는 실측으로 0.81 이내였다.
- **v2 W8A8:** M 때문에 flip이 5.0% → 7.9%로 나빠진다(QDrop 위에서는 AP −0.22로 유의). L W8A8도 −0.31 ± 0.72다. W8A8은 부록에 두고 한계로 적는다.
- **v2에서 GPM이 PM보다 나쁘다:** W4A8 3 seed 평균 GPM 19.05, seed 0 PM 22.45. 원인은 v2에서 G와 함께 쓰는 xmax 256장 통계로 보인다. 진단 선택의 한계로 보고할지, v2 표에 PM을 함께 실을지 교수님과 정한다.
- **W4A4:** 우리 방법은 열리지만 FP에 못 미친다. v2(10.90 / FP 22.19)와 M(14.95 / FP 29.80)은 절반 정도까지만 돌아온다. per-tensor A4가 여전히 절벽이다.
- **YOLOE의 W4A8 이득이 작다**(+0.18 ~ +0.87). 텍스트 게이트가 없어 G가 적용되지 않고 P만 남기 때문이다. ODinW W4A8에서는 BRECQ와 같거나 약간 낮다. 이전 규칙 결과에서는 YOLOE-v8s의 하락이 thermalDogs 한 데이터셋에서 비롯됐는데, 새 결과로는 다시 확인하지 않았다.
- **Grounding DINO:** W4A5, seed 1개, LVIS 2,000장 subset만 있다.
  - flip은 `lvis_gtflip_c`(FP가 맞히던 GT의 top-1 변화)로만 쟀다. 다른 모델과 정의가 다르다.
  - PM은 BRECQ보다 4.7%p 낮지만, QDrop과의 차이는 표준오차 안이다.
- **OWLv2:** Ours가 seed 0뿐이고, W4A8 절대 성능이 낮다(ViT 전용 양자화기는 범위 밖).
- **실기기 지연:** 측정하지 않았다. 연산 형태는 기준선과 같고 크기는 +1.3–1.6%다.

**기준선 관련 (리뷰 대비)**
- **QDrop이 원 논문과 달리 BRECQ보다 약하다.** A 비트가 낮을수록 격차가 커진다. 모든 방법을 2,000 iters로 맞추고 학습률은 공식값을 썼다. M W4A5에서 20,000 iters로 늘려도 QDrop은 BRECQ보다 낮았고, 우리 처방의 이득(+4.0)은 유지됐다. 반복 수 부족 탓이 아니다.
- **PD-Quant(CVPR 2023):** 공식 설정 그대로 옮겼으나, YOLO-World에서는 PD 손실과 분포 보정(DC) 모두 QDrop보다 성능을 떨어뜨렸다. 가중치를 바꿔도 QDrop을 넘는 설정이 없어 본 실험에서 제외했다(그 위에서도 GPM은 LVIS AP를 +2.0~16.8 올렸다, 기존 규칙). 부록이나 rebuttal에서만 언급한다.
- **최신 검출 PTQ(Reg-PTQ, InlierQ):** closed-set 검출기용이고, 모두 BRECQ·QDrop을 기준선으로 쓴다. 그래서 재구성 계열의 대표인 BRECQ·QDrop 위에서 GPM이 독립적으로 동작함을 보이는 것으로 갈음한다.
- **M 대신 AWQ/SmoothQuant:** S에서는 GPS / GPW가 GPM과 비슷하다(W4A8 seed 0, BRECQ 대비 +1.17 / +1.24 vs GPM +1.22). YOLO-World-M에서는 AWQ가 M보다 나았는데, 이것은 기존 규칙 결과이고 새 규칙으로 다시 돌리지 않았다. M의 이점은 주장하지 않고, "이전 규칙은 바꿔 써도 된다"로 쓴다.

**남은 일 (모두 선택 사항, 11-07 기준 돌고 있는 실험 없음)**
- W4A6 seed 1·2: 비트 곡선 그림에 오차 막대를 넣을 때만 필요하다.
- GDINO 메인 비트(W4A8 / W4A4)와 seed 1·2
- OWLv2 Ours seed 1·2
- AdaRound W4A6 이하: 채울지, 표에서 뺄지 정한다.
- M 검사 그림을 73건으로 다시 그리기

---

### 참고
- QATMA: https://arxiv.org/abs/2603.05964
- YOLO-World: https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html
- Reg-PTQ: https://openaccess.thecvf.com/content/CVPR2024/html/Ding_Reg-PTQ_Regression-specialized_Post-training_Quantization_for_Fully_Quantized_Object_Detector_CVPR_2024_paper.html
- InlierQ: https://arxiv.org/abs/2602.03472
- YOLO-Jevois: https://cnx-software.com/2025/02/06/yolo-jevois-leverages-yolo-world-to-enable-open-vocabulary-object-detection-at-runtime-no-dataset-or-training-needed/
