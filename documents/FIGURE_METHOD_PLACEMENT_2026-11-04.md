# 방법 그림 편집용: 각 기법이 YOLO-World의 어디에, 어떻게 들어가는가 (2026-11-04)

**대상 독자:** 방법 그림([figures/method_arch.png](../figures/method_arch.png), 코드 [figures/src/make_method_arch.py](../figures/src/make_method_arch.py))을 고치는 사람.
YOLO-World 논문의 Fig. 3(전체 구조)과 Fig. 4(RepVL-PAN)를 기준 좌표로 삼아, 우리 단계 D / G / P / M / R과 양자화 프로토콜이 각각 어느 모듈에서 무엇을 바꾸는지 적는다.

**기준:**
- 모델: YOLO-World-S(`yolov8s-world.pt`, v1). 모듈 번호는 ultralytics의 `model.model[i]`다.
- 규칙: 11-02 새 규칙이다.
  - P는 G 대상을 빼고 예산을 다시 채운다.
  - M의 xmax는 calibration 256장으로 잰다.
  - M 사전 검사는 ρ = 1.0이다.
- 형태 수치: 11-04에 모델을 직접 읽어 확인했다(§1.3).
- 근거 문서: [docs/METHOD.md](../docs/METHOD.md) §2–4, [PAPER_OUTLINE_2026-11-03.md](PAPER_OUTLINE_2026-11-03.md) §0.

---

## 0. 한 장 요약

| 단계 | YOLO-World 논문의 위치 (Fig. 3 / Fig. 4 용어) | 모듈 (S) | 하는 일 | 언제 | 배포 모델에 남는 것 |
|---|---|---|---|---|---|
| **D** 진단 | 모델 밖. PTQ 시점의 오프라인 절차 | 양자화 대상 conv 63개를 하나씩 | conv 하나만 W4로 내리고 calibration 쪽 COCO top-1 flip 증가를 잰다. G/P/M을 어디에 쓸지 정한다 | 빌드 전, 모델당 1회 | 없음 |
| **G** 게이트 교환 | RepVL-PAN 안의 **T-CSPLayer 4개**. Concat 뒤의 1×1 conv(Fig. 4에는 그려져 있지 않음) | `12, 15, 19, 22`의 `cv2` (+ `attn`의 게이트) | 텍스트 게이트를 1×1 conv **뒤로** 옮긴다(FP 등가). 텍스트 분기가 전용 conv와 양자화기를 가진다 | 빌드 첫 단계(양자화기 삽입 전) | 1×1 conv 2개 + 위치별 곱 + 덧셈. 추가 비트 0 |
| **P** 보호 | **YOLO Backbone 앞쪽**: P2 단계(Conv, C2f)와 C3·C4를 만드는 C2f | `1`, `2`(5개 conv), `4`(3개), `6`(1개) = 9 conv | 진단 상위 conv의 weight를 W8로 둔다(예산 = 양자화 weight의 1.5%) | 양자화기 삽입 직후, calibrate 전 | 해당 conv만 INT8 weight. 크기 +1.6% |
| **M** 공유 scale 이전 | **conv에서 conv로 가는 모든 연결**(Backbone, PAN 안, Head 안). 단, T-CSPLayer 출력 뒤와 이미지 입력은 제외 | 생산 Conv 61개 → 소비 conv | 채널별 s를 생산 conv의 requant에 넣고(÷s) 모든 소비 conv weight에 곱한다(×s) | calibrate 전. 켤지 말지는 사전 검사가 정한다 | 생산 쪽 채널별 곱 1회(fusion 가능) |
| **R** 재구성 | 양자화된 검출기 전체(Backbone, RepVL-PAN, Head) | 모든 QuantConv2d | BRECQ 또는 QDrop. 우리가 바꾸지 않는다 | 마지막 | 학습된 rounding이 반영된 정적 weight |
| (프로토콜) | Text Encoder만 FP. 나머지는 양자화 | – | §6 | – | – |

그림에서 꼭 지킬 것 세 가지는 다음과 같다.
1. **G는 T-CSPLayer에만** 붙인다. I-Pooling Attention(16)에는 붙이지 않는다.
2. **P는 백본 앞쪽 단계에만** 붙인다. C5와 SPPF, 즉 백본 뒤쪽에는 붙이지 않는다.
3. **D는 모델 밖**(점선 영역)에 둔다. 배포 모델에는 없다.

---

## 1. 기준 좌표: YOLO-World 논문 그림 ↔ 실제 모듈

### 1.1 Fig. 3 (Overall Architecture)

| Fig. 3의 상자 | 실제 모듈 (YOLOv8s-World) | 비고 |
|---|---|---|
| Text Encoder (❄, frozen) | CLIP 텍스트 인코더 | 오프라인 FP. 어휘가 정해지면 임베딩을 미리 계산한다 |
| Vocabulary Embeddings | 텍스트 임베딩 t_j | PAN의 guide와 Head의 classifier 두 곳에 쓰인다 |
| Input Image | 640×640 | |
| YOLO Backbone | `0`–`9`: Conv0, Conv1, C2f2, Conv3, C2f4, Conv5, C2f6, Conv7, C2f8, SPPF9 | |
| Multi-scale Image Features | C3 = `4`의 출력(stride 8), C4 = `6`(stride 16), C5 = `9`(stride 32) | Fig. 4의 C3/C4/C5 |
| Vision-Language PAN | `10`–`22` (RepVL-PAN) | §1.2 |
| (PAN → 위) Image-aware Embeddings | `16` ImagePoolingAttn이 갱신한 텍스트 임베딩 | v1만 있다. 갱신된 임베딩은 `19`, `22`의 guide로만 쓰인다 |
| Text Contrastive Head | `23.cv3[l]`(Conv → Conv → 1×1, 512차원 region 임베딩) + `23.cv4[l]` ContrastiveHead | Fig. 3의 Object Embeddings = `cv3` 출력 |
| Region-Text Matching | `cv4`: τ·cos(x, t_j) + b. 영역×텍스트 matmul이고 conv가 없다 | Head는 **원래 텍스트 임베딩**을 쓴다(갱신본이 아님) |
| Box Head | `23.cv2[l]`(Conv → Conv → 1×1) + DFL | |

### 1.2 Fig. 4 (RepVL-PAN)

| Fig. 4의 요소 | 실제 모듈 | 비고 |
|---|---|---|
| C5 → (×2 up) → T-CSPLayer at C4 | `10` Upsample → `11` Concat(↑C5, C4) → **`12` C2fAttn** | 텍스트 손상이 가장 큰 곳(12.cv2) |
| → (×2 up) → T-CSPLayer at C3 | `13` Upsample → `14` Concat(↑, C3) → **`15` C2fAttn** | 출력 = P3 |
| I-Pooling Attention (첫 번째) | **`16` ImagePoolingAttn** (입력 15, 12, 9) | 이미지 → 텍스트 갱신 |
| (×½) → T-CSPLayer at P4 | `17` Conv(3×3, s2) → `18` Concat(↓, 12) → **`19` C2fAttn** | 출력 = P4 |
| (×½) → T-CSPLayer at P5 | `20` Conv(3×3, s2) → `21` Concat(↓, 9) → **`22` C2fAttn** | 출력 = P5 |
| I-Pooling Attention (두 번째, 오른쪽 끝) | **ultralytics v1에는 없다** | Fig. 4를 그대로 따라 그리면 실제 모델과 달라진다 |
| T-CSPLayer 내부: Split → Dark Bottleneck → Max-Sigmoid → Concat | `cv1`(1×1) → chunk → `m.0` Bottleneck → `attn`(MaxSigmoidAttnBlock) → concat | **Concat 뒤의 1×1 `cv2`는 Fig. 4에 그려져 있지 않다.** G가 고치는 곳이 바로 이 conv라서, 우리 그림에는 반드시 넣어야 한다 |
| Max-Sigmoid 내부 | 게이트 aw = σ(maxⱼ⟨y₂, guideⱼ⟩/√d + b)·scale (head별), 값 p = `proj_conv`(y₂) **3×3 conv**, 출력 y₃ = p ⊙ aw | Fig. 4는 proj_conv를 그리지 않는다 |

**변형 모델:** v2(`yolov8s-worldv2`)에는 ImagePoolingAttn이 없고, C2fAttn은 `12, 15, 18, 21`이다. M과 L은 S와 번호가 같고 채널만 넓다.

### 1.3 T-CSPLayer 4곳의 형태 (S, 11-04 확인)

| 블록 | 위치 | cv1 | cv2 입력 = y₀+y₁+y₂+y₃ | 텍스트 head 수 (×32ch) | proj_conv |
|---|---|---|---|---|---|
| 12 | top-down, C4 | 768→256 | 512 = 128×4 | 4 | 3×3, 128→128 |
| 15 | top-down, C3 → P3 | 384→128 | 256 = 64×4 | 2 | 3×3, 64→64 |
| 19 | bottom-up → P4 | 384→256 | 512 = 128×4 | 4 | 3×3, 128→128 |
| 22 | bottom-up → P5 | 768→512 | 1024 = 256×4 | 8 | 3×3, 256→256 |

---

## 2. D: 누수 없는 진단

**위치:** 모델 밖이다. PTQ를 만드는 서버에서 한 번 돌리는 오프라인 절차다. 그림에서는 YOLO-World Fig. 3의 "Training: Online Vocabulary" 점선 영역 자리, 즉 검출기 바깥에 두는 것이 맞다.

**방법** (`diag_w4_sensitivity.py::rank_convs_leakfree`)
1. calibration 이미지(train2017 앞 200장)에 FP 모델을 돌리고 COCO 점수를 저장한다.
2. naive W8A8 모델에서 **conv 하나만** weight를 W4로 내리고 다시 돌린다. 프로토콜은 본 실험과 같다(attn_cls, A16).
3. 그 conv의 손상 = FP가 확신(σ > 0.25)하는 anchor에서 **COCO top-1 클래스가 바뀐 비율의 증가량(pp)**이다.
4. 양자화 대상 conv 63개 모두에 반복한다. 첫·마지막 W8 conv 7개는 제외한다.

**누수 방지:** val2017을 쓰지 않는다. LVIS 어휘는 로드하지 않는다. 진단 전후로 RNG를 저장하고 복원한다.

**결과가 정하는 것 (S)**

| 진단 결과 | 처방 |
|---|---|
| 1위 `12.cv2` 30.8 pp. 텍스트 게이트 출력을 받는 T-CSPLayer의 1×1 | **G**. 같은 종류인 15/19/22.cv2도 함께 처리된다 |
| 나머지 상위: `1` 4.8, `2.cv2` 3.2, `4.cv1` 2.6, `4.cv2` 2.2, `2.m.0.cv1` 1.7, … | **P** (예산 1.5%, §4) |
| (M은 진단 막대가 아니라 사전 검사로 정한다) | **M** 켬/끔 (§5.4) |

**그림 표현:** 층별 막대(실측)에 G 대상은 초록, P 선택은 보라, 나머지는 회색. 지금 그림과 같다. "1.5% of the weights hold X% of the damage"를 넣으려면 S는 72%다(concentration_curve와 같은 수치).

---

## 3. G: 게이트 교환 (T-CSPLayer 안)

**위치:** RepVL-PAN의 T-CSPLayer 4곳(`12, 15, 19, 22`)이다. 각 블록에서 **Concat 직후의 1×1 conv `cv2`**와 Max-Sigmoid의 게이트 위치를 바꾼다.
- I-Pooling Attention(16)은 건드리지 않는다.
- Head의 Contrastive도 건드리지 않는다.

**원래 구조와 문제**
```
y3 = proj_conv(y2) ⊙ aw          # Max-Sigmoid 출력: 값 p에 위치별·head별 게이트를 곱함
out = SiLU( cv2( concat(y0, y1, y2, y3) ) )   # cv2 하나 = activation 양자화기 하나, weight scale도 출력 채널마다 공유
```
- 12.cv2 입력의 분기별 activation 최대값은 y₀ 7.7, y₁ 7.0, y₂ 9.3, **y₃ 20.4**다.
- y₃를 받는 weight 열의 최대값도 3.32로, 다른 분기(≤ 0.89)보다 크다.
- 그래서 **게이트가 곱해진 텍스트 분기 하나가 네 분기의 공유 activation 범위와 weight 범위를 정한다.** 이것이 "공유 scale 오염"이다.

**바꾼 구조** (`quant/fusion_quant.py::gate_commute`)
```
cv2(concat(y0, y1, y2, p ⊙ aw)) = W_main · concat(y0, y1, y2) + b  +  Σ_h aw_h ⊙ (W_h · p_h)     # FP에서 정확히 같다
```
- `cv2`의 weight 열을 둘로 나눈다.
  - `cv2_main`: y₀·y₁·y₂ 열과 bias.
  - `cv2_side[h]`: 텍스트 head h의 32ch 열, bias 없음. S는 12/19에 4개, 15에 2개, 22에 8개다.
- 게이트 aw는 원래 식 그대로 계산한다. p는 **게이트 없이** `cv2_side`에 들어가고, 그 출력에 aw를 곱해 `cv2_main` 출력에 더한다. 마지막에 SiLU를 적용한다.
- 성립 이유: 1×1 conv는 선형이고 aw는 위치별(그리고 head별) 스칼라다.

**양자화에서 달라지는 것**
1. 텍스트 분기가 **자기 activation 양자화기와 weight scale**을 가진다(quantizer A ≠ B).
2. 양자화되는 입력이 게이트를 곱하기 **전**의 p다. 게이트가 만드는 위치별 크기 편차가 범위에서 빠진다.
3. 게이트가 작은 위치(텍스트와 무관한 영역)에서는 텍스트 분기의 양자화 오차도 게이트 배율만큼 줄어든다.

**비용:** 추가 비트가 없고 FP 연산량도 같다. 배포 형태는 1×1 conv 두 개, 위치별 곱, 덧셈이다.

**다른 단계와의 관계**
- P: G가 처리하는 `*.cv2`는 보호 후보에서 **먼저 빼고** 예산을 다시 채운다.
- M: T-CSPLayer 출력은 사용자 정의 forward에서 나와 채널 계보가 끊긴다. 그래서 그 출력을 받는 소비 conv는 s = 1이다(§5.3).
- R: 새 conv(`cv2_main`, `cv2_side`)는 일반 conv처럼 AdaRound/LSQ를 받는다. FP 기준 모델도 같은 구조로 바꿔 넘긴다.
- 프로토콜: 게이트 aw 자체도 8bit로 양자화된다(attn_cls). 교환 전후로 같은 지점이다.

**그림 표현**
- Fig. 3 수준: PAN 상자에 G 배지를 붙이고, 가능하면 PAN 안에 T-CSPLayer 4개를 보이게 한다. 지금 그림은 PAN을 하나의 노란 기둥으로 그렸다. Fig. 4처럼 T-CSPLayer 4개와 I-Pooling 1개를 기둥 안에 작게 나누면, "G는 T-CSPLayer에만"이 정확히 보인다.
- 펼친 그림: Fig. 4의 "T-CSPLayer (C2f Block)" 상자 문법(S / Dark Bottleneck / Max-Sigmoid / C)을 쓰되 **C 뒤에 cv2를 꼭 그린다.** 원래 구조와 G 적용 후를 나란히 놓는다.
  - 원래 구조: Max-Sigmoid 출력 y₃ = p ⊙ a가 C로 들어가고, cv2 하나에 양자화기가 하나다.
  - G 적용 후: Dark Bottleneck 출력 y₂에서 p(proj_conv, 3×3)가 cv2_side(quantizer B)로 가고, 그 뒤 ⊙ a를 거쳐 cv2_main(quantizer A) 출력과 ⊕로 합쳐진 뒤 SiLU로 간다.
  - 지금 그림의 단순화: p를 y₂에서 바로 뽑았다. proj_conv(3×3)를 생략한 것이다. 정확히 하려면 y₂ → proj_conv → p로 그린다. 게이트 a도 y₂와 Text 둘 다에서 계산되므로, y₂ → gate 화살표를 하나 더 넣으면 정확하다.

---

## 4. P: 보호 (Backbone 앞쪽)

**위치:** YOLO Backbone의 앞쪽 단계다. Fig. 3의 "YOLO Backbone" 상자 안, Fig. 4 기준으로는 C3·C4를 만드는 C2f와 그 앞이다. RepVL-PAN과 Head에는 S에서 P가 하나도 없다.

**S에서 실제로 W8로 남는 9개 conv** (새 규칙, 크기 +1.6%, 평균 4.06 bit)

| conv | 모듈 (YOLO-World 용어) | 파라미터 (M) | 진단 손상 (pp) |
|---|---|---|---|
| `1.conv` | Backbone Conv1 (P2/4로 내리는 3×3, stride 2) | 0.018 | 4.84 |
| `2.cv1` | C2f2 입력 1×1 (Split 앞) | 0.004 | 0.63 |
| `2.cv2` | C2f2 Concat 뒤 1×1 | 0.006 | 3.15 |
| `2.m.0.cv1` | C2f2 Bottleneck 첫 3×3 | 0.009 | 1.73 |
| `2.m.0.cv2` | C2f2 Bottleneck 둘째 3×3 | 0.009 | 1.26 |
| `4.cv1` | C2f4 입력 1×1 (출력 = **C3**) | 0.016 | 2.56 |
| `4.cv2` | C2f4 Concat 뒤 1×1 | 0.033 | 2.24 |
| `4.m.1.cv1` | C2f4 둘째 Bottleneck 첫 3×3 | 0.037 | 0.71 |
| `6.cv1` | C2f6 입력 1×1 (출력 = **C4**) | 0.066 | 0.55 |

합은 0.199M이고, 양자화 weight 12.46M의 1.59%다.

**방법**
- 진단 손상이 큰 순서로 담는다. G 대상 `12/15/19/22.cv2`는 먼저 뺀다.
- 담은 파라미터 합이 1.5%의 110%를 넘지 않게 하고, 90% 이상 차면 멈춘다. 분모는 양자화 weight 전체다.
- 담은 conv는 **weight 비트만** 8로 바꾼다(`set_conv_wbits`). activation 비트는 그대로(A4–A8)이고, calibrate 전에 적용해 scale이 8bit 기준으로 잡힌다.

**그림 표현**
- 지금 그림은 Backbone 상자 전체에 P 배지 하나를 붙였다. 이러면 "백본 전체가 W8"로 읽힐 수 있다.
- 권장: Backbone 상자 안을 단계별로 쪼갠다(Conv1 · C2f2 · C2f4 · C2f6 · … · SPPF). P 배지는 **1, 2, 4, 6**에만 붙이고, 7·8·9(C5 쪽)는 그대로 둔다. 기존 [method_overview.png](../figures/method_overview.png)의 (a)가 이 방식이다.
- 문구: "diagnosed convs keep W8 (9 convs, 1.6% of weights)". activation은 저비트 그대로임을 오해하지 않게 "W8"만 쓴다(A8이라고 쓰지 않는다).
- 첫 conv `0`(stem)과 head의 마지막 1×1은 **프로토콜상** W8A8이다. P와 다른 이유이므로 P 배지를 붙이지 않는다(§6).

---

## 5. M: 공유 제약 scale 이전 (conv → conv 연결 전체)

**위치:** 특정 블록 하나가 아니라 **생산 conv → 소비 conv로 이어지는 연결**에 작용한다. Fig. 3 기준으로는 Backbone 내부, Backbone → PAN(측면 연결), PAN 내부, Head 내부가 모두 해당한다. 예외는 §5.3에 있다.

### 5.1 무엇을 하나
- 소비 conv 입력 x의 채널별 크기 편차를 s로 나눠 평평하게 만든다. 대가로 weight 열을 s배 키운다.
  - `conv(x, W) = conv(x/s, W·diag(s))`
  - s_c = max|x_c|^α / max|W_{:,c}|^(1−α)
  - x 통계(xmax)는 **calibration 256장 전체**에서 잰다(새 규칙).
- 배포에서 x/s는 소비 conv가 아니라 **생산 conv**가 만든다. 생산 conv의 `conv → SiLU → requant`에서 반올림 직전에 채널별 1/s를 곱한다. 그래서 두 제약이 생긴다.
  1. 같은 생산 채널을 받는 **모든 소비 conv는 같은 s**를 쓴다. 예: C2f의 y₁은 `m.0.cv1`과 `cv2`가 함께 받는다.
  2. residual add로 합쳐지는 생산 채널들은 같은 s를 쓴다.
- concat, split, upsample, maxpool은 s를 그대로 통과시킨다. 그래서 C2f의 Split/Concat, PAN의 Upsample/Concat을 지나도 된다.

### 5.2 어떻게 찾나
- **채널 계보:** 생산 conv 61개마다 출력을 교란해 어느 소비 입력 채널이 어느 생산 채널에서 왔는지 **측정으로** 찾는다(`quant/channel_graph.py`). 구조를 하드코딩하지 않는다.
- **s 탐색:** residual add로 묶인 생산자 연결요소마다 α ∈ {0, .25, .5, .75, 1}과 "이전 없음(s = 1)"을 놓고, 영향받는 소비 conv 출력의 상대 오차가 가장 작은 것을 고른다. s = 1도 후보라 어떤 단위도 나빠지지 않는다(calibration 출력 오차 기준).

### 5.3 M이 들어가지 않는 곳
- **이미지 입력 → stem(`0`):** 생산 conv가 없다.
- **T-CSPLayer(12/15/19/22) 출력을 받는 채널:** 사용자 정의 forward라 계보가 끊긴다. 그래서 s = 1이다. 여기에는 다음이 포함된다.
  - `14` Concat을 거쳐 `15.cv1`로 가는 12 출력 채널
  - `16`의 projection 입력
  - `17` Conv 입력(15 출력)
  - `18`→`19.cv1`의 12 출력 채널
  - Head 첫 conv들(15/19/22 출력)
- **Text Encoder 쪽, attention의 Linear/matmul:** conv가 아니므로 대상이 아니다. attention이 x를 conv 없이 쓰는 곳에서는 그 앞에서 x·s로 되돌린다.

즉 M이 실제로 걸리는 곳은 다음과 같다.
- **Backbone 전체**
- PAN에서 **Backbone 측면 입력(C3/C4/C5 → Concat 11/14/21)**과 T-CSPLayer 내부 conv 사이
- 다운샘플 Conv `17`/`20`의 출력
- Head 내부(Conv → Conv → 1×1) 연결

### 5.4 켤지 말지: 사전 검사 (seed마다)
- M을 켠 모델과 끈 모델(= GPM과 GP)을 둘 다 만든다. 두 모델은 BRECQ를 두 번 돌린다.
- 검사 데이터: **calibration 밖** train2017 200장(calibration 256장 다음 순서)과 COCO 어휘다.
- 규칙: flip_on ≤ flip_off일 때만 켠다(ρ = 1.0).
- 표의 "Ours"는 이 검사로 seed마다 고른 GP 또는 GPM이다.
- 검사는 PTQ 시점에만 쓰이고, 배포 모델에는 없다.

### 5.5 그림 표현
- Fig. 3 수준: 검출기 전체에 걸친 가로 괄호와 M 배지. 지금 그림과 같다. 문구는 "per-channel scale moved from each producer conv into all of its consumer convs (one shared s)"이다.
  - 더 정확히 하려면 괄호를 T-CSPLayer 출력 앞에서 끊어 그리거나, 캡션에 "except at T-CSPLayer outputs"를 넣는다.
- 펼친 그림: 생산 conv → ÷s → 소비 A·B(W·diag(s), **same s**). 채널 범위 막대(전/후)는 개념도라 실측이 아님을 코드 주석에 남겨 두었다.
- D 영역의 "M: kept only if a calib.-side flip check approves it"는 §5.4와 맞다.

---

## 6. R과 양자화 프로토콜: 무엇이 저비트이고 무엇이 FP인가

**R (기반 PTQ, 바꾸지 않음)** (`quant/brecq.py`)
- Backbone은 **블록 단위**(C2f 하나를 통째로) 재구성한다.
- PAN은 **conv 단위** 재구성이다(QDrop 공식 COCO 프로토콜과 같은 `neck_layerwise`).
- Head의 cv2/cv3는 conv 단위다.
- 2000 iter이고, weight rounding(AdaRound)과 activation step(LSQ)을 함께 학습한다.
- QDrop은 같은 함수에 activation drop 0.5를 준 것이다.
- G·P·M이 바꾼 모델을 그대로 받는다.

**양자화 범위 (확정 프로토콜 `attn_cls`)**

| Fig. 3 위치 | 정밀도 |
|---|---|
| Text Encoder (CLIP) | **FP32, 오프라인**. 배포 기기에서 돌지 않는다 |
| Vocabulary Embeddings에서 나온 상수(guide Linear 출력, 정규화된 t_j) | 클래스별 INT8 상수(`ConstQ`) |
| YOLO Backbone, RepVL-PAN, Head의 conv (70개) | W4 / A4–A8 (P 대상은 W8) |
| stem `0`, Head cv2/cv3 마지막 1×1 | W8A8(첫·마지막 레이어 관례). Head 마지막 1×1의 **입력만 A16** |
| T-CSPLayer 게이트 경로, I-Pooling Attention의 Linear·matmul·softmax 출력 | 8bit |
| Region-Text Matching (contrastive matmul) | 8bit (영역 임베딩 × 텍스트) |
| LayerNorm, softmax/sigmoid/max 계산, residual add, DFL, NMS | FP |

**그림 표현:** 검출기(Backbone ~ Head)를 점선으로 감싸 "quantized"로 표시하고, Text Encoder는 밖에 FP로 둔다. 지금 그림과 같다. R 배지는 그 점선 영역에 하나만 붙인다. v1에서는 I-Pooling(16)이 이미지로 텍스트를 갱신하므로, 19·22가 받는 guide는 상수가 아니라 실행 중 8bit 값이다(그림에서 구분할 필요는 없다).

---

## 7. 시간 순서 (빌드 순서): 그림의 왼쪽→오른쪽 흐름과 다르다

```
D 진단(오프라인) → G 구조 교환 → 양자화기 삽입(W/A 비트, 첫·마지막 W8, head 입력 A16)
→ P (보호 conv의 weight 비트를 8로) → attention·contrastive 양자화 지점 삽입
→ M (계보 추적 → 공유 s 탐색 → W←W·diag(s), x←x/s) → calibrate(범위 확정) → R (BRECQ/QDrop)
→ [M 사전 검사: M 끔/켬 두 모델 중 선택]
```

- 위치를 보이는 그림(Fig. 3식)과 순서를 보이는 띠(기존 method_overview 상단의 D → G → P → M → R 띠)는 다른 정보다.
- 한 그림에 넣으려면 배지에 순서 번호를 매기는 방식(①D ②G ③P ④M ⑤R)이 가장 간단하다.

---

## 8. 지금 그림(method_arch.png)에서 고칠 만한 곳

| # | 위치 | 지금 | 권장 | 이유 |
|---|---|---|---|---|
| 1 | Backbone | 상자 하나에 P 배지 | 단계별로 쪼개고 1·2·4·6에만 P | 백본 전체가 W8로 읽힌다 (§4) |
| 2 | PAN | 노란 기둥 하나에 G 배지 | 기둥 안에 T-CSPLayer ×4(G)와 I-Pooling ×1(배지 없음) | G가 I-Pooling에도 붙는 것처럼 읽힌다 (§3) |
| 3 | G 펼친 그림 | p를 y₂에서 바로 뽑음 | y₂ → proj_conv(3×3) → p, y₂ → gate 화살표 추가 | 실제 구조와 맞춘다 (§1.2) |
| 4 | G 펼친 그림 | "one quantizer" | (선택) 분기별 범위 7.7 / 7.0 / 9.3 / 20.4 막대 | 왜 문제인지 숫자로 보인다(기존 overview (b)에 있음) |
| 5 | M 괄호 | 검출기 전체 | 캡션 또는 괄호에 "except T-CSPLayer outputs" | §5.3 |
| 6 | Head | "Text Contrastive Head" 하나 | (선택) cv3 → Object Embeddings, cv4 = Region-Text Matching 명시 | Fig. 3과 같은 수준이면 지금도 충분하다 |
| 7 | 순서 | 배지 문자만 | (선택) 배지에 순서 번호 | §7 |
| 8 | I-Pooling | 그리지 않음 | Fig. 4를 따라 그린다면 **하나만** | v1에는 I-Pooling이 하나뿐이다 (§1.2) |

---

## 9. 다른 모델에 같은 그림을 쓸 때

| 모델 | G 위치 | P 위치 | 비고 |
|---|---|---|---|
| YOLO-World-S v2 | 12, 15, 18, 21 | 진단에 따름 | I-Pooling 없음. 새 규칙에서 GPM < PM 문제가 아직 열려 있다(outline 열린 문제) |
| YOLO-World-M / L | 12, 15, 19, 22 | 진단에 따름 | L은 12.cv2가 56.8 pp로 예산을 넘지만 G가 처리한다 |
| YOLOE-v8s / 11s | 없음(텍스트 게이트 없음) | 초반 backbone split/concat 경계 | GPM = PM |
| OWLv2, Grounding DINO | 없음 | 여러 층에 분산 | 손상이 퍼져 있어 P의 집중도가 낮다(35%, 5%) |

본문 그림은 S 하나로 그리고, 다른 모델의 위치는 sensitivity_maps_all과 부록 표로 보이는 구성이 자연스럽다.
