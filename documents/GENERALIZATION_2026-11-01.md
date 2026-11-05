# 일반화 실험 결과 정리 (GDINO 제외)

작성: 2026-11-01

일반화 주장을 위해 돌린 실험을 축별로 모은 문서다. GDINO는 [`GDINO_CHANGES.md`](../docs/GDINO_CHANGES.md)와 [`PROGRESS.md`](../docs/PROGRESS.md)에 따로 있다.

- 수치 출처: [`../docs/RESULTS.md`](../docs/RESULTS.md), [`../docs/PAPER_DIRECTION.md`](../docs/PAPER_DIRECTION.md), [`METHOD_SCENARIO.md`](METHOD_SCENARIO.md)
- 표기
  - Δ는 BRECQ 대비 LVIS AP 차이다(같은 seed끼리의 짝 차이 평균).
  - \*는 95% CI가 0 밖, 즉 유의하다는 뜻이다.
  - LVIS AP는 점(×100) 단위다.
- **주의:** YOLO-World-L, YOLO-World-M W4A4, OWLv2는 xmax 하한(0.3%) 반영 재실행이 진행 중이다(`runs/223_floor_rerun`, 11-02 새벽 완료 예정). 끝나면 해당 수치를 교체한다.

---

## 0. 한눈에

| 축 | 질문 | 실험 | 결론 |
|---|---|---|---|
| ① 모델 | 다른 크기·버전·구조에서도 되나 | v2, M, L, YOLOE ×2, OWLv2 | W4에서 모두 개선 |
| ② calibration 데이터 | COCO가 아니어도 되나 | Objects365, Objects365+GQA | 결정·성능 거의 같음, 12/12 개선 |
| ③ 도메인 | COCO/LVIS 밖에서도 되나 | ODinW-13 | W4A5 큰 개선, W4A8 일부 예외 |
| ④ 비트 | 더 낮은 비트에서도 되나 | W4A6, W4A4 | BRECQ 붕괴, GPM 유지 |
| ⑤ 기반 PTQ | BRECQ에만 붙나 | QDrop 위 plug-in | QDrop 위에서도 개선 |
| ⑥ 프로토콜 | 양자화 범위를 엄격히 해도 되나 | attention·contrastive 8bit | 결론 변화 0.3점 이내 |

---

## 1. 모델 일반화

| 모델 | 구조 | 진단이 고른 처방 | seed | W8A8 | W4A8 | W4A5 |
|---|---|---|---|---|---|---|
| YOLO-World-S (주 모델) | CNN, 텍스트 게이트 | G+P+M | 6 | +0.31\* | +0.81\* | +2.53\* (seed 0 제외) |
| YOLO-World-S v2 | CNN, 텍스트 게이트 | G+P+M | 6 | +0.39\* | +5.82\* | +5.64\* |
| YOLO-World-M | CNN, 텍스트 게이트 | G+P (M 끔) | 6 | −0.02 | +1.12\* | +2.23\* |
| YOLO-World-L | CNN, 텍스트 게이트 | G+P+M | 3 | – | [확인 필요] | 16.3 → 29.5 |
| YOLOE-v8s | CNN, 게이트 없음 | P+M | 6 | −0.02 | +0.74\* | +4.40\* |
| YOLOE-11s | CNN(YOLO11), 게이트 없음 | P+M | 6 | +0.11 | +0.73\* | +4.56\* |
| OWLv2-B/16 | ViT | M | 3 | +1.07\* | +1.66\* | – |

- **W4에서는 모든 모델이 개선된다.**
- **W8A8은 일부 모델에서 AP 차이가 없고 flip 감소가 주 효과다.** 예: YOLOE-v8s는 flip이 2.30%에서 1.76%로 23% 줄었다.
- **모델마다 진단이 다른 처방을 고르는데, 그래도 모두 개선된다.** 게이트가 없으면 G가 빠지고, M이 해로우면 사전 검사가 끈다.

**회복률** (FP와 BRECQ 사이의 LVIS AP 손상 중 GPM이 되찾은 비율)

| 모델 | W4A8 | W4A5 |
|---|---|---|
| YOLO-World-S | 26% | 41% (seed 0 제외) |
| YOLO-World-S v2 | 91% | 64% |
| YOLOE-v8s | 34% | 60% |

---

## 2. calibration 데이터 교체 (YOLO-World-S, 3 seed)

양자화 과정 전체(calibration, 보호 진단, M 사전 검사)에서 COCO를 빼고 Objects365 또는 Objects365+GQA로 바꿨다.

- **진단 결과가 같다.** 보호 대상과 M 결정이 COCO calibration과 같다.
- **성능이 거의 같다.**

  | calibration | GPM W4A8 LVIS AP | GPM W4A5 LVIS AP |
  |---|---|---|
  | COCO (주 결과, 6 seed) | 23.55 | 22.31 |
  | Objects365 / +GQA (3 seed) | 23.6 | 21.9~22.0 |

- **모든 경우에 BRECQ보다 좋다.** 3 seed × 2 비트 × 2 데이터, 12개 짝 모두다.
- [확인 필요] Objects365 calibration에서 BRECQ의 수치와 짝 Δ는 문서에 없다. run 로그에서 채운다.

---

## 3. 도메인: ODinW-13 (`docs/RESULTS.md` §4c)

설정을 하나도 바꾸지 않고, 같은 seed로 모델을 다시 빌드해 13개 데이터셋 평균 AP(GLIP 관행)로 평가했다. 다시 빌드한 70개 모델의 COCO AP가 원 로그와 모두 같은 것도 확인했다.

| 모델 | FP32 | W4A8 Δ (우리가 나은 seed) | W4A5 Δ (우리가 나은 seed) | W4A5 회복률 |
|---|---|---|---|---|
| YOLO-World-S | 31.89 | +1.51\* (6/6) | +3.26\* (5/5) | 58% |
| YOLOE-v8s | 29.73 | **−1.17\*** (1/6) | +7.95\* (6/6) | 91% |
| YOLOE-11s | 31.79 | +0.24 (4/6) | +9.68\* (6/6) | 69% |

- **W4A5에서는 세 모델 모두 LVIS보다 ODinW에서 더 크게 오른다.** Δ LVIS가 +2.53 / +4.40 / +4.56일 때 Δ ODinW는 +3.26 / +7.95 / +9.68이다.
- 개선은 특정 데이터셋이 아니라 대부분의 데이터셋에서 나온다. 예: YOLOE-11s W4A5에서 PascalVOC 21.4 → 37.1, Vehicles 11.6 → 33.9.
- **예외: YOLOE-v8s W4A8 (−1.17)**
  - thermalDogsAndPeople(열화상 41장) 하나에서 6 seed 모두 약 15점 낮다.
  - 이 데이터셋을 빼면 +0.02 ± 0.54로 차이가 없다.
  - 이 데이터셋에서는 BRECQ(35.0)가 FP(25.1)보다 10점 높다. 양자화 오차가 우연히 이득으로 작용한 것으로 본다.
- YOLOE 두 모델은 W4A8 BRECQ가 이미 FP 수준이라(29.97 vs 29.73, 31.62 vs 31.79) 되찾을 여지가 없다.
- OWLv2는 W4A8 seed 0의 PM(34.06, FP 42.43)만 있다. BRECQ는 아직 ODinW로 재지 않았다.

---

## 4. 비트: W4A6, W4A4

**W4A6 (YOLO-World-S, 6 seed):** BRECQ 21.52 → GPM 23.18 (+1.66\*)

**W4A4 (2 seed, `docs/RESULTS.md` §4d):** BRECQ는 붕괴하고, GPM은 버틴다.

| 모델 | FP LVIS | BRECQ LVIS (seed 0 / 1) | GPM LVIS (seed 0 / 1) |
|---|---|---|---|
| YOLO-World-S | 25.89 | 0.45 / 0.52 | 18.65 / 18.16 |
| YOLO-World-M | 29.80 | 0.54 / 0.69 | 11.90 / 12.34 |

- **W4A4에서는 M이 결정적이다.** S의 ablation에서 G나 P만으로는 LVIS 0~3점이고, M이 들어가야 15~18점이다.
- **같은 규칙이 비트에 맞게 다른 결정을 내린다.** YOLO-World-M은 W4A5에서 사전 검사가 M을 끄지만(6.24% / 43.46%), W4A4에서는 켠다(94.67% / 53.04%). W4A5 결정을 그대로 썼다면 GP가 되어 붕괴했다(LVIS 0.04~0.16).
- 첫·마지막 층을 FP로 둔 조건(QATMA와 같은 조건)에서도 같은 경향이다.

---

## 5. 기반 PTQ: QDrop 위 plug-in

| 설정 | QDrop | QDrop + GPM | Δ vs QDrop |
|---|---|---|---|
| S W8A8 | 25.41 | 25.86 | +0.45\* |
| S W4A8 | 22.52 | 23.63 | +1.12\* |
| S W4A6 | 19.95 | 22.99 | +3.04\* |
| S W4A5 | 14.14 | 22.27 | +6.42\* (seed 0 제외) |
| L W4A5 | 1.4 (붕괴) | 26.9 | – |

- **GPM을 쓰면 BRECQ와 QDrop의 차이가 사라진다.** QDrop + GPM과 BRECQ + GPM의 LVIS AP 차이가 모든 설정에서 ±0.2점 이내이고 유의하지 않다.
- QDrop 단독은 W4A5에서 BRECQ보다 3.9점 낮았다. GPM이 기반 PTQ를 무너뜨리던 손상을 직접 막는다는 근거다.

---

## 6. 프로토콜 ablation (YOLO-World-S, 6 seed)

attention·contrastive를 FP로 둘 때와 8bit로 양자화할 때를 비교했다.

| GPM − BRECQ (Δ LVIS AP) | attention·contrastive FP | attention·contrastive 8bit |
|---|---|---|
| W4A8 | +1.09 | +1.21 |
| W4A5 (seed 0 제외) | +2.62 | +2.62 |

결론 변화는 0.3점 이내다.

---

## 7. 보조: 누수 없는 결정의 안정성

- **M 사전 검사를 일반 어휘로 바꿔도 결정이 같다.** COCO 대신 WordNet 명사 4,000개(COCO·LVIS 이름 제외)를 썼을 때, 6개 경우 모두 COCO 검사와 결정이 같았다(`docs/RESULTS.md` §6).
- **보호 목록은 프로토콜이 바뀌어도 같다.** S, v2, M 모두 이전 프로토콜과 확정 프로토콜에서 같은 목록이 나왔다.
- **calibration 데이터를 바꿔도 같다.** 2절 참고.

---

## 8. 빈 곳과 약점

| 항목 | 상태 | 할 일 |
|---|---|---|
| YOLO-World-L W4A8 Δ | 문서에 BRECQ 값이 없음 | run 로그에서 확인 |
| YOLO-World-L, M W4A4, OWLv2 | 하한 반영 재실행 중 | 11-02 결과로 교체 |
| Objects365 calibration의 BRECQ 수치·짝 Δ | 문서에 GPM 값과 "12/12"만 있음 | run 로그에서 확인 |
| OWLv2 W4A5 | 미실행 | 붕괴 여부만이라도 확인(naive/BRECQ 1 seed) |
| OWLv2 ODinW | seed 0 PM만 있음 | BRECQ 대비 seed 실행 여부 결정 |
| YOLOE-v8s ODinW W4A8 | −1.17 | 한계로 서술(thermalDogs 하나 때문) |
| 실기기 지연 | 미측정 | 한계로 서술 |
