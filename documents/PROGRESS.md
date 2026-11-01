# 실험 진행 기록 (10-30 ~ 11-01, GDINO 추가와 M 하한 통일)

변경 내용과 그 이유는 [GDINO_CHANGES.md](GDINO_CHANGES.md)에 있다. 이 문서는 무엇을 돌렸고, 무엇이 돌고 있고, 무엇이 남았는지를 기록한다.

최종 갱신: 2026-11-01 16:50

---

## 1. 결정 사항 (시간순)

| 날짜 | 결정 | 근거 |
|---|---|---|
| 10-30 | GDINO에서 M이 붕괴한 원인은 8장 xmax의 구멍이다. 하한(xmax floor)으로 고친다 | brecq+M W4A5 AP 0.01, 희소 채널의 s ≈ 1e-5 |
| 10-31 | xmax는 8장 그대로 두고 하한 0.3%를 넣는다. 256장 규칙은 부록 ablation 후보로만 둔다 | 256장 규칙은 기존 α를 20~50% 바꿔 약 170 GPU-h 재실험이 필요. 0.3%는 기존 α를 하나도 바꾸지 않음 |
| 10-31 | GDINO는 LVIS로도 평가한다(minival 일부) | COCO는 calibration과 같은 어휘라 우리 주장(보지 않은 어휘 보호)을 재지 못함 |
| 11-01 | YOLO/OWL 기본값도 하한 0.3%로 바꾼다(M 정의 하나로 통일) | `run_comparison.py`, `run_owl.py`의 `--xmax-floor` 기본값 0 → 0.003 |
| 11-01 | 하한이 실제로 작동하는 실험은 전부 다시 돌린다 | 안 돌리면 "옛 코드 결과"라고 논문에 기재해야 함 |
| 11-01 | 하한이 작동하지 않는 모델(S·YOLOE·v2)은 seed 0 대표 재현만 먼저 하고, 나머지 seed는 시간이 남으면 돌린다 | 같은 모델이 만들어지므로 숫자가 같아야 함. GPU 시간은 GDINO 보강을 우선 |
| 11-01 | S 보호 예산 스윕을 ③ 재현보다 먼저 돌린다(①② 다음). 예산 1.5%는 학습 쪽 flip이 다른 값을 명확히 가리킬 때만 바꾼다 | 예산을 정당화하는 실험이 없음. LVIS로 예산을 고르면 누수이므로 LVIS는 보고용 |

---

## 2. 완료

### 2.1 GDINO W4A5 (seed 0, 8장 + 하한 0.3%)

| 조건 | COCO AP (5000장) | LVIS AP (앞 1000장) | 실행 |
|---|---|---|---|
| FP32 | 48.50 | 34.43 | |
| BRECQ | 40.33 | 31.72 | `runs/216_gdino`, `runs/220_gdino_lvis` |
| QDrop | 41.79 | 33.76 | 위와 같음 |
| +M | 41.12 | 34.57 | `runs/218_gdino_m8`, `runs/220_gdino_lvis` |
| +PM | 41.19 | **34.96** | 위와 같음 |

- PM은 COCO에서 QDrop보다 0.6 낮다. LVIS에서는 QDrop보다 1.2 높다. 평가 전에 정한 기준(1점 이상)을 넘는다.
- PM과 M의 LVIS AP가 FP보다 높게 나왔다. 그래서 1000장 subset의 노이즈를 감안해 같은 이미지 위의 차이로만 해석한다.
- 저장 모델: `/data/taeho/cvpr2027_qmodels/gdino/` (256장 + 1% 규칙 모델은 `mig256/`)

### 2.2 하한 0.3%가 기존 모델을 바꾸는지 점검 (α·s 비교)

| 모델 | 점검한 비트 | 결과 | 실행 |
|---|---|---|---|
| YOLO-World-S | W8A8, W4A8, W4A6, W4A5, W4A4 | α·s 변화 0 | `runs/217_mig_rule`, `runs/221_floor_bits` |
| YOLOE-v8s, YOLOE-11s | W8A8, W4A8, W4A6, W4A5, W4A4 | α·s 변화 0 | 위와 같음 |
| YOLO-World-v2 | W8A8, W4A8, W4A5 | α·s 변화 0 | `runs/222_floor_v2owl` |
| YOLO-World-m | W4A8, W4A5 | α 같음, s 일부 변화 | `runs/217_mig_rule` |
| YOLO-World-L | W4A8, W4A5 | α 같음, s 일부 변화 (최대 41배) | 위와 같음 |
| OWLv2 | W4A8, W4A5, W8A8 | α 같음, s 일부 변화 (최대 267배) | `runs/217_mig_rule`, `runs/222_floor_v2owl` |

- calibration 8장은 seed와 무관하다(train2017 정렬 앞 8장). 따라서 α·s 변화 0인 모델은 모든 seed에서 같은 모델이 만들어진다.

### 2.3 하한 0.3% seed 0 검증 (평가까지)

| 실험 | 기존 → 하한 0.3% (LVIS AP) | 판단 | 실행 |
|---|---|---|---|
| L W4A8 GPMA | 31.86 → 31.61 | 3-seed 범위 안 | `runs/219_floor_verify` |
| L W4A5 GPMA | 29.12 → 29.25 | 3-seed 범위 안 | 위와 같음 |
| m W4A5 GPMA | +A 판정 M 끔 (같음) | 최종 모델 동일 | 위와 같음 |
| m W4A4 GPM | 11.90 → 11.82 | 3-seed 범위 안 | 위와 같음 |
| OWL W4A8 PM | 21.24 → 21.22 (COCO −0.54, APr −0.51) | LVIS 같음 | 위와 같음 |

- 자세한 표는 [GDINO_CHANGES.md](GDINO_CHANGES.md) 3절에 있다.

### 2.4 코드 변경

| 파일 | 변경 |
|---|---|
| `core/gdino_quant.py`, `core/owl_quant.py`, `quant/channel_graph.py` | M의 xmax 하한(`xmax_floor`), xmax 측정 이미지 선택(`stat_images`) |
| `run_comparison.py`, `run_owl.py` | `--xmax-floor`(11-01부터 기본 0.003), `--mig-stat`. mcheck 캐시 이름에 하한 표시(`_xf0.003`) |
| `run_gdino.py` | `--xmax-floor`(기본 0.003), `--mig-stat`. 11-01: `--lvis-start`(LVIS 구간 시작), `--save-dets`(검출 결과 json 저장) |
| `run_comparison.py` (11-01 오후) | `--protect-budget`(기본 1.5%, 기존과 동일), `--train-flip`(calib 밖 train 200장 flip 기록) |
| `scripts/mig_rule_check.py` | 규칙·하한별 α·s 비교 점검 스크립트 |

---

## 3. 진행 중 / 대기 (GPU 5, 6, 7)

GPU별로 앞 작업이 끝나면 다음 작업이 자동으로 시작된다. 각 작업은 시작 전에 해당 GPU에 다른 사용자 프로세스가 있는지 확인한다.

### ① 하한이 작동하는 실험 재실행 (`runs/223_floor_rerun`, 하한 0.3%, 원래와 같은 조건 묶음)

| GPU | 먼저 (OWL) | 이후 |
|---|---|---|
| 5 | W8A8 brecq+M s0 → s1 | YOLO 대기열 |
| 6 | W4A8 brecq+PM s2 → W8A8 brecq+M s2 | YOLO 대기열 |
| 7 | W4A8 brecq+PM s1 | YOLO 대기열 |

- YOLO 대기열(`queue.tsv`, 15건)은 세 GPU가 나눠 가진다.
  - m W4A4: s1, s2, 첫·마지막 16bit 변형 s0~s2
  - L BRECQ+GPMA: W4A8·W4A5의 s1, s2
  - L QDrop+GPMA: W4A8·W4A5의 s0~s2
- OWL은 단독 실행과 묶음 실행 결과가 소수점까지 같다(2.3절). 그래서 M 조건만 단독으로 돌린다.
- seed 0 중 이미 하한으로 돌린 것(L W4A8·W4A5, m W4A4, OWL W4A8 PM)은 `runs/219_floor_verify` 결과를 쓴다.
- 저장 모델: `/data/taeho/cvpr2027_qmodels/xf003/`. 기존 모델은 덮어쓰지 않는다.
- 규모: 약 50 GPU-시간. 11-02 03~04시 완료 예상.

### ② GDINO LVIS 두 번째 1000장 (`runs/224_gdino_lvis2`)
- minival 1000~2000번 이미지를 저장된 모델로 평가만 한다(재빌드 없음).
- GPU 배정: GPU 5 = +PM, GPU 6 = QDrop, GPU 7 = BRECQ.
- 검출 결과는 `/data/taeho/cvpr2027_dets/gdino/`에 저장한다. 나중에 앞 1000장을 다시 평가해 합치면 정확한 2000장 AP를 낼 수 있다.
- 판단: 첫 1000장과 같은 순서(PM > QDrop > BRECQ)가 나오는지 본다.
- 규모: GPU당 약 8.5시간. 11-02 정오 전후 완료 예상.

### ④ S 보호 예산 스윕 (`runs/225_repro` 대기열 앞 36건, 태그 `bud-*`)
- YOLO-World-S, W4A5·W4A8, `brecq+GP` 단독, 예산 0 / 0.5 / 1 / 1.5 / 3 / 5%, seed 0~2.
- `--train-flip`: calib 밖 train2017 200장(+A 검사와 같은 이미지, 보호 진단 200장과 겹치지 않음)의 flip을 같이 기록한다.
- 판정 규칙(돌리기 전 확정): 예산은 학습 쪽 flip 곡선으로만 판단한다. 1.5%가 아닌 값을 명확히 가리킬 때만 바꾼다. LVIS·COCO는 보고용.
- 규모: 약 16 GPU-시간. ② 다음, ③보다 먼저 시작한다. 기존 ③ 대기열은 `queue.repro_only.tsv`에 보관.

### ③ S·YOLOE·v2 seed 0 재현 (`runs/225_repro`, 15건, ④ 다음)
- 원래와 같은 조건 묶음으로 하한 0.3% 기본값을 써서 다시 돌린다.
  - S: W4A8, W4A5, W4A6, W8A8(M), W8A8 GPM, W4A5 ablation
  - v2: W4A8, W4A5, W8A8
  - YOLOE-v8s, YOLOE-11s: 각각 W4A8, W4A5, W8A8
- 기대: 원본과 소수점까지 같다.
  - 다르면 원본(10월 초) 이후의 다른 코드 변경 때문일 수 있다. 그 실험만 하한 0으로 한 번 더 돌려 원인을 가린다.
- 규모: 약 26 GPU-시간. 11-02 밤 완료 예상.

---

## 4. 남은 일 (미승인 또는 미정)

### 실험

| 우선순위 | 실험 | 채우는 빈틈 | 규모 |
|---|---|---|---|
| 1 | S·YOLOE·v2 나머지 seed 재현 | 모든 표가 하한 코드로 재현됨을 직접 확인 | 약 200 GPU-h (SmoothQuant/AWQ 비교 포함 시 +52) |
| 2 | GDINO seed 1, 2 (BRECQ, QDrop, PM) | GDINO가 seed 1개뿐 | 약 80 GPU-h |
| 3 | GDINO W4A8 PM + LVIS | GDINO 비트 설정이 하나뿐 | 약 30 GPU-h |
| 4 | GDINO qdrop+PM | plug-in 주장의 GDINO 근거 | 약 13 GPU-h |
| 5 | GDINO naive+M 하한 0.1% (COCO 500장) | 하한 민감도 표 완성 | 약 30분 |

### 문서·원고
- RESULTS.md 갱신: GDINO 표, 하한 재실험 결과로 m·L·OWL 표와 seed 짝 통계(Holm 보정 p값) 재계산
- 논문 부록:
  - 하한 없음 vs 있음(GDINO 붕괴)
  - 하한 민감도(0.1/0.3/1%)
  - 256장 규칙 ablation
  - "S·YOLOE·v2에서는 하한에 걸리는 채널이 없다"는 설명
- baseline·모델별 재구현 문서(원 논문·코드 대비 변경점과 이유) — 10-30 요청분
- PAPER_OUTLINE 주장 대조표 갱신(W4A4, OWL, GDINO 반영)
- abstract 등록 11-10, 본문 11-16

---

## 5. 주의 사항
- GDINO LVIS는 1000장 subset이다. 절대값이 FP보다 높게 나오는 칸이 있으므로 차이로만 해석한다. APr은 참고용이다.
- GDINO에서는 flip이 AP와 같은 방향으로 움직이지 않는다(QDrop: flip이 BRECQ보다 높은데 AP도 높음).
- 하한 0.3%를 정할 때 COCO val 500장 AP를 참고한 적이 있다. 논문에는 학습 쪽 기준만 쓴다(기존 α 불변 + calibration 밖 train 이미지 mcheck).
- 평가 시간: PM·M이 BRECQ보다 느린 것은 시뮬레이션 비용 2% 정도다(M의 x/s 나눗셈). 실제 배포 시 추가 연산은 없다.
