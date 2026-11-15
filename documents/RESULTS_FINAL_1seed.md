# 실험 결과 고정본 (현재 규칙, seed 0)

**생성:** 2026-11-15 10:51 · `scripts/leaderboard/export_final.py` · 코드 커밋 `e66c491` (**커밋 안 된 코드 변경 있음:** `figures/src/bits_curve.py`, `run_owl.py`, `scripts/leaderboard/export_final.py`)

**대상 독자:** 공저자, 그리고 논문 표를 만드는 사람. 이 파일은 스크립트가 만든다. 손으로 고치지 말고 다시 돌린다. 해석과 논의는 [RESULTS_2026-11-04.md](RESULTS_2026-11-04.md), [PAPER_STRUCTURE_2026-11-05.md](PAPER_STRUCTURE_2026-11-05.md)에 둔다.

**포함 기준**
- 범위: 이 저장소의 `runs/`만 읽는다(이전 저장소 promptcal-ptq 제외). 평가 이미지를 잘라 돌린 점검용 실행도 뺀다.
- 규칙: 11-02 규칙만 쓴다. P는 G 대상을 먼저 빼고 1.5% 예산을 다시 채우고(분모 = 양자화 weight 전체), M은 xmax를 calibration 전체에서 재고(하한 0.3%), 사전 검사는 ρ = 1.0이다. 기존 규칙 결과는 표에 넣지 않는다(§6).
- 프로토콜: attn_cls(conv, attention, region–텍스트 내적 양자화), 첫·마지막 W8A8, head 마지막 1×1 입력 A16, 재구성 2,000 iters, calibration 256장, 결정적 실행. OWLv2·GDINO는 각자의 확정 프로토콜이다.
- 값: **LVIS minival AP(×100) / LVIS top-1 flip(%)**, **모든 표가 seed 0 한 번 실행**이다(11-11 결정: 논문 결과를 seed 0으로 통일). Δ는 같은 seed 0끼리의 차이다. 3 seed 평균과 95% 신뢰구간은 [RESULTS_FINAL_3seed.md](RESULTS_FINAL_3seed.md)(부록용)에 있다. ⏳ = 대기열에 있음, – = 없음.
- **YOLO-World-S W4A5 seed 0은 BRECQ·QDrop 기준선이 무너지는 seed다**(BRECQ LVIS AP 11.29). 이 칸의 Δ는 seed 1·2보다 크게 나오므로 논문에는 각주로 밝힌다(3 seed 평균은 3seed 파일).
- Ours: BRECQ 위 G+P(+M). seed 0의 calibration 밖 검사(ρ = 1.0)로 GP / GPM 중 하나를 고른다(다른 seed에도 같은 선택). QDrop + Ours, PD-Quant + Ours는 같은 선택을 그 기반 위에 쓴다. YOLOE는 G가 없으므로 P / PM으로 적는다.
- GDINO의 LVIS flip은 정의가 달라 재측정 전까지 비운다. seed별 원자료와 칸별 출처(log, 저장 모델)는 [results_final.csv](results_final.csv)에 있다.

---

## 1. 메인 표: YOLO 계열 6개 모델 (W4A8–W4A4)

| 모델 | 비트 | FP32 | BRECQ | + Ours | Δ AP (vs BRECQ) | 선택 | QDrop | + Ours | Δ AP (vs QDrop) | PD-Quant | + Ours | Δ AP (vs PD-Quant) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO-World-S | W4A8 | 25.89 | 22.75 / 17.5 | **23.97 / 10.7** | +1.22 | GPM | 22.63 / 18.6 | **24.40 / 11.1** | +1.77 | 22.19 / 24.0 | **23.91 / 15.9** | +1.72 |
| YOLO-World-S | W4A6 | 25.89 | 21.33 / 20.6 | **23.66 / 13.7** | +2.33 | GP | 20.10 / 30.1 | **22.84 / 16.8** | +2.74 | 19.90 / 32.7 | **22.11 / 21.8** | +2.21 |
| YOLO-World-S | W4A5 | 25.89 | 11.29 / 62.3 | **22.94 / 16.2** | +11.65 | GPM | 5.74 / 81.2 | **22.39 / 17.5** | +16.65 | 5.33 / 83.3 | **21.97 / 22.6** | +16.64 |
| YOLO-World-S | W4A4 | 25.89 | 0.45 / 97.2 | **18.60 / 32.2** | +18.15 | GPM | 0.27 / 97.9 | **12.71 / 46.2** | +12.44 | 0.26 / 98.0 | **11.74 / 52.7** | +11.48 |
| YOLO-World-S v2 | W4A8 | 22.19 | 15.98 / 45.6 | **19.44 / 27.1** | +3.46 | GPM | 17.01 / 42.8 | **19.26 / 25.5** | +2.25 | 17.19 / 45.7 | **19.03 / 28.2** | +1.84 |
| YOLO-World-S v2 | W4A6 | 22.19 | 15.72 / 49.3 | **21.15 / 24.9** | +5.43 | GPM | 13.41 / 58.2 | **21.06 / 27.9** | +7.65 | 12.71 / 61.9 | **20.29 / 29.8** | +7.58 |
| YOLO-World-S v2 | W4A5 | 22.19 | 13.30 / 55.5 | **17.19 / 38.8** | +3.89 | GPM | 6.52 / 81.4 | **17.21 / 36.5** | +10.69 | 6.56 / 82.2 | **16.42 / 39.0** | +9.86 |
| YOLO-World-S v2 | W4A4 | 22.19 | 0.21 / 99.3 | **10.86 / 67.2** | +10.65 | GPM | 0.01 / 99.2 | **6.10 / 75.1** | +6.09 | 0.01 / 99.1 | **5.29 / 78.0** | +5.28 |
| YOLO-World-M | W4A8 | 29.80 | 24.82 / 30.6 | **26.14 / 22.4** | +1.32 | GP | 24.74 / 31.7 | **26.50 / 22.5** | +1.76 | 23.50 / 38.2 | **25.61 / 29.0** | +2.11 |
| YOLO-World-M | W4A6 | 29.80 | 23.48 / 35.7 | **24.84 / 28.2** | +1.36 | GP | 22.22 / 42.0 | **24.69 / 32.5** | +2.47 | 21.51 / 47.0 | **24.62 / 35.6** | +3.11 |
| YOLO-World-M | W4A5 | 29.80 | 19.44 / 50.0 | **23.65 / 35.2** | +4.21 | GP | 10.00 / 81.6 | **17.42 / 55.6** | +7.42 | 9.20 / 82.0 | **18.24 / 54.6** | +9.04 |
| YOLO-World-M | W4A4 | 29.80 | 0.54 / 97.6 | **15.59 / 64.9** | +15.05 | GPM | 0.06 / 98.5 | **13.15 / 72.2** | +13.09 | 0.20 / 98.0 | **12.14 / 75.6** | +11.94 |
| YOLO-World-L | W4A8 | 33.43 | 28.34 / 22.4 | **31.98 / 9.9** | +3.64 | GPM | 25.02 / 32.9 | **31.90 / 9.5** | +6.88 | 24.51 / 36.7 | **30.65 / 17.5** | +6.14 |
| YOLO-World-L | W4A6 | 33.43 | 26.34 / 29.3 | **31.53 / 11.7** | +5.19 | GPM | 20.83 / 42.0 | **31.27 / 12.7** | +10.44 | 22.73 / 39.6 | **29.54 / 19.6** | +6.81 |
| YOLO-World-L | W4A5 | 33.43 | 15.37 / 53.9 | **29.37 / 17.2** | +14.00 | GPM | 1.39 / 89.2 | **25.68 / 23.8** | +24.29 | 2.21 / 89.4 | **23.80 / 30.3** | +21.59 |
| YOLO-World-L | W4A4 | 33.43 | 0.06 / 98.4 | **22.60 / 34.0** | +22.54 | GPM | 0.00 / 99.2 | **14.93 / 55.0** | +14.93 | 0.00 / 99.3 | **12.84 / 59.0** | +12.84 |
| YOLOE-v8s | W4A8 | 25.94 | 23.71 / 14.2 | **24.57 / 10.4** | +0.86 | P | 23.64 / 14.6 | **24.21 / 10.6** | +0.57 | 23.26 / 19.9 | **24.90 / 15.7** | +1.64 |
| YOLOE-v8s | W4A6 | 25.94 | 23.28 / 17.5 | **24.34 / 13.5** | +1.06 | PM | 21.53 / 20.4 | **23.97 / 13.9** | +2.44 | 21.69 / 26.2 | **24.37 / 18.6** | +2.68 |
| YOLOE-v8s | W4A5 | 25.94 | 17.18 / 33.8 | **23.19 / 16.4** | +6.01 | PM | 12.60 / 55.2 | **23.15 / 16.5** | +10.55 | 11.78 / 58.0 | **23.27 / 21.5** | +11.49 |
| YOLOE-v8s | W4A4 | 25.94 | 4.64 / 79.2 | **20.39 / 25.8** | +15.75 | PM | 0.10 / 95.7 | **19.28 / 27.8** | +19.18 | 0.03 / 96.2 | **18.99 / 32.6** | +18.96 |
| YOLOE-11s | W4A8 | 26.81 | 22.57 / 20.1 | **22.85 / 18.0** | +0.28 | P | 22.86 / 20.1 | **23.14 / 17.9** | +0.28 | 22.07 / 27.0 | **22.73 / 23.9** | +0.66 |
| YOLOE-11s | W4A6 | 26.81 | 20.96 / 27.1 | **22.25 / 23.6** | +1.29 | PM | 20.00 / 28.9 | **21.91 / 23.1** | +1.91 | 18.38 / 35.5 | **20.43 / 30.8** | +2.05 |
| YOLOE-11s | W4A5 | 26.81 | 11.56 / 58.8 | **17.88 / 35.5** | +6.32 | PM | 5.96 / 77.8 | **16.18 / 39.2** | +10.22 | 4.81 / 83.8 | **14.80 / 46.6** | +9.99 |
| YOLOE-11s | W4A4 | 26.81 | 0.34 / 94.6 | **10.92 / 58.8** | +10.58 | PM | 0.13 / 94.4 | **8.95 / 56.9** | +8.82 | 0.12 / 97.0 | **8.11 / 61.8** | +7.99 |

### 1b. 보충 표: W8A8

- 11-13 결정: W8A8은 모든 방법이 FP 근처라 본문 메인 표에서 빼고 보충자료에 둔다. Ours가 기반보다 낮은 칸도 있으므로(아래 Δ) 본문에는 "차이가 작다, 전체는 보충자료" 한 문장만 쓰고 "해가 없다"고는 쓰지 않는다.

| 모델 | 비트 | FP32 | BRECQ | + Ours | Δ AP (vs BRECQ) | 선택 | QDrop | + Ours | Δ AP (vs QDrop) | PD-Quant | + Ours | Δ AP (vs PD-Quant) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO-World-S | W8A8 | 25.89 | 25.54 / 3.8 | **25.78 / 2.7** | +0.24 | GPM | 25.37 / 3.9 | **25.80 / 2.7** | +0.43 | 25.44 / 7.2 | **25.69 / 6.3** | +0.25 |
| YOLO-World-S v2 | W8A8 | 22.19 | 22.69 / 4.8 | **22.76 / 8.1** | +0.07 | GPM | 22.80 / 5.5 | **22.52 / 8.1** | -0.28 | 22.65 / 8.1 | **22.44 / 10.9** | -0.21 |
| YOLO-World-M | W8A8 | 29.80 | 28.91 / 11.5 | **28.88 / 10.1** | -0.03 | GP | 29.14 / 9.7 | **28.69 / 9.3** | -0.45 | 28.56 / 16.4 | **28.65 / 15.5** | +0.09 |
| YOLO-World-L | W8A8 | 33.43 | 33.37 / 3.6 | **32.77 / 4.0** | -0.60 | GPM | 33.45 / 3.8 | **33.24 / 3.7** | -0.21 | 32.87 / 8.8 | **32.60 / 9.0** | -0.27 |
| YOLOE-v8s | W8A8 | 25.94 | 25.79 / 2.3 | **25.86 / 1.8** | +0.07 | PM | 25.78 / 2.5 | **25.75 / 1.8** | -0.03 | 27.01 / 7.8 | **27.05 / 7.2** | +0.04 |
| YOLOE-11s | W8A8 | 26.81 | 26.54 / 3.5 | **26.73 / 2.7** | +0.19 | PM | 26.55 / 3.5 | **26.63 / 2.7** | +0.08 | 26.55 / 11.9 | **26.93 / 11.4** | +0.38 |

## 2. 표 1 후보: YOLO-World-S, 비교 방법 전체

| 방법 | W8A8 | W4A8 | W4A6 | W4A5 | W4A4 |
|---|---|---|---|---|---|
| FP32 | 25.89 | | | | |
| RTN | 25.44 / 5.0 | 1.30 / 85.2 | 0.57 / 92.4 | 0.12 / 99.2 | 0.00 / 99.8 |
| AdaRound | 25.34 / 4.5 | 7.79 / 61.8 | – | – | – |
| BRECQ | 25.54 / 3.8 | 22.75 / 17.5 | 21.33 / 20.6 | 11.29 / 62.3 | 0.45 / 97.2 |
| QDrop | 25.37 / 3.9 | 22.63 / 18.6 | 20.10 / 30.1 | 5.74 / 81.2 | 0.27 / 97.9 |
| BRECQ + SmoothQuant | 25.30 / 4.2 | 21.66 / 22.1 | 21.10 / 24.9 | 18.74 / 35.4 | 9.25 / 66.0 |
| BRECQ + AWQ | 25.53 / 3.4 | 21.80 / 19.8 | 20.38 / 26.9 | 18.95 / 32.5 | 0.33 / 98.1 |
| **Ours (BRECQ)** | 25.78 / 2.7 (GPM) | 23.97 / 10.7 (GPM) | 23.66 / 13.7 (GP) | 22.94 / 16.2 (GPM) | 18.60 / 32.2 (GPM) |
|   └ GP | 25.70 / 2.9 | 24.57 / 10.3 | 23.66 / 13.7 | 22.33 / 19.9 | 2.54 / 89.0 |
|   └ GPM | 25.78 / 2.7 | 23.97 / 10.7 | 23.34 / 12.6 | 22.94 / 16.2 | 18.60 / 32.2 |
| Ours (QDrop) | 25.80 / 2.7 (GPM) | 24.40 / 11.1 (GPM) | 22.84 / 16.8 (GP) | 22.39 / 17.5 (GPM) | 12.71 / 46.2 (GPM) |
| PD-Quant | 25.44 / 7.2 | 22.19 / 24.0 | 19.90 / 32.7 | 5.33 / 83.3 | 0.26 / 98.0 |
| Ours (PD-Quant) | 25.69 / 6.3 (GPM) | 23.91 / 15.9 (GPM) | 22.11 / 21.8 (GP) | 21.97 / 22.6 (GPM) | 11.74 / 52.7 (GPM) |

- Δ는 §1(기반 PTQ 대비)과 §4(조건별)에 있다.

## 3. ViT·DETR 계열: OWLv2-B/16, Grounding DINO-T

| 모델 | 비트 | FP32 | BRECQ | BRECQ + Ours | 구성 | Δ AP | QDrop | QDrop + Ours | Δ AP | 평가 |
|---|---|---|---|---|---|---|---|---|---|---|
| OWLv2-B/16 | W8A8 | 32.40 | 28.14 / 17.1 | **29.80 / 12.4** | M | +1.67 | – | – | – | LVIS minival |
| OWLv2-B/16 | W4A8 | 32.40 | 19.74 / 28.2 | **21.81 / 25.9** | PM | +2.06 | 25.45 / 27.3 | **26.63 / 25.2** | +1.18 | LVIS minival |
| OWLv2-B/16 | W4A5 | 32.40 | 2.22 / 57.5 | **7.15 / 44.2** | PM | +4.93 | – | – | – | LVIS minival |
| Grounding DINO-T | W4A5 | 30.06 | 27.26 | **27.83** | P (검사) | +0.56 | – | – | – | LVIS 0–2000 |

- OWLv2: W8A8은 M, W4는 PM. 사후 M 검사(runs/262, YOLO와 같은 기준)가 세 칸 모두 같은 구성을 고른다.
- GDINO W4A5: 검사가 P를 고른다(8.73 vs PM 8.79%). 11-15부터 P를 싣는다(runs/263, 같은 2000장). 검사가 놓친 PM은 30.23이다. flip은 재측정 전이라 비운다.
- QDrop + Ours가 없는 칸은 QDrop 기준선도 싣지 않는다(11-15). 있는 칸은 OWLv2 W4A8뿐이다.

## 4. 표 3 후보: YOLO-World-S ablation (Δ AP vs BRECQ)

| 조건 | W4A8 AP / flip | W4A8 Δ | W4A5 AP / flip | W4A5 Δ | W4A4 AP / flip | W4A4 Δ |
|---|---|---|---|---|---|---|
| BRECQ | 22.75 / 17.5 |  | 11.29 / 62.3 |  | 0.45 / 97.2 |  |
| +G | 23.48 / 13.8 | +0.73 | 21.20 / 22.2 | +9.91 | 2.24 / 89.8 | +1.79 |
| +P | 23.71 / 11.9 | +0.96 | 14.30 / 51.8 | +3.01 | 0.81 / 94.2 | +0.36 |
| +M | 21.97 / 20.4 | -0.78 | 20.66 / 25.7 | +9.37 | 9.05 / 66.6 | +8.60 |
| +GP | 24.57 / 10.3 | +1.82 | 22.33 / 19.9 | +11.04 | 2.54 / 89.0 | +2.09 |
| +GM | 23.31 / 14.0 | +0.56 | 21.48 / 19.7 | +10.19 | 17.19 / 36.0 | +16.74 |
| +PM | 23.82 / 12.4 | +1.07 | 22.11 / 21.3 | +10.82 | 16.59 / 39.8 | +16.14 |
| +GPM | 23.97 / 10.7 | +1.22 | 22.94 / 16.2 | +11.65 | 18.60 / 32.2 | +18.15 |
| +R | 23.00 / 17.1 | +0.25 | 11.48 / 62.6 | +0.19 | – | – |
| +H | 23.62 / 12.0 | +0.87 | 14.47 / 51.5 | +3.18 | – | – |
| +GPS | 23.92 / 11.9 | +1.17 | 21.93 / 19.1 | +10.64 | – | – |
| +GPW | 23.99 / 10.5 | +1.24 | 21.90 / 19.0 | +10.61 | – | – |

- R = 같은 예산의 무작위 보호, H = HAWQ식 보호, GPS / GPW = M 대신 SmoothQuant / AWQ식 이전.

## 5. 그 밖의 현재 규칙 결과 (부록 후보)

| 분류 | 모델 | 프로토콜 | 비트 | 조건 | 꼬리표 | seed | LVIS AP / flip | Δ AP (vs 기반) | 출처 |
|---|---|---|---|---|---|---|---|---|---|
| 미지정 | YOLO-World-S | 확정 (attn8+A16) | W4A5 | brecq+GH | – | 0 | 21.61 / 20.9 | +10.32 | 212_next |
| 미지정 | YOLO-World-S | 확정 (attn8+A16) | W4A8 | combined(s1=brecq) | – | 0 | 21.48 / 21.0 | -1.27 | 158_final |
| 미지정 | OWLv2-B/16 | OWL | W4A8 | brecq+PW | – | 0 | 19.66 / 26.2 | -0.08 | 185_owl_awq |
| 추가지표1 | YOLO-World-S | 확정 (attn8+A16) | W4A6 | brecq+PM | – | 0 | 23.10 / 15.0 | +1.77 | 234_rerun_s0 |
| 추가지표1 | YOLO-World-S | 확정 (attn8+A16) | W8A8 | brecq+M | – | 0 | 25.67 / 3.2 | +0.13 | 234_rerun_s0 |
| 추가지표1 | YOLO-World-S | 확정 (attn8+A16) | W8A8 | naive+S | – | 0 | 25.27 / 5.2 | -0.27 | 170_sw |
| 추가지표1 | YOLO-World-S | 확정 (attn8+A16) | W8A8 | naive+W | – | 0 | 25.66 / 4.2 | +0.12 | 170_sw |
| 추가지표1 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A5 | brecq+P | – | 0 | 16.12 / 45.3 | +2.82 | 249_budget_palone |
| 추가지표1 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A5 | brecq+PM | – | 0 | 19.72 / 39.4 | +6.42 | 234_rerun_s0 |
| 추가지표1 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A8 | brecq+P | – | 0 | 20.32 / 26.6 | +4.34 | 249_budget_palone |
| 추가지표1 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A8 | brecq+PM | – | 0 | 22.45 / 21.6 | +6.47 | 234_rerun_s0 |
| 추가지표1 | YOLO-World-S v2 | 확정 (attn8+A16) | W8A8 | brecq+PM | – | 0 | 23.25 / 7.7 | +0.56 | 234_rerun_s0 |
| 추가지표1 | YOLO-World-M | 확정 (attn8+A16) | W4A5 | brecq+G | – | 0 | 22.61 / 36.5 | +3.17 | 154_protocol |
| 추가지표1 | YOLO-World-M | 확정 (attn8+A16) | W4A5 | brecq+P | – | 0 | 21.79 / 42.2 | +2.35 | 154_protocol, 186_m_p |
| 추가지표1 | YOLO-World-M | 확정 (attn8+A16) | W4A8 | brecq+G | – | 0 | 25.59 / 24.4 | +0.77 | 154_protocol |
| 추가지표1 | YOLO-World-M | 확정 (attn8+A16) | W4A8 | brecq+P | – | 0 | 25.82 / 22.8 | +1.00 | 154_protocol, 186_m_p |
| 추가지표1 | YOLO-World-L | 확정 (attn8+A16) | W4A5 | brecq+G | – | 0 | 23.57 / 29.6 | +8.20 | 212_next |
| 추가지표1 | YOLO-World-L | 확정 (attn8+A16) | W4A5 | brecq+P | – | 0 | 17.47 / 49.5 | +2.10 | 249_budget_palone |
| 추가지표1 | YOLO-World-L | 확정 (attn8+A16) | W4A8 | brecq+P | – | 0 | 29.34 / 20.2 | +1.00 | 249_budget_palone |
| 추가지표1 | OWLv2-B/16 | OWL | W4A8 | brecq+P | – | 0 | 19.40 / 28.3 | -0.35 | 184_owl_main |
| 추가지표1 | Grounding DINO-T | GDINO | W4A5 | brecq+P | – | 0 | – | – | 216_gdino |
| 추가지표1 | Grounding DINO-T | GDINO · LVIS 0–400 | W4A5 | brecq+P | – | 0 | 35.13 | – | 263_gdino_w4a5_P_lvis |
| 추가지표1 | Grounding DINO-T | GDINO · LVIS 1200–1600 | W4A5 | brecq+P | – | 0 | 36.01 | – | 263_gdino_w4a5_P_lvis |
| 추가지표1 | Grounding DINO-T | GDINO · LVIS 1600–2000 | W4A5 | brecq+P | – | 0 | 35.06 | – | 263_gdino_w4a5_P_lvis |
| 추가지표1 | Grounding DINO-T | GDINO · LVIS 400–500 | W4A5 | brecq+P | – | 0 | 47.87 | +2.15 | 263_gdino_w4a5_P_lvis |
| 추가지표1 | Grounding DINO-T | GDINO · LVIS 400–800 | W4A5 | brecq+P | – | 0 | 38.39 | – | 263_gdino_w4a5_P_lvis |
| 추가지표1 | Grounding DINO-T | GDINO · LVIS 800–1200 | W4A5 | brecq+P | – | 0 | 37.97 | – | 263_gdino_w4a5_P_lvis |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A4 | pdquant+GPM | – | 0 | 11.74 / 52.7 | +11.48 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A4 | qdrop+GP | – | 0 | 0.48 / 96.9 | +0.21 | 234_rerun_s0 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A4 | qdrop+GPM | – | 0 | 12.71 / 46.2 | +12.44 | 234_rerun_s0, 246_w4a4_seeds |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A5 | pdquant+GPM | – | 0 | 21.97 / 22.6 | +16.64 | 256_pdquant_s12 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A5 | qdrop+GPM | – | 0 | 22.39 / 17.5 | +16.65 | 234_rerun_s0, 236_fill_s0 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A5 | qdrop+PM | – | 0 | 21.17 / 25.0 | +15.43 | 234_rerun_s0 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A6 | pdquant+GP | – | 0 | 22.11 / 21.8 | +2.21 | 256_pdquant_s12 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A6 | qdrop+GP | – | 0 | 22.84 / 16.8 | +2.74 | 247_w4a6_s0, 251_w4a6_seeds |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A6 | qdrop+GPM | – | 0 | 23.17 / 14.0 | +3.07 | 247_w4a6_s0 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A6 | qdrop+PM | – | 0 | 22.69 / 15.4 | +2.59 | 234_rerun_s0 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A8 | pdquant+GPM | – | 0 | 23.91 / 15.9 | +1.72 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A8 | qdrop+GPM | – | 0 | 24.40 / 11.1 | +1.77 | 234_rerun_s0, 236_fill_s0 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A8 | qdrop+PM | – | 0 | 24.10 / 12.7 | +1.47 | 234_rerun_s0 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W8A8 | pdquant+GPM | – | 0 | 25.69 / 6.3 | +0.25 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W8A8 | qdrop+GPM | – | 0 | 25.80 / 2.7 | +0.43 | 234_rerun_s0, 240_w8seeds_w4a4 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W8A8 | qdrop+M | – | 0 | 25.63 / 3.1 | +0.26 | 234_rerun_s0 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) · calib o365 | W4A5 | brecq | – | 0 | 18.37 / 31.1 | – | 208_calib_src |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) · calib o365 | W4A8 | brecq | – | 0 | 22.78 / 17.5 | – | 208_calib_src |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) · calib o365gqa | W4A5 | brecq | – | 0 | 19.19 / 27.9 | – | 209_calib_o365gqa |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) · calib o365gqa | W4A8 | brecq | – | 0 | 23.00 / 16.9 | – | 209_calib_o365gqa |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) · 첫·마지막 FP | W4A4 | brecq | – | 0 | 0.47 / 97.3 | – | 203_w4a4, 214_w4a4_s2 |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A4 | pdquant+GPM | – | 0 | 5.29 / 78.0 | +5.28 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A4 | qdrop+GP | – | 0 | 0.32 / 98.6 | +0.31 | 240_w8seeds_w4a4 |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A4 | qdrop+GPM | – | 0 | 6.10 / 75.1 | +6.09 | 240_w8seeds_w4a4, 246_w4a4_seeds |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A5 | pdquant+GPM | – | 0 | 16.42 / 39.0 | +9.86 | 256_pdquant_s12 |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A5 | qdrop+GPM | – | 0 | 17.21 / 36.5 | +10.69 | 236_fill_s0 |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A6 | pdquant+GPM | – | 0 | 20.29 / 29.8 | +7.58 | 256_pdquant_s12 |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A6 | qdrop+GP | – | 0 | 17.25 / 35.0 | +3.84 | 247_w4a6_s0 |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A6 | qdrop+GPM | – | 0 | 21.06 / 27.9 | +7.65 | 247_w4a6_s0, 251_w4a6_seeds |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A8 | pdquant+GPM | – | 0 | 19.03 / 28.2 | +1.84 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A8 | qdrop+GPM | – | 0 | 19.26 / 25.5 | +2.25 | 236_fill_s0 |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W8A8 | pdquant+GPM | – | 0 | 22.44 / 10.9 | -0.21 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W8A8 | qdrop+GPM | – | 0 | 22.52 / 8.1 | -0.28 | 238_w8a8, 240_w8seeds_w4a4 |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) | W4A4 | pdquant+GPM | – | 0 | 12.14 / 75.6 | +11.94 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) | W4A4 | qdrop+GPM | – | 0 | 13.15 / 72.2 | +13.09 | 240_w8seeds_w4a4, 246_w4a4_seeds |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) | W4A5 | pdquant+GP | – | 0 | 18.24 / 54.6 | +9.04 | 256_pdquant_s12 |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) | W4A5 | qdrop+GP | – | 0 | 17.42 / 55.6 | +7.42 | 236_fill_s0 |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) | W4A6 | pdquant+GP | – | 0 | 24.62 / 35.6 | +3.11 | 256_pdquant_s12 |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) | W4A6 | qdrop+GP | – | 0 | 24.69 / 32.5 | +2.47 | 247_w4a6_s0, 251_w4a6_seeds |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) | W4A6 | qdrop+GPM | – | 0 | 18.24 / 59.4 | -3.98 | 247_w4a6_s0 |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) | W4A8 | pdquant+GP | – | 0 | 25.61 / 29.0 | +2.11 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) | W4A8 | qdrop+GP | – | 0 | 26.50 / 22.5 | +1.76 | 236_fill_s0 |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) | W8A8 | pdquant+GP | – | 0 | 28.65 / 15.5 | +0.09 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) | W8A8 | qdrop+GP | – | 0 | 28.69 / 9.3 | -0.45 | 238_w8a8, 240_w8seeds_w4a4 |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) · 첫·마지막 FP | W4A4 | brecq | – | 0 | 1.12 / 96.5 | – | 205_w4a4_m_abl, 214_w4a4_s2 |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) · 첫·마지막 FP | W4A4 | brecq+GPM | – | 0 | 15.99 / 64.4 | +14.87 | 234_rerun_s0 |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W4A4 | pdquant+GPM | – | 0 | 12.84 / 59.0 | +12.84 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W4A4 | qdrop+GP | – | 0 | 0.00 / 99.1 | +0.00 | 240_w8seeds_w4a4 |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W4A4 | qdrop+GPM | – | 0 | 14.93 / 55.0 | +14.93 | 240_w8seeds_w4a4, 246_w4a4_seeds |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W4A5 | pdquant+GPM | – | 0 | 23.80 / 30.3 | +21.59 | 256_pdquant_s12 |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W4A5 | qdrop+GPM | – | 0 | 25.68 / 23.8 | +24.29 | 234_rerun_s0, 236_fill_s0 |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W4A6 | pdquant+GPM | – | 0 | 29.54 / 19.6 | +6.81 | 256_pdquant_s12 |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W4A6 | qdrop+GP | – | 0 | 28.49 / 20.1 | +7.66 | 247_w4a6_s0 |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W4A6 | qdrop+GPM | – | 0 | 31.27 / 12.7 | +10.44 | 247_w4a6_s0, 251_w4a6_seeds |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W4A8 | pdquant+GPM | – | 0 | 30.65 / 17.5 | +6.14 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W4A8 | qdrop+GPM | – | 0 | 31.90 / 9.5 | +6.88 | 234_rerun_s0, 236_fill_s0 |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W8A8 | pdquant+GPM | – | 0 | 32.60 / 9.0 | -0.27 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W8A8 | qdrop+GP | – | 0 | 33.53 / 3.9 | +0.08 | 238_w8a8 |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W8A8 | qdrop+GPM | – | 0 | 33.24 / 3.7 | -0.21 | 238_w8a8, 240_w8seeds_w4a4 |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W4A4 | pdquant+GPM | – | 0 | 18.99 / 32.6 | +18.96 | 255_pdquant_s0 |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W4A4 | qdrop+GP | – | 0 | 0.05 / 96.0 | -0.05 | 240_w8seeds_w4a4 |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W4A4 | qdrop+GPM | – | 0 | 19.28 / 27.8 | +19.18 | 240_w8seeds_w4a4, 246_w4a4_seeds |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W4A5 | pdquant+GPM | – | 0 | 23.27 / 21.5 | +11.49 | 256_pdquant_s12 |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W4A5 | qdrop+GPM | – | 0 | 23.15 / 16.5 | +10.55 | 236_fill_s0 |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W4A6 | pdquant+GPM | – | 0 | 24.37 / 18.6 | +2.68 | 256_pdquant_s12 |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W4A6 | qdrop+GP | – | 0 | 23.01 / 16.3 | +1.48 | 247_w4a6_s0 |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W4A6 | qdrop+GPM | – | 0 | 23.97 / 13.9 | +2.44 | 247_w4a6_s0, 251_w4a6_seeds |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W4A8 | pdquant+GP | – | 0 | 24.90 / 15.7 | +1.64 | 255_pdquant_s0 |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W4A8 | qdrop+GP | – | 0 | 24.21 / 10.6 | +0.57 | 236_fill_s0 |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W8A8 | pdquant+GPM | – | 0 | 27.05 / 7.2 | +0.04 | 255_pdquant_s0 |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W8A8 | qdrop+GPM | – | 0 | 25.75 / 1.8 | -0.03 | 238_w8a8, 240_w8seeds_w4a4 |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W4A4 | pdquant+GPM | – | 0 | 8.11 / 61.8 | +7.99 | 255_pdquant_s0 |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W4A4 | qdrop+GP | – | 0 | 0.22 / 96.2 | +0.09 | 240_w8seeds_w4a4 |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W4A4 | qdrop+GPM | – | 0 | 8.95 / 56.9 | +8.82 | 240_w8seeds_w4a4, 246_w4a4_seeds |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W4A5 | pdquant+GPM | – | 0 | 14.80 / 46.6 | +9.99 | 256_pdquant_s12 |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W4A5 | qdrop+GPM | – | 0 | 16.18 / 39.2 | +10.22 | 236_fill_s0 |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W4A6 | pdquant+GPM | – | 0 | 20.43 / 30.8 | +2.05 | 256_pdquant_s12 |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W4A6 | qdrop+GP | – | 0 | 20.68 / 25.7 | +0.68 | 247_w4a6_s0 |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W4A6 | qdrop+GPM | – | 0 | 21.91 / 23.1 | +1.91 | 247_w4a6_s0, 251_w4a6_seeds |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W4A8 | pdquant+GP | – | 0 | 22.73 / 23.9 | +0.66 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W4A8 | qdrop+GP | – | 0 | 23.14 / 17.9 | +0.28 | 236_fill_s0 |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W8A8 | pdquant+GPM | – | 0 | 26.93 / 11.4 | +0.38 | 255_pdquant_s0 |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W8A8 | qdrop+GPM | – | 0 | 26.63 / 2.7 | +0.08 | 238_w8a8, 240_w8seeds_w4a4 |
| 추가지표2 | OWLv2-B/16 | OWL | W4A8 | qdrop+PM | – | 0 | 26.63 / 25.2 | +1.18 | 252_fill |

- 분류는 리더보드의 표 배정(judge + table_assign.json)을 따른다. 추가지표1 = ablation, 2 = 일반화 보조, 3 = 분석·민감도.

## 6. 아직 기존 규칙 결과만 있는 묶음

새 규칙 결과로 대체된 기존 규칙 묶음 89개는 쓰지 않는다. 아래는 새 규칙 결과가 아직 없어 표에서 빠진 묶음이다.

| 분류 | 모델 | 프로토콜 | 비트 | 조건 | 꼬리표 | seed | 출처 |
|---|---|---|---|---|---|---|---|
| 미지정 | Grounding DINO-T | GDINO | W4A5 | brecq+M | M 8장, 하한 0% | 0 | 216_gdino |
| 미지정 | Grounding DINO-T | GDINO | W4A5 | brecq+M | M 8장 | 0 | 218_gdino_m8 |
| 미지정 | Grounding DINO-T | GDINO · LVIS 0–1000 | W4A5 | brecq+M | M 8장 | 0 | 220_gdino_lvis |
| 미지정 | Grounding DINO-T | GDINO | W4A5 | brecq+PM | 하한 1% | 0 | 216_gdino |
| 미지정 | Grounding DINO-T | GDINO | W4A5 | brecq+PM | M 8장 | 0 | 218_gdino_m8 |
| 미지정 | Grounding DINO-T | GDINO · LVIS 0–1000 | W4A5 | brecq+PM | M 8장 | 0 | 220_gdino_lvis |
| 미지정 | YOLOE-11s | 확정 (attn8+A16) | W4A5 | brecq+GPMA | P 기존, M 8장, 하한 0%, ρ 1.5 | 0 | 171_yoloe11 |
| 미지정 | YOLO-World-L | 확정 (attn8+A16) | W4A5 | brecq+GPMA | P 기존, M 8장, 하한 0%, ρ 1.5 | 0 1 2 | 211_L_main |
| 미지정 | YOLO-World-L | 확정 (attn8+A16) | W4A5 | brecq+GPMA | P 기존, M 8장, ρ 1.5 | 0 | 219_floor_verify |
| 미지정 | YOLO-World-L | 확정 (attn8+A16) | W4A5 | qdrop+GPMA | P 기존, M 8장, 하한 0%, ρ 1.5 | 0 1 2 | 212_next |
| 미지정 | YOLO-World-L | 확정 (attn8+A16) | W4A8 | brecq+GPMA | P 기존, M 8장, 하한 0%, ρ 1.5 | 0 1 2 | 211_L_main |
| 미지정 | YOLO-World-L | 확정 (attn8+A16) | W4A8 | brecq+GPMA | P 기존, M 8장, ρ 1.5 | 0 | 219_floor_verify |
| 미지정 | YOLO-World-L | 확정 (attn8+A16) | W4A8 | qdrop+GPMA | P 기존, M 8장, 하한 0%, ρ 1.5 | 0 1 2 | 212_next |
| 미지정 | YOLO-World-M | 확정 (attn8+A16) | W4A4 | brecq+GPMA | P 기존, M 8장, ρ 1.5 | 0 1 | 219_floor_verify, 223_floor_rerun |
| 미지정 | YOLO-World-M | 확정 (attn8+A16) | W4A4 | brecq+GPMA | P 기존, M 8장, 하한 0%, ρ 1.5 | 0 1 2 | 205_w4a4_m_abl, 214_w4a4_s2 |
| 미지정 | YOLO-World-M | 확정 (attn8+A16) | W4A5 | brecq+GPMA | P 기존, M 8장, ρ 1.5 | 0 | 219_floor_verify |
| 미지정 | YOLO-World-M | 확정 (attn8+A16) | W4A5 | brecq+GPS | P 기존 | 0 1 2 | 212_next |
| 미지정 | YOLO-World-M | 확정 (attn8+A16) | W4A5 | brecq+GPW | P 기존 | 0 1 2 | 212_next |
| 미지정 | YOLO-World-M | 확정 (attn8+A16) | W4A5 | brecq+PM | M 8장, 하한 0% | 0 1 | 154_protocol |
| 미지정 | YOLO-World-M | 확정 (attn8+A16) | W4A8 | brecq+GPS | P 기존 | 0 1 2 | 212_next |
| 미지정 | YOLO-World-M | 확정 (attn8+A16) | W4A8 | brecq+GPW | P 기존 | 0 1 2 | 212_next |
| 미지정 | YOLO-World-M | 확정 (attn8+A16) | W4A8 | brecq+PM | M 8장, 하한 0% | 0 1 | 154_protocol |
| 미지정 | YOLO-World-S | 확정 (attn8+A16) | W4A5 | brecq+GHM | M 8장, 하한 0% | 0 1 2 | 212_next |
| 미지정 | YOLO-World-S | 확정 (attn8+A16) | W4A6 | brecq+GPS | P 기존 | 0 1 2 3 4 5 | 170_sw |
| 미지정 | YOLO-World-S | 확정 (attn8+A16) | W4A6 | brecq+GPW | P 기존 | 0 1 2 3 4 5 | 170_sw |
| 미지정 | YOLO-World-S | 확정 (attn8+A16) | W8A8 | brecq+GPS | P 기존 | 0 1 2 3 4 5 | 170_sw |
| 미지정 | YOLO-World-S | 확정 (attn8+A16) | W8A8 | brecq+GPW | P 기존 | 0 1 2 3 4 5 | 170_sw |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) · 첫·마지막 FP | W4A4 | brecq+GPM | P 기존, M 8장, 하한 0% | 0 | 205_w4a4_m_abl |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) · 첫·마지막 FP | W4A4 | brecq+GPMA | P 기존, M 8장, 하한 0%, ρ 1.5 | 0 1 2 | 205_w4a4_m_abl, 214_w4a4_s2 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) · 첫·마지막 FP | W4A4 | brecq+GPM | P 기존, M 8장, 하한 0% | 0 1 2 | 203_w4a4, 214_w4a4_s2 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) · calib o365 | W4A5 | brecq+GPMA | P 기존, M 8장, 하한 0%, ρ 1.5 | 0 1 2 | 208_calib_src |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) · calib o365gqa | W4A5 | brecq+GPMA | P 기존, M 8장, 하한 0%, ρ 1.5 | 0 1 2 | 209_calib_o365gqa |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) · calib o365 | W4A8 | brecq+GPMA | P 기존, M 8장, 하한 0%, ρ 1.5 | 0 1 2 | 208_calib_src |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) · calib o365gqa | W4A8 | brecq+GPMA | P 기존, M 8장, 하한 0%, ρ 1.5 | 0 1 2 | 209_calib_o365gqa |

## 7. 절별 출처 (run 디렉토리)

- §1 메인 표: `154_protocol`, `158_final`, `160_gen6`, `164_yoloe`, `171_yoloe11`, `203_w4a4`, `211_L_main`, `212_next`, `214_w4a4_s2`, `219_floor_verify`, `223_floor_rerun`, `229_budget_seeds_S`, `234_rerun_s0`, `235_v2_refill_check`, `236_fill_s0`, `238_w8a8`, `240_w8seeds_w4a4`, `246_w4a4_seeds`, `247_w4a6_s0`, `251_w4a6_seeds`, `255_pdquant_s0`, `256_pdquant_s12`
- §2 표 1: `154_protocol`, `158_final`, `170_sw`, `203_w4a4`, `214_w4a4_s2`, `229_budget_seeds_S`, `234_rerun_s0`, `236_fill_s0`, `240_w8seeds_w4a4`, `246_w4a4_seeds`, `247_w4a6_s0`, `251_w4a6_seeds`, `255_pdquant_s0`, `256_pdquant_s12`
- §3 ViT·DETR: `184_owl_main`, `234_rerun_s0`, `237_gdino_lvis2000`, `250_owl_seeds`, `252_fill`
- §4 ablation: `154_protocol`, `158_final`, `203_w4a4`, `205_w4a4_m_abl`, `212_next`, `214_w4a4_s2`, `229_budget_seeds_S`, `233_noG_repro`, `234_rerun_s0`, `236_fill_s0`, `246_w4a4_seeds`, `253_seg_w4a4_abl_w4a8`, `253b_rerun_killed`
- 칸별 log와 저장 모델 경로는 csv의 `log`, `save_model` 열에 있다.
