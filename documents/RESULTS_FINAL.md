# 실험 결과 고정본 (현재 규칙)

**생성:** 2026-11-11 14:43 · `scripts/leaderboard/export_final.py` · 코드 커밋 `901d26c` (**커밋 안 된 코드 변경 있음:** `figures/src/bits_curve.py`, `figures/src/mcheck_scatter.py`, `quant/brecq.py`, `quant/pdquant.py`, `run_comparison.py`, `run_gdino.py`, `scripts/leaderboard/export_final.py`, `scripts/leaderboard/make_leaderboard.py`)

**대상 독자:** 공저자, 그리고 논문 표를 만드는 사람. 이 파일은 스크립트가 만든다. 손으로 고치지 말고 다시 돌린다. 해석과 논의는 [RESULTS_2026-11-04.md](RESULTS_2026-11-04.md), [PAPER_STRUCTURE_2026-11-05.md](PAPER_STRUCTURE_2026-11-05.md)에 둔다.

**포함 기준**
- 범위: 이 저장소의 `runs/`만 읽는다(이전 저장소 promptcal-ptq 제외). 평가 이미지를 잘라 돌린 점검용 실행도 뺀다.
- 규칙: 11-02 규칙만 쓴다. P는 G 대상을 먼저 빼고 1.5% 예산을 다시 채우고(분모 = 양자화 weight 전체), M은 xmax를 calibration 전체에서 재고(하한 0.3%), 사전 검사는 ρ = 1.0이다. 기존 규칙 결과는 표에 넣지 않는다(§6).
- 프로토콜: attn_cls(conv, attention, region–텍스트 내적 양자화), 첫·마지막 W8A8, head 마지막 1×1 입력 A16, 재구성 2,000 iters, calibration 256장, 결정적 실행. OWLv2·GDINO는 각자의 확정 프로토콜이다.
- 값: **LVIS minival AP(×100) / LVIS top-1 flip(%)**. 표 1–4는 seed 0–2 평균이고, 위첨자 ⁽ⁿ⁾는 seed 수가 3이 아닌 칸이다. Δ는 같은 seed끼리 짝지은 차이의 평균 ± 95% t 신뢰구간 반폭이다(seed 1개면 ± 없음). ⏳ = 대기열에 있음, – = 없음.
- Ours: BRECQ 위 G+P(+M). seed 0의 calibration 밖 검사(ρ = 1.0)로 GP / GPM 중 하나를 골라 모든 seed에 쓴다. QDrop + Ours는 같은 선택을 QDrop 위에 쓴다. YOLOE는 G가 없으므로 P / PM으로 적는다.
- GDINO의 LVIS flip은 정의가 달라 재측정 전까지 비운다. seed별 원자료와 칸별 출처(log, 저장 모델)는 [results_final.csv](results_final.csv)에 있다.

---

## 1. 메인 표: YOLO 계열 6개 모델

| 모델 | 비트 | FP32 | BRECQ | + Ours | Δ AP (vs BRECQ) | 선택 | QDrop | + Ours | Δ AP (vs QDrop) | PD-Quant | + Ours | Δ AP (vs PD-Quant) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO-World-S | W8A8 | 25.89 | 25.49 / 3.8 | **25.71 / 2.7** | +0.22 ± 0.12 | GPM | 25.38 / 4.0 | **25.79 / 2.7** | +0.41 ± 0.05 | 25.50 / 7.2 ⁽²⁾ | **25.65 / 6.3 ⁽²⁾** | +0.15 ± 1.21 |
| YOLO-World-S | W4A8 | 25.89 | 22.71 / 17.7 | **24.20 / 10.7** | +1.49 ± 0.74 | GPM | 22.45 / 18.5 | **24.48 / 11.1** | +2.03 ± 0.60 | 22.30 / 23.7 ⁽²⁾ | **23.83 / 15.8 ⁽²⁾** | +1.53 ± 2.48 |
| YOLO-World-S | W4A5 | 25.89 | 16.97 / 39.4 | **22.98 / 16.3** | +6.01 ± 12.14 | GPM | 12.69 / 55.6 | **22.35 / 17.8** | +9.66 ± 15.04 | 5.33 / 83.3 ⁽¹⁾ | **21.97 / 22.6 ⁽¹⁾** | +16.64 |
| YOLO-World-S | W4A4 | 25.89 | 0.43 / 97.1 | **18.09 / 32.9** | +17.66 ± 1.06 | GPM | 0.23 / 98.0 | **13.95 / 42.7** | +13.72 ± 2.75 | 0.30 / 98.1 ⁽²⁾ | **12.43 / 50.4 ⁽²⁾** | +12.13 ± 8.26 |
| YOLO-World-S v2 | W8A8 | 22.19 | 22.73 / 5.0 | **22.69 / 7.9** | -0.04 ± 0.29 | GPM | 22.75 / 5.4 | **22.53 / 7.9** | -0.22 ± 0.15 | 22.50 / 8.0 ⁽²⁾ | **22.47 / 10.8 ⁽²⁾** | -0.03 ± 2.29 |
| YOLO-World-S v2 | W4A8 | 22.19 | 15.77 / 46.0 | **19.05 / 27.0** | +3.28 ± 0.50 | GPM | 17.05 / 42.4 | **19.43 / 25.1** | +2.39 ± 1.00 | 17.23 / 45.6 ⁽²⁾ | **19.01 / 28.1 ⁽²⁾** | +1.78 ± 0.83 |
| YOLO-World-S v2 | W4A5 | 22.19 | 13.22 / 56.2 | **17.03 / 38.0** | +3.81 ± 0.80 | GPM | 7.85 / 74.5 | **16.78 / 35.8** | +8.93 ± 4.73 | 6.56 / 82.2 ⁽¹⁾ | **16.42 / 39.0 ⁽¹⁾** | +9.86 |
| YOLO-World-S v2 | W4A4 | 22.19 | 0.21 / 99.2 | **10.90 / 68.4** | +10.68 ± 0.63 | GPM | 0.01 / 99.2 | **6.12 / 77.4** | +6.12 ± 2.14 | 0.01 / 99.2 ⁽²⁾ | **5.37 / 78.3 ⁽²⁾** | +5.36 ± 1.02 |
| YOLO-World-M | W8A8 | 29.80 | 28.48 / 11.4 | **28.46 / 8.2** | -0.03 ± 3.32 | GP | 28.79 / 10.7 | **28.69 / 10.2** | -0.10 ± 0.76 | 28.59 / 15.9 ⁽²⁾ | **28.66 / 15.7 ⁽²⁾** | +0.07 ± 0.19 |
| YOLO-World-M | W4A8 | 29.80 | 24.44 / 30.5 | **26.14 / 22.3** | +1.70 ± 1.19 | GP | 24.67 / 32.1 | **26.53 / 22.2** | +1.86 ± 0.42 | 23.41 / 38.1 ⁽²⁾ | **25.61 / 29.0 ⁽²⁾** | +2.20 ± 1.14 |
| YOLO-World-M | W4A5 | 29.80 | 19.83 / 48.8 | **22.97 / 36.1** | +3.14 ± 2.32 | GP | 11.36 / 77.6 | **16.73 / 57.2** | +5.37 ± 4.93 | 9.20 / 82.0 ⁽¹⁾ | **18.24 / 54.6 ⁽¹⁾** | +9.04 |
| YOLO-World-M | W4A4 | 29.80 | 0.65 / 97.1 | **14.95 / 65.9** | +14.30 ± 1.61 | GPM | 0.03 / 98.3 | **11.49 / 74.2** | +11.46 ± 3.62 | 0.11 / 98.5 ⁽²⁾ | **10.97 / 76.6 ⁽²⁾** | +10.87 ± 13.66 |
| YOLO-World-L | W8A8 | 33.43 | 33.33 / 3.6 | **33.02 / 3.8** | -0.31 ± 0.72 | GPM | 33.44 / 3.7 | **33.22 / 3.6** | -0.22 ± 0.40 | 32.90 / 8.8 | **32.70 / 8.9** | -0.19 ± 0.17 |
| YOLO-World-L | W4A8 | 33.43 | 27.91 / 23.5 | **32.04 / 9.7** | +4.13 ± 1.24 | GPM | 24.96 / 33.3 | **32.05 / 9.6** | +7.09 ± 0.89 | 24.44 / 36.8 ⁽²⁾ | **30.61 / 17.4 ⁽²⁾** | +6.18 ± 0.44 |
| YOLO-World-L | W4A5 | 33.43 | 16.25 / 52.8 | **29.89 / 16.3** | +13.65 ± 0.91 | GPM | 1.44 / 87.8 | **26.87 / 21.8** | +25.44 ± 3.68 | 2.21 / 89.4 ⁽¹⁾ | **23.80 / 30.3 ⁽¹⁾** | +21.59 |
| YOLO-World-L | W4A4 | 33.43 | 0.04 / 98.5 | **23.43 / 32.3** | +23.39 ± 2.86 | GPM | 0.00 / 99.2 | **16.76 / 49.3** | +16.76 ± 3.95 | 0.00 / 99.2 | **14.74 / 54.5** | +14.74 ± 4.12 |
| YOLOE-v8s | W8A8 | 25.94 | 25.81 / 2.3 | **25.87 / 1.8** | +0.06 ± 0.06 | PM | 25.80 / 2.5 | **25.84 / 1.9** | +0.04 ± 0.16 | 27.01 / 7.8 ⁽¹⁾ | **27.05 / 7.2 ⁽¹⁾** | +0.04 |
| YOLOE-v8s | W4A8 | 25.94 | 23.69 / 14.0 | **24.55 / 10.4** | +0.87 ± 0.08 | P | 23.57 / 14.6 | **24.40 / 10.5** | +0.83 ± 0.56 | 23.26 / 19.9 ⁽¹⁾ | **24.90 / 15.7 ⁽¹⁾** | +1.64 |
| YOLOE-v8s | W4A5 | 25.94 | 17.80 / 32.7 | **23.20 / 16.2** | +5.40 ± 2.62 | PM | 12.31 / 52.7 | **23.37 / 16.5** | +11.07 ± 6.88 | 11.78 / 58.0 ⁽¹⁾ | **23.27 / 21.5 ⁽¹⁾** | +11.49 |
| YOLOE-v8s | W4A4 | 25.94 | 3.44 / 84.9 | **20.15 / 26.0** | +16.71 ± 3.85 | PM | 0.13 / 95.7 | **19.24 / 29.4** | +19.11 ± 0.37 | 0.03 / 96.2 ⁽¹⁾ | **18.99 / 32.6 ⁽¹⁾** | +18.96 |
| YOLOE-11s | W8A8 | 26.81 | 26.55 / 3.4 | **26.72 / 2.6** | +0.17 ± 0.34 | PM | 26.55 / 3.4 | **26.65 / 2.6** | +0.10 ± 0.19 | 26.55 / 11.9 ⁽¹⁾ | **26.93 / 11.4 ⁽¹⁾** | +0.38 |
| YOLOE-11s | W4A8 | 26.81 | 22.82 / 20.0 | **23.00 / 17.9** | +0.18 ± 0.91 | P | 22.81 / 20.2 | **23.23 / 18.0** | +0.42 ± 0.31 | 22.09 / 27.1 ⁽²⁾ | **22.76 / 24.1 ⁽²⁾** | +0.67 ± 0.13 |
| YOLOE-11s | W4A5 | 26.81 | 13.25 / 54.1 | **18.48 / 33.6** | +5.22 ± 4.30 | PM | 6.86 / 74.4 | **17.01 / 36.5** | +10.15 ± 5.06 | 4.81 / 83.8 ⁽¹⁾ | **14.80 / 46.6 ⁽¹⁾** | +9.99 |
| YOLOE-11s | W4A4 | 26.81 | 0.47 / 93.0 | **10.97 / 58.4** | +10.50 ± 1.49 | PM | 0.10 / 96.2 | **9.01 / 56.7** | +8.91 ± 1.71 | 0.12 / 97.0 ⁽¹⁾ | **8.11 / 61.8 ⁽¹⁾** | +7.99 |

## 2. 표 1 후보: YOLO-World-S, 비교 방법 전체

| 방법 | W8A8 | W4A8 | W4A6 | W4A5 | W4A4 |
|---|---|---|---|---|---|
| FP32 | 25.89 | | | | |
| RTN | 25.39 / 5.5 | 1.22 / 85.6 | 0.56 / 92.5 | 0.19 / 96.7 | 0.00 / 99.8 ⁽¹⁾ |
| AdaRound | 25.28 / 4.9 | 8.08 / 60.7 | – | – | – |
| BRECQ | 25.49 / 3.8 | 22.71 / 17.7 | 21.52 / 20.5 | 16.97 / 39.4 | 0.43 / 97.1 |
| QDrop | 25.38 / 4.0 | 22.45 / 18.5 | 19.79 / 29.5 | 12.69 / 55.6 | 0.23 / 98.0 |
| BRECQ + SmoothQuant | 25.47 / 4.1 | 21.81 / 20.6 | 21.12 / 23.2 | 18.83 / 32.8 | 9.25 / 66.0 ⁽¹⁾ |
| BRECQ + AWQ | 25.58 / 3.4 | 21.94 / 19.8 | 20.38 / 27.0 | 18.30 / 34.5 | 0.33 / 98.1 ⁽¹⁾ |
| **Ours (BRECQ)** | 25.71 / 2.7 (GPM) | 24.20 / 10.7 (GPM) | 23.57 / 13.6 (GP) | 22.98 / 16.3 (GPM) | 18.09 / 32.9 (GPM) |
|   └ GP | 25.67 / 2.9 | 24.58 / 10.2 | 23.57 / 13.6 | 22.07 / 19.8 | 2.54 / 89.0 ⁽¹⁾ |
|   └ GPM | 25.71 / 2.7 | 24.20 / 10.7 | 23.34 / 12.6 ⁽¹⁾ | 22.98 / 16.3 | 18.09 / 32.9 |
| Ours (QDrop) | 25.79 / 2.7 (GPM) | 24.48 / 11.1 (GPM) | 22.92 / 16.6 (GP) | 22.35 / 17.8 (GPM) | 13.95 / 42.7 (GPM) |
| PD-Quant | 25.50 / 7.2 ⁽²⁾ | 22.30 / 23.7 ⁽²⁾ | 19.90 / 32.7 ⁽¹⁾ | 5.33 / 83.3 ⁽¹⁾ | 0.30 / 98.1 ⁽²⁾ |
| Ours (PD-Quant) | 25.65 / 6.3 ⁽²⁾ (GPM) | 23.83 / 15.8 ⁽²⁾ (GPM) | 22.11 / 21.8 ⁽¹⁾ (GP) | 21.97 / 22.6 ⁽¹⁾ (GPM) | 12.43 / 50.4 ⁽²⁾ (GPM) |

- Δ와 신뢰구간은 §1(BRECQ·QDrop 대비)과 §4(조건별)에 있다.

## 3. ViT·DETR 계열: OWLv2-B/16, Grounding DINO-T

| 모델 | 비트 | FP32 | BRECQ | QDrop | Ours | 구성 | Δ AP (vs BRECQ) | 평가 |
|---|---|---|---|---|---|---|---|---|
| OWLv2-B/16 | W8A8 | 32.40 | 26.62 / 16.1 | – | **28.08 / 11.1** | M | +1.45 ± 0.62 | LVIS minival |
| OWLv2-B/16 | W4A8 | 32.40 | 19.57 / 27.5 | 25.45 / 27.3 ⁽¹⁾ | **21.12 / 25.4** | PM | +1.56 ± 1.09 | LVIS minival |
| OWLv2-B/16 | W4A5 | 32.40 | 2.22 / 57.5 ⁽¹⁾ | 3.99 / 66.5 ⁽¹⁾ | **7.15 / 44.2 ⁽¹⁾** | PM | +4.93 | LVIS minival |
| Grounding DINO-T | W4A5 | 30.06 | 27.26 ⁽¹⁾ | 29.23 ⁽¹⁾ | **30.23 ⁽¹⁾** | PM | +2.97 | LVIS 0–2000 |

- OWLv2: W8A8은 M, W4는 PM(고정 구성). GDINO: PM(고정 구성), flip은 재측정 전이라 비운다.

## 4. 표 3 후보: YOLO-World-S ablation (Δ AP vs BRECQ)

| 조건 | W4A8 AP / flip | W4A8 Δ | W4A5 AP / flip | W4A5 Δ | W4A4 AP / flip | W4A4 Δ |
|---|---|---|---|---|---|---|
| BRECQ | 22.71 / 17.7 |  | 16.97 / 39.4 |  | 0.43 / 97.1 |  |
| +G | 23.38 / 13.9 | +0.67 ± 0.14 | 21.32 / 22.4 | +4.35 ± 11.96 | 1.52 / 92.9 ⁽²⁾ | +1.03 ± 9.59 |
| +P | 23.80 / 11.9 | +1.09 ± 0.36 | 18.65 / 33.4 | +1.68 ± 2.88 | 0.86 / 94.8 ⁽²⁾ | +0.38 ± 0.25 |
| +M | 21.99 / 20.0 | -0.72 ± 0.29 | 20.66 / 25.7 ⁽¹⁾ | +9.37 | 9.05 / 66.6 ⁽¹⁾ | +8.60 |
| +GP | 24.58 / 10.2 | +1.87 ± 0.44 | 22.07 / 19.8 | +5.10 ± 12.79 | 2.54 / 89.0 ⁽¹⁾ | +2.09 |
| +GM | 23.02 / 14.2 | +0.31 ± 0.53 | 21.48 / 19.7 ⁽¹⁾ | +10.19 | 17.19 / 36.0 ⁽¹⁾ | +16.74 |
| +PM | 23.82 / 12.4 ⁽¹⁾ | +1.07 | 22.11 / 21.3 ⁽¹⁾ | +10.82 | 16.59 / 39.8 ⁽¹⁾ | +16.14 |
| +GPM | 24.20 / 10.7 | +1.49 ± 0.74 | 22.98 / 16.3 | +6.01 ± 12.14 | 18.09 / 32.9 | +17.66 ± 1.06 |
| +R | 22.94 / 17.2 | +0.23 ± 0.06 | 11.48 / 62.6 ⁽¹⁾ | +0.19 | – | – |
| +H | 23.58 / 12.1 | +0.87 ± 0.65 | 18.66 / 33.2 | +1.69 ± 3.31 | – | – |
| +GPS | 23.92 / 11.9 ⁽¹⁾ | +1.17 | 21.93 / 19.1 ⁽¹⁾ | +10.64 | – | – |
| +GPW | 23.99 / 10.5 ⁽¹⁾ | +1.24 | 21.90 / 19.0 ⁽¹⁾ | +10.61 | – | – |

- R = 같은 예산의 무작위 보호, H = HAWQ식 보호, GPS / GPW = M 대신 SmoothQuant / AWQ식 이전.

## 5. 그 밖의 현재 규칙 결과 (부록 후보)

| 분류 | 모델 | 프로토콜 | 비트 | 조건 | 꼬리표 | seed | LVIS AP / flip | Δ AP (vs BRECQ) | 출처 |
|---|---|---|---|---|---|---|---|---|---|
| 미지정 | YOLO-World-S | 확정 (attn8+A16) | W4A5 | brecq+GH | – | 0 1 2 | 21.55 / 20.8 | +4.58 ± 12.35 | 212_next |
| 미지정 | YOLO-World-S | 확정 (attn8+A16) | W4A8 | combined(s1=brecq) | – | 0 1 2 3 4 5 | 21.68 / 20.2 | -1.06 ± 0.20 | 158_final |
| 미지정 | OWLv2-B/16 | OWL | W4A8 | brecq+PW | – | 0 1 2 | 19.79 / 25.5 | +0.23 ± 1.65 | 185_owl_awq |
| 추가지표1 | YOLO-World-S | 확정 (attn8+A16) | W4A6 | brecq+PM | – | 0 | 23.10 / 15.0 | +1.77 | 234_rerun_s0 |
| 추가지표1 | YOLO-World-S | 확정 (attn8+A16) | W8A8 | brecq+M | – | 0 | 25.67 / 3.2 | +0.13 | 234_rerun_s0 |
| 추가지표1 | YOLO-World-S | 확정 (attn8+A16) | W8A8 | naive+S | – | 0 1 2 3 4 5 | 25.25 / 5.4 | -0.21 ± 0.07 | 170_sw |
| 추가지표1 | YOLO-World-S | 확정 (attn8+A16) | W8A8 | naive+W | – | 0 1 2 3 4 5 | 25.55 / 4.4 | +0.09 ± 0.07 | 170_sw |
| 추가지표1 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A5 | brecq+P | – | 0 1 2 | 16.70 / 44.3 | +3.48 ± 1.48 | 249_budget_palone |
| 추가지표1 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A5 | brecq+PM | – | 0 | 19.72 / 39.4 | +6.42 | 234_rerun_s0 |
| 추가지표1 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A8 | brecq+P | – | 0 1 2 | 20.44 / 26.9 | +4.67 ± 0.73 | 249_budget_palone |
| 추가지표1 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A8 | brecq+PM | – | 0 | 22.45 / 21.6 | +6.47 | 234_rerun_s0 |
| 추가지표1 | YOLO-World-S v2 | 확정 (attn8+A16) | W8A8 | brecq+PM | – | 0 | 23.25 / 7.7 | +0.56 | 234_rerun_s0 |
| 추가지표1 | YOLO-World-M | 확정 (attn8+A16) | W4A5 | brecq+G | – | 0 1 | 22.41 / 37.6 | +2.72 ± 5.78 | 154_protocol |
| 추가지표1 | YOLO-World-M | 확정 (attn8+A16) | W4A5 | brecq+P | – | 0 1 2 3 4 5 | 21.76 / 41.2 | +1.81 ± 0.35 | 154_protocol, 186_m_p |
| 추가지표1 | YOLO-World-M | 확정 (attn8+A16) | W4A8 | brecq+G | – | 0 1 | 25.57 / 24.3 | +1.14 ± 4.64 | 154_protocol |
| 추가지표1 | YOLO-World-M | 확정 (attn8+A16) | W4A8 | brecq+P | – | 0 1 2 3 4 5 | 25.96 / 23.0 | +1.58 ± 0.33 | 154_protocol, 186_m_p |
| 추가지표1 | YOLO-World-L | 확정 (attn8+A16) | W4A5 | brecq+G | – | 0 1 2 | 23.43 / 30.3 | +7.18 ± 2.41 | 212_next |
| 추가지표1 | YOLO-World-L | 확정 (attn8+A16) | W4A5 | brecq+P | – | 0 1 2 | 17.92 / 48.2 | +1.67 ± 0.93 | 249_budget_palone |
| 추가지표1 | YOLO-World-L | 확정 (attn8+A16) | W4A8 | brecq+P | – | 0 1 2 | 29.49 / 20.0 | +1.58 ± 1.33 | 249_budget_palone |
| 추가지표1 | OWLv2-B/16 | OWL | W4A8 | brecq+P | – | 0 1 2 | 19.26 / 27.7 | -0.31 ± 0.20 | 184_owl_main |
| 추가지표1 | Grounding DINO-T | GDINO | W4A5 | brecq+P | – | 0 | – | – | 216_gdino |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A4 | pdquant+GPM | – | 0 1 | 12.43 / 50.4 | +11.94 ± 8.26 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A4 | qdrop+GP | – | 0 | 0.48 / 96.9 | +0.03 | 234_rerun_s0 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A4 | qdrop+GPM | – | 0 1 2 | 13.95 / 42.7 | +13.52 ± 2.71 | 234_rerun_s0, 246_w4a4_seeds |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A5 | pdquant+GPM | – | 0 | 21.97 / 22.6 | +10.68 | 256_pdquant_s12 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A5 | qdrop+GPM | – | 0 1 2 | 22.35 / 17.8 | +5.38 ± 12.31 | 234_rerun_s0, 236_fill_s0 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A5 | qdrop+PM | – | 0 | 21.17 / 25.0 | +9.88 | 234_rerun_s0 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A6 | pdquant+GP | – | 0 | 22.11 / 21.8 | +0.78 | 256_pdquant_s12 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A6 | qdrop+GP | – | 0 1 2 | 22.92 / 16.6 | +1.41 ± 0.34 | 247_w4a6_s0, 251_w4a6_seeds |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A6 | qdrop+GPM | – | 0 | 23.17 / 14.0 | +1.84 | 247_w4a6_s0 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A6 | qdrop+PM | – | 0 | 22.69 / 15.4 | +1.36 | 234_rerun_s0 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A8 | pdquant+GPM | – | 0 1 | 23.83 / 15.8 | +1.07 ± 1.14 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A8 | qdrop+GPM | – | 0 1 2 | 24.48 / 11.1 | +1.77 ± 0.29 | 234_rerun_s0, 236_fill_s0 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W4A8 | qdrop+PM | – | 0 | 24.10 / 12.7 | +1.35 | 234_rerun_s0 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W8A8 | pdquant+GPM | – | 0 1 | 25.65 / 6.3 | +0.13 ± 0.19 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W8A8 | qdrop+GPM | – | 0 1 2 | 25.79 / 2.7 | +0.30 ± 0.09 | 234_rerun_s0, 240_w8seeds_w4a4 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) | W8A8 | qdrop+M | – | 0 | 25.63 / 3.1 | +0.09 | 234_rerun_s0 |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) · calib o365 | W4A5 | brecq | – | 0 1 2 | 18.99 / 29.6 | – | 208_calib_src |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) · calib o365 | W4A8 | brecq | – | 0 1 2 | 22.88 / 17.2 | – | 208_calib_src |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) · calib o365gqa | W4A5 | brecq | – | 0 1 2 | 19.15 / 28.3 | – | 209_calib_o365gqa |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) · calib o365gqa | W4A8 | brecq | – | 0 1 2 | 22.86 / 16.8 | – | 209_calib_o365gqa |
| 추가지표2 | YOLO-World-S | 확정 (attn8+A16) · 첫·마지막 FP | W4A4 | brecq | – | 0 1 2 | 0.42 / 97.2 | – | 203_w4a4, 214_w4a4_s2 |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A4 | pdquant+GPM | – | 0 1 | 5.37 / 78.3 | +5.15 ± 0.89 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A4 | qdrop+GP | – | 0 | 0.32 / 98.6 | +0.11 | 240_w8seeds_w4a4 |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A4 | qdrop+GPM | – | 0 1 2 | 6.12 / 77.4 | +5.91 ± 2.09 | 240_w8seeds_w4a4, 246_w4a4_seeds |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A5 | pdquant+GPM | – | 0 | 16.42 / 39.0 | +3.12 | 256_pdquant_s12 |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A5 | qdrop+GPM | – | 0 1 2 | 16.78 / 35.8 | +3.56 ± 1.21 | 236_fill_s0 |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A6 | pdquant+GPM | – | 0 | 20.29 / 29.8 | +4.57 | 256_pdquant_s12 |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A6 | qdrop+GP | – | 0 | 17.25 / 35.0 | +1.53 | 247_w4a6_s0 |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A6 | qdrop+GPM | – | 0 1 2 | 21.05 / 28.3 | +5.63 ± 0.64 | 247_w4a6_s0, 251_w4a6_seeds |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A8 | pdquant+GPM | – | 0 1 | 19.01 / 28.1 | +3.13 ± 1.02 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W4A8 | qdrop+GPM | – | 0 1 2 | 19.43 / 25.1 | +3.66 ± 0.82 | 236_fill_s0 |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W8A8 | pdquant+GPM | – | 0 1 | 22.47 / 10.8 | -0.20 ± 0.70 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-S v2 | 확정 (attn8+A16) | W8A8 | qdrop+GPM | – | 0 1 2 | 22.53 / 7.9 | -0.20 ± 0.27 | 238_w8a8, 240_w8seeds_w4a4 |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) | W4A4 | pdquant+GPM | – | 0 1 | 10.97 / 76.6 | +10.36 ± 15.76 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) | W4A4 | qdrop+GPM | – | 0 1 2 | 11.49 / 74.2 | +10.84 ± 3.91 | 240_w8seeds_w4a4, 246_w4a4_seeds |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) | W4A5 | pdquant+GP | – | 0 | 18.24 / 54.6 | -1.20 | 256_pdquant_s12 |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) | W4A5 | qdrop+GP | – | 0 1 2 | 16.73 / 57.2 | -3.10 ± 2.83 | 236_fill_s0 |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) | W4A6 | pdquant+GP | – | 0 | 24.62 / 35.6 | +1.14 | 256_pdquant_s12 |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) | W4A6 | qdrop+GP | – | 0 1 2 | 24.52 / 32.4 | +1.44 ± 0.52 | 247_w4a6_s0, 251_w4a6_seeds |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) | W4A6 | qdrop+GPM | – | 0 | 18.24 / 59.4 | -5.24 | 247_w4a6_s0 |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) | W4A8 | pdquant+GP | – | 0 1 | 25.61 / 29.0 | +1.18 ± 4.96 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) | W4A8 | qdrop+GP | – | 0 1 2 | 26.53 / 22.2 | +2.09 ± 0.88 | 236_fill_s0 |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) | W8A8 | pdquant+GP | – | 0 1 | 28.66 / 15.7 | -0.19 ± 0.89 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) | W8A8 | qdrop+GP | – | 0 1 2 | 28.69 / 10.2 | +0.20 ± 1.55 | 238_w8a8, 240_w8seeds_w4a4 |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) · 첫·마지막 FP | W4A4 | brecq | – | 0 1 2 | 0.76 / 97.5 | – | 205_w4a4_m_abl, 214_w4a4_s2 |
| 추가지표2 | YOLO-World-M | 확정 (attn8+A16) · 첫·마지막 FP | W4A4 | brecq+GPM | – | 0 | 15.99 / 64.4 | +14.87 | 234_rerun_s0 |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W4A4 | pdquant+GPM | – | 0 1 2 | 14.74 / 54.5 | +14.70 ± 4.16 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W4A4 | qdrop+GP | – | 0 | 0.00 / 99.1 | -0.06 | 240_w8seeds_w4a4 |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W4A4 | qdrop+GPM | – | 0 1 2 | 16.76 / 49.3 | +16.72 ± 4.00 | 240_w8seeds_w4a4, 246_w4a4_seeds |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W4A5 | pdquant+GPM | – | 0 | 23.80 / 30.3 | +8.43 | 256_pdquant_s12 |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W4A5 | qdrop+GPM | – | 0 1 2 | 26.87 / 21.8 | +10.63 ± 1.36 | 234_rerun_s0, 236_fill_s0 |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W4A6 | pdquant+GPM | – | 0 | 29.54 / 19.6 | +3.20 | 256_pdquant_s12 |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W4A6 | qdrop+GP | – | 0 | 28.49 / 20.1 | +2.15 | 247_w4a6_s0 |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W4A6 | qdrop+GPM | – | 0 1 2 | 31.21 / 13.0 | +5.64 ± 1.91 | 247_w4a6_s0, 251_w4a6_seeds |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W4A8 | pdquant+GPM | – | 0 2 | 30.61 / 17.4 | +2.48 ± 2.16 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W4A8 | qdrop+GPM | – | 0 1 2 | 32.05 / 9.6 | +4.14 ± 1.29 | 234_rerun_s0, 236_fill_s0 |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W8A8 | pdquant+GPM | – | 0 1 2 | 32.70 / 8.9 | -0.62 ± 0.34 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W8A8 | qdrop+GP | – | 0 | 33.53 / 3.9 | +0.16 | 238_w8a8 |
| 추가지표2 | YOLO-World-L | 확정 (attn8+A16) | W8A8 | qdrop+GPM | – | 0 1 2 | 33.22 / 3.6 | -0.11 ± 0.23 | 238_w8a8, 240_w8seeds_w4a4 |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W4A4 | pdquant+GPM | – | 0 | 18.99 / 32.6 | +14.35 | 255_pdquant_s0 |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W4A4 | qdrop+GP | – | 0 | 0.05 / 96.0 | -4.59 | 240_w8seeds_w4a4 |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W4A4 | qdrop+GPM | – | 0 1 2 | 19.24 / 29.4 | +15.80 ± 4.61 | 240_w8seeds_w4a4, 246_w4a4_seeds |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W4A5 | pdquant+GPM | – | 0 | 23.27 / 21.5 | +6.09 | 256_pdquant_s12 |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W4A5 | qdrop+GPM | – | 0 1 2 | 23.37 / 16.5 | +5.57 ± 2.77 | 236_fill_s0 |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W4A6 | pdquant+GPM | – | 0 | 24.37 / 18.6 | +1.09 | 256_pdquant_s12 |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W4A6 | qdrop+GP | – | 0 | 23.01 / 16.3 | -0.27 | 247_w4a6_s0 |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W4A6 | qdrop+GPM | – | 0 1 2 | 24.13 / 13.8 | +0.96 ± 0.77 | 247_w4a6_s0, 251_w4a6_seeds |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W4A8 | pdquant+GP | – | 0 | 24.90 / 15.7 | +1.19 | 255_pdquant_s0 |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W4A8 | qdrop+GP | – | 0 1 2 | 24.40 / 10.5 | +0.71 ± 0.47 | 236_fill_s0 |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W8A8 | pdquant+GPM | – | 0 | 27.05 / 7.2 | +1.26 | 255_pdquant_s0 |
| 추가지표2 | YOLOE-v8s | 확정 (attn8+A16) | W8A8 | qdrop+GPM | – | 0 1 2 | 25.84 / 1.9 | +0.03 ± 0.16 | 238_w8a8, 240_w8seeds_w4a4 |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W4A4 | pdquant+GPM | – | 0 | 8.11 / 61.8 | +7.77 | 255_pdquant_s0 |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W4A4 | qdrop+GP | – | 0 | 0.22 / 96.2 | -0.12 | 240_w8seeds_w4a4 |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W4A4 | qdrop+GPM | – | 0 1 2 | 9.01 / 56.7 | +8.54 ± 1.76 | 240_w8seeds_w4a4, 246_w4a4_seeds |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W4A5 | pdquant+GPM | – | 0 | 14.80 / 46.6 | +3.24 | 256_pdquant_s12 |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W4A5 | qdrop+GPM | – | 0 1 2 | 17.01 / 36.5 | +3.76 ± 5.29 | 236_fill_s0 |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W4A6 | pdquant+GPM | – | 0 | 20.43 / 30.8 | -0.53 | 256_pdquant_s12 |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W4A6 | qdrop+GP | – | 0 | 20.68 / 25.7 | -0.28 | 247_w4a6_s0 |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W4A6 | qdrop+GPM | – | 0 1 2 | 21.73 / 24.3 | +0.92 ± 0.37 | 247_w4a6_s0, 251_w4a6_seeds |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W4A8 | pdquant+GP | – | 0 1 | 22.76 / 24.1 | -0.17 ± 4.26 | 255_pdquant_s0, 256_pdquant_s12 |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W4A8 | qdrop+GP | – | 0 1 2 | 23.23 / 18.0 | +0.41 ± 0.90 | 236_fill_s0 |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W8A8 | pdquant+GPM | – | 0 | 26.93 / 11.4 | +0.39 | 255_pdquant_s0 |
| 추가지표2 | YOLOE-11s | 확정 (attn8+A16) | W8A8 | qdrop+GPM | – | 0 1 2 | 26.65 / 2.6 | +0.10 ± 0.21 | 238_w8a8, 240_w8seeds_w4a4 |
| 추가지표2 | OWLv2-B/16 | OWL | W4A8 | qdrop+PM | – | 0 | 26.63 / 25.2 | +6.89 | 252_fill |

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

- §1 메인 표: `154_protocol`, `158_final`, `160_gen6`, `164_yoloe`, `171_yoloe11`, `203_w4a4`, `211_L_main`, `212_next`, `214_w4a4_s2`, `219_floor_verify`, `223_floor_rerun`, `229_budget_seeds_S`, `234_rerun_s0`, `235_v2_refill_check`, `236_fill_s0`, `238_w8a8`, `240_w8seeds_w4a4`, `246_w4a4_seeds`, `255_pdquant_s0`, `256_pdquant_s12`
- §2 표 1: `154_protocol`, `158_final`, `170_sw`, `203_w4a4`, `214_w4a4_s2`, `229_budget_seeds_S`, `234_rerun_s0`, `236_fill_s0`, `240_w8seeds_w4a4`, `246_w4a4_seeds`, `247_w4a6_s0`, `251_w4a6_seeds`, `255_pdquant_s0`, `256_pdquant_s12`
- §3 ViT·DETR: `184_owl_main`, `234_rerun_s0`, `237_gdino_lvis2000`, `250_owl_seeds`, `252_fill`
- §4 ablation: `154_protocol`, `158_final`, `203_w4a4`, `205_w4a4_m_abl`, `212_next`, `214_w4a4_s2`, `229_budget_seeds_S`, `233_noG_repro`, `234_rerun_s0`, `236_fill_s0`, `246_w4a4_seeds`, `253_seg_w4a4_abl_w4a8`, `253b_rerun_killed`
- 칸별 log와 저장 모델 경로는 csv의 `log`, `save_model` 열에 있다.
