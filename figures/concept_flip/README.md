# 개념 그림: 점수 오차가 아니라 판정(top-1) 뒤집힘 (11-13)

메시지: 같은 크기의 양자화 오차도 어휘가 커지면 top-1을 뒤집는다. 우리 방법은 오차를 고르게 줄이는 대신, 판정을 뒤집는 오차가 나오는 층(Δ_ℓ로 진단)을 막는다.

| 파일 | 내용 | 데이터 |
|---|---|---|
| `concept_A.{pdf,png}` | 결정 경계(Voronoi) 3칸: (a) 작은 어휘 (b) LVIS 어휘에서 같은 ‖e‖로 flip (c) 진단된 층 성분 e_ℓ* 제거 | **합성 좌표**(설명용, 실측 아님) |
| `concept_B.{pdf,png}` | 실측 점수: helmet 영역에서 FP / BRECQ / Ours의 클래스별 점수. BRECQ만 cap이 1등 | `data/scores_450559_helmet.json` |
| `concept_C.{pdf,png}` | A의 (a)(b) + B | 위 둘 |

- 실측 조건: YOLO-World-S W4A8 seed 0, LVIS minival 이미지 450559, LVIS 1203 어휘. FP에서 helmet 점수가 가장 높은 앵커(4546) 하나의 sigmoid 점수. BRECQ = `qmodels/yolov8s-world_w4a8_s0_brecq.pt`, Ours = `qmodels/r1102/yolov8s-world_w4a8_s0_brecq-GPM.pt`.
- 실측에서 BRECQ는 helmet이 0.87 → 0.64로 내려가고, 비슷한 모자류(cap 0.08 → 0.65, hat, baseball cap)가 함께 올라가 cap이 0.007 차이로 1등이 된다.
- 주의: A (c)의 "한 층 성분이 크다"는 진단(Δ_ℓ, 12.cv2 30.8%p)을 개념으로 옮긴 것이다. 특징 공간에서 층별 오차 벡터를 실제로 분해해 잰 것은 아니다.

다시 만들기(저장소 루트):
```
CUDA_DEVICE_ORDER=PCI_BUS_ID .venv/bin/python figures/concept_flip/src/dump_scores.py --device <빈 GPU>   # B의 점수(이미지 1장, 수십 초)
.venv/bin/python figures/concept_flip/src/make_concept.py                                              # A, B, C (CPU)
```

## 그림 1 후보: fig1_voronoi(_ap) (11-13 수정)

- 설정: YOLO-World-S W4A8 seed 0 전체 모델(서론·초록과 같은 수치). 칸 모양과 (a) 클래스 이름은 설명용.
- flip(실측, 리더보드 seed 0, LVIS minival 전체): BRECQ COCO 어휘 3.94% / LVIS 어휘 17.49%, Ours LVIS 10.69%.
- 원 반경 비율(실측, `data/w4a8_emb_flip_n500.json`): 영역 임베딩 오차 ‖e‖² BRECQ 0.0702 → Ours 0.0373(×0.53, 반경 ×0.73).
  정의: L2 정규화 영역 임베딩의 FP 대비 제곱 거리, FP 확신 위치(τ=0.25) 가중 평균, LVIS minival 앞 500장(`core/diag_w4_sensitivity.measure`).
  같은 500장의 flip: BRECQ COCO 4.33 / LVIS 16.39%, Ours 2.10 / 10.60%.
- (b)(c) 칸이 같은 근거: head의 텍스트 임베딩은 8bit 상수(`quant/attn_quant.py`, 데이터 불필요)라 BRECQ와 Ours에서 똑같다(FP와는 8bit만큼 다름).
- (a)를 COCO 클래스로 그리지 않은 이유: 이 helmet 영역을 COCO 80 어휘로 채점하면 모든 클래스 점수 ≤ 0.002(`data/coco_scores_450559.json`), 즉 COCO 어휘에서는 배경이다.
- 측정 스크립트: `src/measure_w4a8.py`(GPU, 수 분), 그림: `src/make_fig1_voronoi.py`(CPU).
