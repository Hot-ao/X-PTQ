# Baseline 확정 상태 및 재현 절차

> **2026-10-08 — 이 문서는 10-02 확정 프로토콜 이전(head 제외, W8A8)의 기록이다.** 지금 프로토콜과 표는
> [`README.md`](README.md)와 [`docs/RESULTS.md`](docs/RESULTS.md)를 볼 것.
> 아래에 나오는 `run_comparison.py` 인자 가운데 `--neck-layerwise`, `--brecq-batch`, `--brecq-two-stage`,
> `--act-observer`, `--recon-iters-*`, `--scale-reg-weight` 등 확정 대기열이 쓰지 않는 것은 10-08 정리에서 제거했다
> (값은 `run_comparison.py` 상단 상수와 `build()`에 고정). 그 인자로 재현하려면 커밋 `282854d`의 코드를 쓴다.

**최종 갱신: 2026-09-28** · 대상 독자: 이 파이프라인으로 baseline 수치를 재현하거나,
baseline을 기준으로 제안 방법을 설계하는 사람.

이 문서는 **"지금 baseline이 어떤 상태이고, 어떻게 재현하며, 무엇이 남았는가"** 를 다룬다.
각 방법이 원 논문/공식 코드와 항목별로 어떻게 대응되는지는 원 저장소(promptcal-ptq) 루트의
`PROMPTCAL_BASELINE_FIDELITY_2026-09-23.md`
(특히 09-24에 추가된 **§7 6-seed 확정 재측정**)를 볼 것. 여기서는 그 결론만 요약한다.

---

## 1. 확정 baseline (`runs/97_ada_iters2000_6seed`, 6-seed 평균)

| | COCO_AP | Heval_flip | Top1_flip | lost | LVIS_AP |
|---|---|---|---|---|---|
| naive | 36.617 | 5.945% | 0.667% | 291 | 0.2554 |
| AdaRound | 36.648 | 5.142% | 0.472% | 234 | **0.2577** |
| QDrop | **36.807** | 4.090% | 0.397% | 180 | 0.2556 |
| BRECQ | 36.755 | **3.705%** | **0.345%** | **166** | 0.2560 |
| *(참고) Combined* | 36.642 | 5.602% | 0.527% | 248 | 0.2576 |

- 이론적 모델 크기 **9.88 MiB** (W8, detector conv만. 이전에 보고하던 12.13 MiB는
  CLIP 인코더가 섞인 값이었다 — §3 참고)
- **제안 방법이 넘어야 할 선은 BRECQ의 `Heval_flip 3.705% / lost 166`.**
  seed별 짝비교에서 Combined는 naive에만 전승이고 AdaRound·QDrop·BRECQ에게
  decision 지표 0/6 전패다. 유일한 우세인 LVIS_AP(+0.0016~0.0020, 6/6)도
  AdaRound(0.2577)가 거의 같다.

> ⚠️ **09-28 — 위 '(참고) Combined' 행과 마지막 bullet은 stage1=none(naive 기반) 시절 값이라 stale이다.**
> 현재 설계는 `--combined-stage1 brecq`이고 그 위에서 LVIS 3지표가 6/6 개선된다(`runs/112`).
> 현재 설계·결과는 원 저장소의 `promptcal-ptq/PROMPTCAL_PAPER_DESIGN_2026-09-28.md`를 볼 것.
> 위 baseline 4행 자체는 유효하다(head 제외 프로토콜).

### 1.1 head 포함 프로토콜 (09-28, claim23)

`--no-skip-head`로 `WorldDetect` 전체(`cv2` 박스 회귀 9 + `cv3` region 임베딩 9 = conv 18개)를 양자화.
W8A8, calib 256, full probe, `--deterministic`, **6-seed 평균**, 전부 RTX 4000 Ada.
**AdaRound는 MSE observer**(아래 §4.1). 이론적 모델 크기 **12.08 MiB**.

| | COCO_AP | Heval_flip | Top1_flip | lost | LVIS_flip | LVIS_lost | LVIS_AP | LVIS_APr |
|---|---|---|---|---|---|---|---|---|
| naive | 36.36 | 7.94 | 0.74 | 332 | 5.39 | 1197 | 0.2540 | 0.1771 |
| AdaRound(MSE) | 36.50 | 7.01 | 0.55 | 261 | 4.64 | 1049 | 0.2521 | 0.1751 |
| QDrop | 36.54 | 6.35 | 0.49 | 220 | 3.88 | 840 | 0.2538 | 0.1805 |
| BRECQ | 36.29 | 6.52 | 0.48 | 231 | 3.97 | 861 | 0.2514 | 0.1784 |
| **Combined(stage1=brecq)** | **36.58** | 7.56 | 0.51 | 256 | 3.87 | 717 | **0.2555** | 0.1781 |

head를 넣으면 **BRECQ가 naive보다도 나빠지고**(COCO_AP·LVIS_AP 모두) 재구성 기반 3종은 flip/lost는 줄이나 LVIS_AP는
naive를 못 넘는다. Combined vs BRECQ 짝비교: COCO_AP 6/6(+0.29), LVIS_AP 6/6(+0.0041), LVIS_lost 5/6(−144),
**Heval_flip 0/6, lost 0/6(악화)**. 출처는 조합이다 — seed 0·1의 naive/QDrop/BRECQ/Combined = `runs/115`,
seed 2~5 전 조건 = `runs/121`, AdaRound(MSE) = seed 0 `runs/121`(adaround만), seed 1 `runs/117`, seed 2~5 `runs/121`.
seed 0의 naive는 `runs/121`과 `runs/115`가 bit-identical함을 확인했다.

**측정 주의**: `runs/116` seed 0은 `--device 0`이 GPU 1(L40S)로 잡혀 GPU 종류가 달랐다(naive가 36.54 vs 36.30) —
**표에 쓰지 않는다**(seed 1만 유효).

---

## 2. 재현 절차

### 2.1 환경

**기존 editable ultralytics 설치가 가리키던 `/home/taeho/Mamba-YOLO`가 삭제되어
시스템 python으로는 import 자체가 안 된다.** 저장소 안의 전용 venv를 쓴다(gitignore됨).

```
ultralytics 8.4.121 | torch 2.10.0+cu128 | torchvision 0.25.0+cu128
numpy 1.26.4 | opencv 4.13.0 | lvis 0.5.3 | pycocotools 2.0.11 | faster_coco_eval 1.8.0
```

**GPU 인덱스 주의 (2026-09-28 추가).** 이 서버는 L40S 3장(nvidia-smi 1,2,3)과
RTX 4000 Ada 5장(0,4,5,6,7)이 섞여 있는데, torch 기본값
`CUDA_DEVICE_ORDER=FASTEST_FIRST`가 L40S를 먼저 정렬해서 `--device N`이
nvidia-smi 인덱스와 어긋난다(`0→1, 1→2, 2→3, 3→0`, 4~7만 일치). 실제로
`runs/116` seed0이 `--device 0`으로 GPU 1에 올라갔다. **공용 서버라 남이 쓰는
GPU를 덮칠 수 있는 사고**이므로 두 곳에서 막아뒀다:

- `.venv/lib/python3.12/site-packages/sitecustomize.py` — 이 venv로 실행되는
  모든 프로세스에 `CUDA_DEVICE_ORDER=PCI_BUS_ID`를 강제(`scripts/` 아래
  entry point가 59개라 파일마다 고치지 않고 한 곳에서 막음). venv를 새로
  만들면 **이 파일도 다시 만들어야 한다.**
- `run_comparison.py` 상단에 같은 `setdefault` + 시작 시
  `[gpu] --device N -> ... uuid=...` 로그. 로그의 인덱스가 nvidia-smi와
  다르면 환경이 바뀐 것이니 멈추고 확인할 것.

부수 효과: GPU **종류가 섞이면 같은 seed도 bit-identical하지 않을 수 있다**
(L40S와 RTX 4000 Ada는 커널 선택이 다르다). seed 간 비교를 하는 run은
**같은 모델의 GPU에 몰아서** 돌릴 것.

venv를 새로 만들어야 한다면:

```bash
cd /home/taeho/CVPR2027
python -m venv --system-site-packages .venv          # torch/numpy/opencv는 ~/.local 것을 물려받는다
.venv/bin/pip install --no-deps ultralytics==8.4.121 lvis pycocotools ftfy regex "filelock>=3.18"
.venv/bin/python -c "import torch,torchvision,numpy; print(torch.__version__, torchvision.__version__, numpy.__version__)"
# 2.10.0+cu128 0.25.0+cu128 1.26.4 이어야 한다
```

**`--no-deps`가 중요하다.** 빼면 pip이 torch 2.14 스택 전체와 numpy 2.5.3을 venv에
따로 깔아 ~/.local의 torch 2.10과 충돌한다(`operator torchvision::nms does not exist`).

### 2.2 실행

```bash
export YOLO_AUTOINSTALL=false     # 없으면 ultralytics가 numpy 2.5.3을 다시 덮어쓴다

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
.venv/bin/python run_comparison.py \
  --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
  --data configs/coco_local.yaml \
  --lvis-ann /data/taeho/lvis_datasets/labels_dl/extracted/lvis/annotations/lvis_v1_minival.json \
  --calib 256 --recon-iters-ada 2000 \
  --conditions naive,adaround,qdrop,brecq,combined \
  --seed <0..5> --device 0
```

**`--recon-iters-ada 2000`이 확정 설정이다**(기본값 1000이 아님, §4.1 참고).
나머지는 전부 기본값.

**09-28 — 현재 제안 방법(Combined)을 측정하려면** 위 커맨드에 아래를 더한다. `--combined-stage1`의 기본값은
코드상 여전히 `none`이지만 **확정 설계는 `brecq`**이므로 반드시 명시해야 한다(안 하면 naive 기반 Combined가 나온다).

```bash
  --deterministic --combined-stage1 brecq --combined-recon-iters 2000
  # head 포함 프로토콜(§1.1)이면 추가로:
  --no-skip-head --adaround-act-observer mse
```

`--deterministic`은 Combined를 측정하는 모든 실행에서 필수다(§5.1). GPU는 RTX 4000 Ada(0,4,5,6,7)에만 몰아서 쓸 것 —
종류를 섞으면 seed 간 bit-identical이 깨진다.

### 2.3 소요 시간 (RTX 4000 Ada 20GB 기준, 실측)

- 5개 seed 병렬(GPU 5장): **약 5시간 33분**
- 단독 실행 1개 seed: **약 1시간 25분**
- 6 seed를 GPU 5장으로 돌리면 총 **약 7시간**
- GPU당 peak 약 15GB, 프로세스당 CPU RAM 약 20GB (5-way에서 시스템 110/125GB)
- seed0 빌드 내역: naive 7s / AdaRound 4304s / QDrop 2988s / BRECQ 1742s / Combined 109s

---

## 3. 이번 세션에 바꾼 것

| 파일 | 변경 | 결과 영향 |
|---|---|---|
| `quant/quant_model.py` (+ promptcal-ptq의 `src/quant/` 사본) | `ALWAYS_SKIP_NAMES = {"clip_model"}` — `wrap_convs`가 CLIP 인코더로 재귀하지 않게 | **정확도 무영향**(6-seed로 naive bit-identical 확인). 모델 크기 12.13 → **9.88 MiB** |
| `run_comparison.py` | `--adaround-act-observer {minmax,mse}` 추가 | 기본 `minmax` = 기존 동작과 bit-identical |
| `run_comparison.py` | `--deterministic` 추가 | 기본 꺼짐, 무영향 |

**CLIP 건 상세**: `set_classes()`가 `WorldModel.clip_model`에 CLIP(ViT-B/32) 전체를
캐싱하는데, `wrap_convs`가 그 vision tower의 patch-embed conv
(`clip_model.model.visual.conv1`, 3→768 k32, weight 2,359,296)까지 `QuantConv2d`로
감싸고 있었다. detector forward에서 호출되지 않아 정확도에는 영향이 없었지만
`quantized_weight_mib()`에 2.25 MiB(22.8%)가 허수로 더해졌고, AdaRound가 매 run
`1/53 conv 입력 미포착` 경고를 내며 건너뛰던 것의 정체가 이거였다.
`scripts/` 아래 호출처가 50곳 가까이 되어 호출처마다 고치지 않고 `wrap_convs` 안에서
DFL과 같은 방식으로 막았다.

**논문에 모델 크기를 인용했다면 9.88 MiB로 정정할 것.**

### 3.1 09-24~09-28에 추가된 `run_comparison.py` 플래그

**전부 기본 꺼짐이고, 끄면 기존 결과와 bit-identical이다**(`naive`가 매번 3/3~6/6 seed에서 일치함을 확인).
기본 켜짐인 것은 없다 — 켜야 하는 것은 위 §2.2의 세 개(`--deterministic`, `--combined-stage1 brecq`,
`--combined-recon-iters 2000`)뿐이다.

| 플래그 | 용도 | 결론 |
|---|---|---|
| `--deterministic` | `torch.use_deterministic_algorithms(warn_only=True)`. Combined 재현성 | **Combined 측정 시 필수** |
| `--adaround-act-observer {minmax,mse}` | AdaRound의 activation observer | head 제외: minmax(as-published) / **head 포함: mse** |
| `--no-skip-head` | `WorldDetect` 전체 양자화 | §1.1 |
| `--combined-stage1 {none,adaround,brecq}` + `--combined-recon-iters` | Combined 1단계 | **`brecq`, 2000이 확정** |
| `--combined-learn-alpha`(+`-lr`,`-reg-weight`) | Stage 2에서 rounding alpha 학습 | 기각(claim18-a, 21) |
| `--combined-learn-alpha-bias`(+`-lr`,`-reg-weight`,`-limit`) | conv당 스칼라로 rounding 보정 | **기각**(claim22, W4A8 3/3 붕괴) |
| `--combined-block-recon-weight` | Stage 2에 블록 재구성 항 추가(공동 최적화) | 기각(claim21) |
| `--combined-range-blend` | activation 범위를 MSE↔min-max로 보간 | 기각(claim18-b) |
| `--combined-utility-frac` / `-thresh-w` / `-box-w` | 논문 §4.3 utility 항 | 기각(claim19) |
| `--combined-region-dir-weight` | region 방향 보존 항 | 6-seed 무효(claim20) |
| `--margin-one-sided`, `--combined-random-sample`, `--combined-per-group-anchors`, `--combined-local-recon-weight` | 손실 항 결함 수정 시도 | 측정상 이득 없음/악화 |

**⚠️ 미해결 — Stage 1 설정 불일치**: Combined의 Stage 1(`optimize_brecq` 호출)은 `neck_layerwise`를 안 넘겨 함수 기본값
`True`(neck conv 단위 재구성)를 받는데, 기준선 BRECQ는 CLI 기본값(block 단위)이다. 의도된 차이가 아니며 Combined-vs-BRECQ
짝비교에 미격리 교란으로 남아 있다. 자세한 내용과 닫는 방법은 설계 문서 §2·§10.

---

## 4. baseline 확정 — 결정된 것 / 남은 것

### 4.1 결정됨

| 항목 | 결정 | 근거 |
|---|---|---|
| **AdaRound iteration 예산** | **2000** (QDrop/BRECQ와 동일) | 1000은 논문 충실도 근거가 없는 그냥 절반 예산이었다(논문 20k, AIMET 10k). 2000으로 맞추니 decision 지표가 실제 개선(Heval_flip 5.93→5.14%, lost 263→234) — 음성 대조군 대비 약 7배. 루트 문서 §7.1 |
| **AdaRound activation observer** | **min-max 유지**(as-published) | 교란변수가 아니라 **COCO↔LVIS 트레이드오프**였다. MSE로 바꾸면 COCO는 6/6 개선, LVIS는 6/6 악화(LVIS_lost +29%). min-max에서 AdaRound의 LVIS_AP 0.2577이 전 baseline 1위라 불이익도 아니다. MSE판은 ablation으로 병기. 루트 문서 §7.2 |
| **BRECQ iters 2000** | 유지 | 20000과 최종 지표 차이가 노이즈 수준, 비용만 6.2배. 단 alpha는 수렴하지 않았다(nearest 대비 flip 5.2%→12.9%) — "수렴해서 충분"이 아니라 "수렴 안 했는데도 지표가 같아서"로 서술할 것. 루트 문서 §6.2 |
| **BRECQ 2단계 분리** | **쓰지 않음**(공동 최적화) | 공식대로 켜면 Heval_flip 3.70→5.10%, lost 180→210으로 명확히 악화. 시간이 아니라 **성능** 때문이다. 루트 문서 §6.3 |
| **QDrop batch 2 / neck layer-wise** | **쓰지 않음** | 차이가 노이즈로 설명된다. 비용만 2.7배. 논문 부록 E와 어긋나는 점은 각주로 밝힐 것. 루트 문서 §6.3 |
| **CLIP 인코더 제외** | 적용 완료 | §3 |
| **head 포함 시 AdaRound observer** | **MSE** | min-max는 head activation의 outlier를 클리핑 없이 담다가 붕괴한다(COCO_AP 36.6→33.4, 2/2 seed). observer 하나만 바꾼 단일변수 실험(`runs/117`)으로 회복 확인(36.59). head 제외 프로토콜에서는 위 min-max 결정이 그대로 유효 |

### 4.2 남은 것

| 항목 | 상태 | 닫는 방법 | 비용 | 중요도 |
|---|---|---|---|---|
| **AdaRound batch** | 공식 32, 우리 1. 미측정 | `optimize_adaround(batch=...)`를 CLI로 노출하고 1-seed 비교. 유의미하면 6-seed | 1-seed ≈ 1.5h | 낮음 — AdaRound만 해당하고, 예산 정렬로 이미 개선됨 |
| **QDrop iters 수렴** | 미측정(BRECQ만 확인) | `--conditions naive,qdrop --recon-iters-strong 20000` 1-seed. QDrop은 drop 때문에 수렴 양상이 BRECQ와 다를 수 있다 | 1-seed ≈ 2h | **중간** — QDrop은 넘어야 할 상대라 신뢰도가 필요 |
| **AdaRound batch·QDrop iters 외 미검증 없음** | — | — | — | — |

### 4.3 확정 완료 판정

**넘어야 할 상대인 BRECQ/QDrop 쪽이 가장 단단하다** — 구현이 공식 소스와 수식 수준에서
일치하고(루트 문서 §2·§3), 설정도 재측정으로 닫혔으며, 실행 분산이 0이다.
남은 두 항목은 AdaRound batch(영향 낮음)와 QDrop iters(확인 권장)뿐이므로,
**baseline을 기준자로 쓰기 시작해도 되는 상태다.**

QDrop iters만 1-seed로 확인해 두면 baseline 쪽은 사실상 종결이다.

---

## 5. 측정 신뢰성 — 반드시 알고 있어야 할 3가지

### 5.1 baseline은 결정적이다 (Combined는 아니다)

naive/AdaRound/QDrop/BRECQ는 **독립된 run 사이에서 bit-identical**이다
(`runs/91`↔`92`, `runs/97`↔`98` 6-seed 전부 확인). 따라서 **6-seed 분산 전체가 seed
효과**(클래스 분할 + torch RNG)이고 실행 노이즈가 0이다 — 기준자로서 필요한 성질.

> **09-24 정정 — `--deterministic`을 켠 BRECQ는 예외다.** `runs/99`↔`runs/101`(5 run)에서
> `naive`는 7/7 bit-identical인데 `brecq`만 두 값 중 하나에 무작위로 떨어졌다
> (AP 36.74↔36.75, Heval_flip 3.72↔3.68, lost 190↔188). `--deterministic`에서도 결정적
> 구현이 없어 경고만 뜨는 `adaptive_max_pool2d_backward_cuda`(`ImagePoolingAttn`의
> `AdaptiveMaxPool2d`)가 유력하다 — BRECQ는 `ImagePoolingAttn`을 블록 타깃으로 재구성하므로
> 그 backward를 정면으로 통과한다. 위 6-seed 확정값(`runs/97`, `--deterministic` 미사용)은
> 영향 없고, 변동폭도 통상 측정 대상의 1/25 수준이다. 자세한 내용은 claim18-d.

**반면 Combined는 같은 seed·같은 코드·같은 RNG 스트림에서도 결과가 달라졌다**
(`runs/97`↔`98`에서 seed 2·4, 2/6). 원인을 격리한 결과(09-24):

| full scale(calib 256, iters 1500) Combined를 같은 seed로 2회 빌드 | s_mult 불일치 | max\|Δ\| | 빌드 |
|---|---|---|---|
| 기본 | **52/52 conv** | 2.48e-01 | 111s / 114s |
| `torch.use_deterministic_algorithms(True, warn_only=True)` | **0/52** | — | 114s / 115s |

**→ `--deterministic`으로 완전히 해결된다.** `cudnn.deterministic=True`는 conv
알고리즘만 고정할 뿐 backward의 atomicAdd 계열을 못 잡는다. (`--deterministic`
상태에서도 `adaptive_max_pool2d_backward_cuda`(ImagePoolingAttn의 AdaptiveMaxPool2d)만
결정적 구현이 없다는 경고가 남지만, 그건 범인이 아니다 — 그 상태로 0/52가 나온다.)

**baseline 수치는 `--deterministic`으로 바뀌지 않는다.** AdaRound/BRECQ/QDrop을
켜고/끄고 빌드해 비교하면 quant weight와 LSQ delta가 전부 bit-identical이다
(52/52 동일, iters=200·calib=64). **즉 §1의 확정 표는 그대로 유효하고 재측정이
필요 없다.** 빌드가 느려지는 비용만 있다(Combined +2~3%, baseline +13~47%
— baseline은 어차피 이미 결정적이라 켤 필요가 없다).

→ **Combined를 다루는 모든 실행에 `--deterministic`을 켤 것.** 켜면 Combined도
실행 분산 0이 되어, 설계 A vs B를 같은 seed에서 짝비교하면 차이가 전부 실재한다.

### 5.2 `--conditions` 목록이 RNG 스트림 위치를 정한다

`torch.manual_seed`는 실행 시작에 한 번만 걸리고 조건은 `--conditions` 순서대로 순차
빌드된다. **따라서 조건 목록이 다르면 같은 조건이라도 다른 난수를 받는다.**
예: qdrop은 `naive,adaround,qdrop,brecq,combined`에서 3번째, `naive,qdrop,brecq`에서
2번째다. 이 재추첨만으로 6-seed 분포만큼의 변동이 생긴다.

증거: **항상 1번째로 빌드되는 naive는 모든 run에서 bit-identical**인 반면 뒤 순서의
조건들은 그렇지 않다.

→ **두 run을 비교할 거면 `--conditions`를 반드시 동일하게 유지할 것.**

### 5.3 "6-seed 범위 밖"은 유의성 기준으로 너무 약하다

아무것도 바꾸지 않은 조건(qdrop의 lost, brecq의 Top1_flip)도 RNG 재추첨만으로 평균이
기존 6-seed 범위를 벗어났다. **변경하지 않은 조건들을 음성 대조군으로 두고 그 드리프트
대비 배수로 판정할 것.** (§4.1의 AdaRound 예산 판정이 그 방식이다 — 약 7배.)

---

## 6. run 인덱스

| run | 내용 | 결론 |
|---|---|---|
| `91_baseline_confirmed_6seed` | 이전 확정값 (AdaRound iters 1000) | `runs/97`로 대체됨 |
| `92_env_recheck` | `runs/91` seed0 재현 (환경 재구축 검증) | baseline 4개 bit-identical, Combined만 다름 |
| `93_iters_recheck` | BRECQ iters 20000, 확정 설정 | 2000과 노이즈 수준 차이 → 2000 유지 |
| `94_adaround_confound` | AdaRound iters/observer 1-seed 예비 | §7이 대체 (1-seed 결론은 부분적으로 틀렸다) |
| `95_official_defaults` | 2단계 + batch2 + neck layer-wise | 2단계만 유의미하게 악화 |
| `96_clipfix_verify` | CLIP 수정 결과 중립성 스모크 | 수정 전후 지표 동일, 크기 9.88 MiB |
| **`97_ada_iters2000_6seed`** | **확정 baseline** (AdaRound iters 2000, min-max) | **§1 표의 출처** |
| `98_ada_mseobs_6seed` | AdaRound observer를 MSE로 통일한 6-seed | ablation. COCO↔LVIS 트레이드오프 확인 |
| `99_combined_ref_det` ~ `111_screen` | Combined 설계 시도들(learn_alpha, range_blend, utility, region_dir, bit-width 스캔 등) | 전부 기각 — claim18~20 |
| **`112_confirmed_6seed`** | **확정 설계(stage1=brecq) W8A8 head 제외 6-seed** | LVIS 3지표 6/6 개선. claim20 |
| `113_w4a8` | W4A8 s_mult만, 3-seed | brecq에 0/3 악화 |
| `114_joint` | W4A8 `learn_alpha`+블록 재구성 공동 최적화 | 붕괴. claim21 |
| `115_headquant` | W8A8 head 포함 5조건, seed 0·1 | §1.1 |
| `116_headquant_probe` | head 포함 naive/adaround(min-max) | AdaRound min-max 붕괴 발견. **seed 0은 GPU 종류가 달라 폐기** |
| `117_ada_obs_isolate` | head 포함 AdaRound observer 단일변수 | 붕괴는 observer 아티팩트로 확정 |
| `118_alphabias_smoke` | `alpha_bias` 배관 스모크 | 통과 |
| `119_alphabias_w4a8_ABORTED_soft_confound` | `alpha_bias` 첫 시도 | soft 모드 교란(weight 3.77% 차이)으로 **폐기** |
| `120_alphabias_ste` | `alpha_bias` STE 버전 W4A8 3-seed | 8지표 0/3 붕괴. claim22 |
| `121_headquant_w8a8_rest` | head 포함 W8A8 seed 2~5(+ seed 0 AdaRound-MSE) | §1.1의 6-seed 완성. claim23 |
| `122_stage1_match_headquant` | head 포함 W8A8 Stage 1 격리 대조 6-seed | — |
| `123_lowbit_vm` | 09-29 E0(W4A4·W8A4, 첫/마지막 8bit) / E1 진단 / E2 W4A8 brecq vs brecq_vm(lam 1) 2-seed | A4 전부 붕괴(W8A4도). brecq_vm LVIS AP −0.3~−0.4점 2/2 |
| `124_vm_metric_ablation` | brecq_vm lam_mean=0 (2-seed), identity (seed0), W4A8 | 전부 LVIS AP 악화 — metric 무관, 공통 장치 탓 |
| `125_vm_w8a8` / `127_vm_w8a8_seed1` | brecq_vm lam0·identity, W8A8 head 포함 2-seed | identity LVIS AP +0.22/+0.24점. vocabulary metric ≤ identity |
| `126_vm_isolate` | brecq_vm anchor uniform / mix 0 / mix 1, W4A8 seed0 | 어느 변형도 LVIS AP 개선 없음 |
| `128_w4_sensitivity` | 학습 없는 W4 민감도 drop/restore (블록·conv 단위) | 12.cv2 하나가 LVIS_flip 손상의 60~83%. 원인은 attention 분기 소수 outlier 채널 |
| `129_mixed_precision` | BRECQ + 블록 W8 (12 / 1,2,4,12), W4A8 seed0 | 1,2,4,12(+7.8% 크기) LVIS AP +1.61점 |
| `130_migration` | BRECQ + scale 이전(conv별 독립 s, 상한) 2-seed | COCO +0.8~1.0, LVIS AP +0.1~0.3점 |
| `131_hicol` | BRECQ + 채널 W8(1%/3%) ± scale 이전 | 채널 1%+이전(독립) LVIS AP +1.1점 2/2 (+1.1% 크기) |
| `132_mig_tied` | 공유 제약 scale 이전(배포 가능) ± 채널 1%, 2-seed | 이전 단독 LVIS AP +0.2~0.4점. 채널 1% 결합 +0.7~0.8점, 채널 W8 위 이전의 LVIS 추가분 ≈ 0 |
| `133_pareto` | 채널 W8 0.5/1/2/5% + 같은 크기 conv 단위 W8 baseline | 채널 W8은 LVIS AP ~0.234에서 포화. 민감 conv 4개 W8(+1.5%) LVIS AP +1.05/+1.47점 2/2 -- 단 선택이 val2017·LVIS 진단(runs/128) 기반이라 평가 누수, 재선택 필요 |

이전 `runs/85`~`88`은 커밋 `9125d1d`(weight 양자화를 채널별 비대칭 MSE로 교체) **이전**
코드에서 측정된 것이라 현재 코드 판단의 근거로 쓸 수 없다.
