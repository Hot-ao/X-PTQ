# CVPR2027 — 저비트 PTQ(GPM) 실험 코드

**대상 독자:** 이 저장소를 넘겨받아 실험을 다시 돌리거나 코드를 고치는 사람. (2026-10-08 인수인계 기준. 10-28 갱신: ODinW 평가, 모델 저장, M 결정 캐시, PD-Quant, W4A4. **11-11 갱신:** 11-02 규칙, YOLO-World-L·Grounding DINO·segmentation, PD-Quant 재구현과 세 번째 기반, 결과 고정본 생성기, 호스트 메모리 주의.)

이 README는 "어떻게 설치하고, 어떻게 돌리고, 표를 어떻게 다시 만드는가"를 다룬다. 다른 내용은 아래 문서에 있다.
- **`documents/`(11-02~):** 논문 작업용 문서. 지금 수치와 논문 반영은 여기를 본다.
- **`docs/`(~11-04):** 원 저장소(promptcal-ptq)에서 옮겨 온 입문서·방법·경위. 방법의 뼈대는 맞지만 **수치와 규칙은 11-02 이전 것**이 섞여 있다. 수치는 `documents/RESULTS_FINAL.md`를 우선한다.

| 알고 싶은 것 | 문서 |
|---|---|
| **지금 결과 (현재 규칙, 3 seed, 논문 표 단위)** | [`documents/RESULTS_FINAL.md`](documents/RESULTS_FINAL.md) + seed별 원자료 [`documents/results_final.csv`](documents/results_final.csv). 스크립트 생성물(§5) |
| 실험 목록과 실험 절 배치 제안 | [`documents/EXPERIMENTS_2026-11-09.md`](documents/EXPERIMENTS_2026-11-09.md) |
| 초안 PDF에 고칠 곳 (위치별 수치·문장) | [`documents/DRAFT_PATCH_2026-11-10.md`](documents/DRAFT_PATCH_2026-11-10.md) |
| 논문 초안 | [`documents/X-PTQ_초안_2026-11-11.md`](documents/X-PTQ_초안_2026-11-11.md), `documents/X-PTQ(초안).pdf` |
| 결과 해석, 논문 구조 | [`documents/RESULTS_2026-11-04.md`](documents/RESULTS_2026-11-04.md), [`documents/PAPER_STRUCTURE_2026-11-05.md`](documents/PAPER_STRUCTURE_2026-11-05.md) |
| 방법을 처음부터 쉽게 (수식·코드 포함) | [`docs/PRIMER.md`](docs/PRIMER.md) |
| 방법의 정의와 근거 실험 | [`docs/METHOD.md`](docs/METHOD.md) (11-02 규칙 변경은 이 README §3) |
| 모델 구조 (C2fAttn, head 등) | [`docs/MODEL_ARCHITECTURE.md`](docs/MODEL_ARCHITECTURE.md) |
| Grounding DINO 이식에서 바꾼 것 | [`docs/GDINO_CHANGES.md`](docs/GDINO_CHANGES.md) |
| 왜 이렇게 됐는지 (시도와 기각 기록) | [`docs/HISTORY.md`](docs/HISTORY.md) (10-09까지) |
| 실험 현황 한눈에 | `results/LEADERBOARD.html`, `results/leaderboard.csv` (모든 규칙이 섞인 현황판, 논문 수치는 RESULTS_FINAL) |
| run 하나의 목적과 인자 | `runs/<run>/README.md` |
| 쓰지 않게 된 파일 | [`archive/README.md`](archive/README.md) |

---

## 0. 처음 받았을 때

**① 설치.** 결과를 낸 환경은 python 3.12.4, torch 2.10.0+cu128, ultralytics 8.4.121이다. ultralytics 버전은 꼭 맞춘다(모델 구조와 평가 코드가 버전에 묶여 있다).

```bash
cd [저장소 루트]
python -m venv .venv && .venv/bin/pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
```

**② 데이터.** 기본 경로는 원래 서버 기준이다. 다른 곳이면 인자로 바꾼다.

| 데이터 | 기본 경로 | 바꾸는 방법 |
|---|---|---|
| COCO 원본 (`train2017/`, `val2017/`, `annotations/`) | `/data/taeho/coco_datasets` | `--coco-root` |
| COCO ultralytics 형식 (val 평가용) | `/data/taeho/coco_ultra` | `configs/coco_local.yaml`의 `path`, 또는 `--data` |
| LVIS minival 주석 | `/data/taeho/lvis_datasets/.../lvis_v1_minival.json` | `--lvis-ann` |
| ODinW-13 (GLIP 공식 배포본, 약 4GB) | `/data/taeho/odinw` | `--odinw-root`. 받기: `scripts/data/download_odinw13.sh [폴더]` |

대기열로 돌릴 때는 `scripts/protocol.sh`의 `PROTOCOL` 끝에 위 인자를 덧붙이면 모든 작업에 적용된다.

**③ 가중치.** `yolov8s-world.pt`, `yolov8s-worldv2.pt`, `yolov8m-world.pt`, `yolov8l-world.pt`, `yoloe-v8s-seg.pt`, `yoloe-11s-seg.pt`는 저장소 루트에 둔다(없으면 ultralytics가 처음 실행 때 내려받는다). YOLOE는 텍스트 인코더 `mobileclip_blt.ts`(약 600MB)도 루트에 내려받는다. OWLv2-B/16, Grounding DINO-T는 HuggingFace 가중치를 쓴다(`run_owl.py`, `run_gdino.py`).

**④ 동작 확인.** 아래 한 줄이 10~15분쯤 걸려 끝나고 FP 수치가 맞으면 설치와 데이터가 정상이다.

```bash
source scripts/protocol.sh
.venv/bin/python run_comparison.py $PROTOCOL --model yolov8s-world.pt --w-bits 8 --a-bits 8 --conditions naive --seed 0 --device 0
```

| 모델 | FP COCO AP | FP LVIS AP |
|---|---|---|
| YOLOv8s-World | 36.80 | 25.89 |
| YOLOE-v8s | 31.29 | 25.94 |

나머지 모델의 FP LVIS AP: S-v2 22.19, M 29.80, L 33.43, YOLOE-11s 26.81, OWLv2-B/16 32.40, GDINO-T 30.06(LVIS 0–2000장).

**⑤ 기존 로그.** `runs/`는 git에 없다. 10-08·10-28 묶음(`runs/logs_handoff_2026-10-08.tar.gz`, `runs/logs_handoff_2026-10-28.tar.gz`)은 run 206까지만 들어 있다. **207~259(11월 run)의 로그 묶음은 아직 만들지 않았다.** 넘겨줄 때는 `runs/2*/`의 `*.log`, `queue.tsv`, `chain.log`, `README.md`를 묶는다(§5의 `export_final.py`가 그 로그만 읽는다).

**⑥ 저장된 양자화 모델.** 10-27부터 빌드한 모델을 `/data/taeho/cvpr2027_qmodels/`(11-11 기준 약 207GB)에 저장한다. **현재 규칙 모델은 `r1102/`**, segmentation은 `r1105seg/`·`r1105ywseg/`, 그 밖은 `fl16/`(첫·마지막 FP), `gdino*/`, `prg/`, `qual_l40s/`(11-03 정성 그림용, 이전 규칙)다. 폴더 바로 아래 파일은 10-27~11-01 빌드라 11-02 이전 규칙이다. `--load-model`로 빌드 없이 다른 평가를 돌릴 수 있다(§4). 같은 이름의 `.json`에 커밋·인자·COCO AP가 있다. 코드(특히 `quant/`)가 바뀌면 못 읽을 수 있으니 `.json`의 커밋을 확인한다.

---

## 1. 파일 구성

```
CVPR2027/
├── run_comparison.py   run_owl.py   run_gdino.py   진입점 (YOLO 계열 / OWLv2 / Grounding DINO). 모든 대기열이 이 경로로 부른다
├── core/               진입점이 쓰는 보조 모듈 (진입점이 sys.path에 넣는다)
├── quant/              양자화기와 G·P·M, BRECQ/QDrop
├── configs/            데이터 설정, 보호 진단 캐시(protect_cache/), M 결정 캐시(mcheck_cache/), ODinW GLIP config(odinw13_glip/)
├── scripts/            실행(작업자·프로토콜), 대기열(queues/), 표 생성기(tables/), 리더보드(leaderboard/), 데이터 받기(data/)
├── figures/            논문 그림(pdf/png)과 그 생성 코드(src/)
├── docs/               입문서, 방법, 경위 (~11-04, 수치는 옛 규칙 포함)
├── documents/          논문 작업 문서, 결과 고정본(RESULTS_FINAL.md, results_final.csv — 생성물), 초안
├── results/            리더보드(LEADERBOARD.html, leaderboard.csv) — 생성물
├── runs/               실험 로그(run마다 디렉터리). 표 생성기와 리더보드가 이름으로 읽으니 이름을 바꾸지 않는다
├── archive/            쓰지 않게 된 코드·run·캐시 (archive/README.md)
└── *.pt, mobileclip_blt.ts, weights/, .venv   promptcal-ptq로 가는 링크. 대기열이 `--model yolov8s-world.pt`처럼
                        루트 기준 이름으로 넘기므로 옮기지 않는다(옮기면 ultralytics가 같은 이름으로 새로 내려받는다)
```

| 파일 | 역할 |
|---|---|
| `run_comparison.py` | **진입점(YOLO 계열).** FP·양자화 모델을 조건별로 빌드하고 COCO AP, LVIS AP, 판정 충실도(flip/lost)를 잰다 |
| `run_owl.py` | **진입점(OWLv2).** 같은 프로토콜 규칙을 ViT에 옮긴 판(softmax 출력 16bit). `--diag`는 보호 진단 캐시만 만든다 |
| `run_gdino.py` | **진입점(Grounding DINO-T, 10-30).** 양자화 구성은 `core/gdino_quant.py`, 재구성은 `quant/brecq_units.py`(41 단위). LVIS는 어휘 chunk 32개라 느려서 `--lvis-start/--lvis-cap`으로 구간을 나누고 `--save-dets`로 합친다. 자세한 것은 파일 머리말과 `docs/GDINO_CHANGES.md` |
| `core/diag_w4_sensitivity.py` | ① 보호용 누수 없는 conv 진단(`rank_convs_leakfree`), 선택(`select_protected`), M 채택 사전 검사(`mcheck_flip`) |
| `core/harness.py` | 유사도/판정 측정 도우미 (`run_comparison`이 사용) |
| `core/odinw.py` | ODinW-13 평가(GLIP 공식 split·프롬프트, 데이터셋별 COCO AP와 13개 평균). 어노테이션 id를 1부터 다시 매긴다(§6) |
| `core/qmodel_io.py` | 양자화 모델 저장/불러오기. 인스턴스에 붙인 forward 메서드를 떼어 저장하고 불러올 때 다시 붙인다 |
| `scripts/data/download_odinw13.sh` | ODinW-13 공식 배포본 다운로드 + GLIP config 복사 |
| `core/model_zoo.py` | 모델 종류별 생성. `yolov8*-world*.pt`는 YOLO-World, `yoloe-*.pt`는 YOLOE(분할 가중치를 검출 구성에 불러옴, 텍스트 임베딩으로 어휘 전환, SAVPE 제외). 결정적 pooling(`DetAdaptiveMaxPool2d`) |
| `core/owl_zoo.py`, `core/owl_quant.py` | OWLv2를 블록 리스트로 펼친 래퍼(Linear → 1×1 conv), OWLv2 양자화 구성과 M |
| `scripts/protocol.sh` | **확정 프로토콜 공통 인자**(아래 §2). 작업자가 이 파일을 읽는다 |
| `scripts/worker.sh`, `scripts/worker_owl.sh` | GPU 하나를 맡는 대기열 작업자(`<run_dir>/queue.tsv`). 각각 `run_comparison.py` / `run_owl.py`를 부른다 |
| `scripts/run_mixed.sh` | GPU 하나에서 여러 대기열(YOLO·OWL 섞어서)을 차례로 처리 |
| `scripts/track.sh` | 대기열 밖에서 띄우는 작업을 `chain.log`에 남겨 리더보드 진행 현황에 보이게 한다 |
| `scripts/queues/*.tsv` | 표 1~5와 확장 실험을 만든 대기열 원본(§5) |
| `scripts/tables/*.py` | 로그에서 표 1~5를 다시 만드는 생성기(§5). `logparse.py`가 로그 파서 |
| `scripts/make_protect_cache.py` | 보호 진단 캐시를 미리 만든다(새 모델을 대기열로 돌리기 전에 1회) |
| `scripts/leaderboard/` | `make_leaderboard.py`(`runs/`의 모든 로그 → `results/`; `collect` → `build_results` → `best` / `pick_ours` / `main_cell`이 모든 표와 그림의 공통 집계 함수), **`export_final.py`(현재 규칙만 → `documents/RESULTS_FINAL.md`, `results_final.csv`, §5)**, 템플릿, 10분 자동 갱신 루프(`leaderboard_loop.sh`) |
| `scripts/convert_yw_seg.py` | YOLO-World-Seg mask 분기를 YOLO-World 가중치에 붙일 형식으로 변환(`--yw-seg`용) |
| `scripts/mig_rule_check.py` | M의 xmax 규칙 비교(10-30, 재구성·평가 없이 α와 scale 변화만 본다). 11-02 규칙의 근거 |
| `scripts/examples/post_build_flip_check.py` | `--post-build` 훅 예시 |
| `figures/src/diag_concentration.py`, `concentration_curve.py` | 민감도 지도(`figures/sensitivity_maps*.pdf`)와 손상 집중도 곡선 |
| `figures/src/bits_curve.py` | 비트별 LVIS AP 곡선(S, seed 0–2 평균, 막대 = seed 범위). 리더보드 집계 함수를 그대로 부른다 |
| `figures/src/mcheck_scatter.py` | M 사전 검사 정확도 산점도(`runs/234`, `236`, `240`, `247`의 73건) |
| `figures/src/teaser_flip.py`, `teaser_v2.py` | teaser 그림 |
| `figures/src/make_method_overview.py`, `make_method_arch.py`, `make_method_flow*.py` | 방법 그림(`method_*.pdf`, `method_flow.pptx`) |
| `figures/src/qual_dets.py` → `qual_figure.py`, `qual_grid.py` | 정성 그림. 1단계(GPU)가 검출을 저장하고 2단계(CPU)가 규칙으로 사례를 고른다. **11-03판은 이전 규칙 모델 기준이라 다시 만들어야 한다** |
| `docs/*.md` | 입문서, 방법, 경위 (위 표). `docs/BASELINE_STATUS.md`는 10-02 프로토콜 이전의 기준선 충실도 기록 |
| `quant/fake_quant.py` | `QuantConv2d`: weight 출력 채널별 MSE, activation per-tensor MSE observer, 이전(`mig`), 채널별 비트 |
| `quant/quant_model.py` | conv 래핑, calibrate, 비트 지정(`set_first_last_bits`, `set_last_abits`, `set_conv_wbits` 등) |
| `quant/adaround.py`, `quant/brecq.py` | ③ AdaRound/BRECQ/QDrop 재구성. 16bit 이상 activation은 LSQ 학습 제외(`LSQ_MAX_BITS`) |
| `quant/fusion_quant.py` | ⓪ 텍스트 게이트 교환(`gate_commute_all`) |
| `quant/channel_graph.py`, `quant/migrate.py` | ② 공유 제약 scale 이전(채널 계보 추적, 공유 그룹, α 탐색) |
| `quant/attn_quant.py` | attention·contrastive matmul 8bit 양자화(`--attn-quant`) |
| `quant/pdquant.py` | PD-Quant(CVPR'23) 기준선의 PD 손실·DC(`optimize_brecq(pd=..., dc_bn_map=...)`가 사용)와 head 캡처 도우미. 11-09 DC 버그 수정 뒤 공식 구현과 대조 통과(`runs/254`). 환경변수: `PDQ_PD_REDUCE`(§3), 진단용 `PDQ_NO_PD`, `PDQ_NO_DC`, `PDQ_DC_SKIP_FIRST` |
| `core/gdino_quant.py`, `quant/brecq_units.py` | Grounding DINO 양자화 구성, 단위별 재구성 |
| `quant/promptcal.py`, `quant/semantic_calib.py`, `quant/vocab_metric.py` | Combined(PromptCal) 기준선, 어휘 임베딩 도우미 |

`quant/`에는 확정 파이프라인이 실제로 부르는 코드만 남겼다(2026-10-08 정리). promptcal-ptq의 `src/quant/`는 그 정리 이전의 전체 구현으로, 그쪽 루트 `scripts/`의 옛 실험 스크립트가 쓴다(이 저장소에는 없다). 이제 서로 맞추지 않으니 파이프라인을 고칠 때는 이 저장소의 `quant/`만 고친다. 기각된 방향의 코드는 `archive/legacy/`에 있다. 읽지 않아도 되고, 지금의 `run_comparison.py`로는 돌지 않는다.

---

## 2. 확정 프로토콜 (2026-10-02)

`scripts/protocol.sh`:

```
--deterministic --calib 256 --no-skip-head --first-last-bits 8
--attn-quant attn_cls --last-abits 16
--combined-stage1 brecq --combined-recon-iters 2000
```

| 항목 | 설정 |
|---|---|
| 양자화 범위 | 모든 conv(head 포함). attention Linear/matmul과 head contrastive matmul도 8bit |
| 첫·마지막 레이어 | stem 첫 conv, head cv2/cv3 마지막 1×1은 W8A8 |
| head 마지막 conv 입력 | A16(`--last-abits 16`). m 모델의 seed 불안정 해결용으로, 모든 모델에 같게 적용한다 |
| FP로 남는 것 | LayerNorm, softmax, sigmoid, max, residual add, DFL, NMS |
| calibration | train2017 정렬 순서 앞 256장 |
| 보호 진단 | train2017 앞 200장, COCO 어휘. 캐시: `configs/protect_cache/<모델>_n200_c256_img640_la16_aqattn_cls.json` |
| 필수 환경 | `CUDA_DEVICE_ORDER=PCI_BUS_ID`. 한 비교의 모든 seed는 같은 GPU 종류(RTX 4000 Ada, GPU 0·4·5·6·7)에서 돌린다. L40S(1–3)는 정성 그림·점검용으로만 쓰고 표 수치에 섞지 않는다 |
| calibration 밖 flip | `--train-flip`: calibration에 안 쓴 train2017 200장, COCO 어휘에서 조건별 top-1 flip. Ours 구성 선택(§3)의 근거라 메인 표 run은 모두 켠다 |

### 11-02 규칙 (현재 규칙)

10-31~11-02에 P와 M의 세부 규칙을 고쳤다. 지금 기본값이 이 규칙이고, `RESULTS_FINAL.md`는 이 규칙 결과만 쓴다. 이전 규칙 로그는 리더보드에서 꼬리표(`P 기존`, `M 8장`, `ρ 1.5`, `하한 x%`)로 구분된다.

| 항목 | 이전 | 11-02 규칙 (인자, 기본값) |
|---|---|---|
| G와 함께 쓰는 P | C2fAttn cv2를 보호 목록에서 빼기만 함(예산이 남음) | 진단 행에서 cv2를 먼저 빼고 1.5% 예산을 다시 채운다. 분모는 양자화 weight 전체(`--protect-refill-g`, 켬) |
| M의 max\|x\| | α 탐색 이미지 8장 | calibration 전체(`--mig-stat all`) |
| M의 max\|x\| 하한 | 없음 | 소비 conv 최댓값의 0.3%(`--xmax-floor 0.003`) |
| M 사전 검사(`A`) 기준 ρ | 1.5배 | 1.0배(`run_comparison.py`의 `MCHECK_RATIO`). 1.1~1.4배 구간에서 M이 손해인 경우를 놓쳤다(YOLOE-v8s W4A8, S W4A8) |

---

## 3. 조건 이름

`<기반>+<접미사>`로 쓴다. 예: `brecq+GPM`. 기반은 `naive`, `brecq`, `qdrop`, `pdquant`, `adaround`, `combined` 중 하나다. SmoothQuant·AWQ는 접미사 `S`·`W`다(`brecq+S`, `brecq+W`; M 대신 쓰면 `brecq+GPS`·`GPW`, `scripts/queues/s_baselines_sw.tsv`).

**`pdquant`(10-28 구현, 11-09 재구현):** `qdrop`과 같은 설정에 PD-Quant의 PD 손실(KL, T = 4, λ_r = 0.1)과 DC(BN 입력 통계 보정, λ_c = 0.02, 500 iter)를 더한 것이다.
- **작업자를 띄우는 셸에서 `export PDQ_PD_REDUCE=mean`을 반드시 한다.** 코드 기본값은 `sum`(공식 구현 그대로)인데, 검출기는 이미지당 anchor가 약 8,400개라 합으로 두면 PD 손실이 재구성 손실의 약 570배가 된다. 메인 표는 anchor 평균(`mean`)으로 돌렸다(11-09 결정, `runs/255` README). 로그에 `[pdquant][진단 변형] PD mean` 줄이 있어야 맞다.
- 11-09 이전 PD-Quant 결과(`runs/206`, `207`)는 DC 버그가 있어 쓰지 않는다.
- DC 때문에 W8A8에서 FP보다 AP가 높은 칸이 있다(YOLOE-v8s 27.01 vs 25.94, `runs/259`에서 DC를 끄면 BRECQ와 같아짐).

| 접미사 | 의미 |
|---|---|
| `G` | ⓪ 텍스트 게이트 교환. 보호 목록에서 C2fAttn cv2를 뺀다 |
| `P` | ① 누수 없는 진단으로 고른 conv를 W8로 보호(예산 1.5%) |
| `M` | ② 공유 제약 scale 이전 |
| `A` | M을 사전 검사로 켤지 정한다. M 끔/켬 두 후보를 빌드하고 calibration 밖 COCO flip을 비교해, M 쪽이 ρ배(11-02부터 1.0)를 넘으면 끈다. 결정은 `configs/mcheck_cache/`에 **모델·비트·프로토콜별로** 저장되고, 다음부터는 저장된 결정대로 한 후보만 빌드한다(처음 한 번만 빌드 시간 약 2.2배). `--mcheck-refresh`면 다시 검사 |
| `R` / `H` | 같은 예산의 무작위 보호 / HAWQ식(출력 MSE) 보호 (대조군) |

**우리 방법 = G + P (+ M).** G는 구조(C2fAttn 유무), P는 진단 캐시가 정한다. M을 쓸지는 **(b) 규칙**으로 정한다.
- **(b) 규칙:** 모델·비트마다 **BRECQ seed 0**의 calibration 밖 flip(`--train-flip`)으로 GP / GPM 중 하나를 고르고, 그 선택을 모든 seed와 모든 기반(BRECQ, QDrop, PD-Quant)에 그대로 쓴다. 코드로는 `make_leaderboard.pick_ours`이고, 표의 "선택" 열이 그 결과다.
- 대기열에는 결정된 구성(`brecq+GPM`, `qdrop+GP`, `pdquant+GPM` 등)을 직접 적는다. `+A`(빌드 중 자동 검사)는 같은 생각을 한 run 안에서 하는 것이고, 메인 표에는 쓰지 않는다.
- YOLOE는 텍스트 게이트가 없어 G가 아무것도 하지 않는다. 표에는 P / PM으로 적는다.

11-11 기준 선택(`RESULTS_FINAL.md` §1 "선택" 열):

| 모델 | W8A8 | W4A8 | W4A6 | W4A5 | W4A4 |
|---|---|---|---|---|---|
| YOLO-World-S | GPM | GPM | GP | GPM | GPM |
| YOLO-World-S-v2, L | GPM | GPM | GPM | GPM | GPM |
| YOLO-World-M | GP | GP | GP | GP | GPM |
| YOLOE-v8s, 11s | PM | P | PM | PM | PM |

M의 W4A4처럼 비트마다 결정이 달라질 수 있어서 결정을 비트별로 둔다.

조건마다 빌드 직전에 RNG를 복원한다. 그래서 대부분의 조건은 한 run에 여러 조건을 묶어도, 따로 돌려도 결과가 bit 단위로 같다(10-08 확인: s의 `naive`, `brecq+M`, `combined`). **예외: `brecq+GPM`은 묶음에 따라 달라진다.** s W4A8 seed 0에서 `naive,brecq,brecq+PM,brecq+GPM`으로 묶으면 LVIS AP 23.47, `brecq+GPM,combined`나 단독으로 돌리면 23.71이 나온다(COCO AP는 같고 APr이 1.08점 다르다). 원인은 미상이다. 로그를 재현할 때는 대기열 줄의 조건 묶음을 그대로 쓴다.

---

## 4. 실행

```bash
cd [저장소 루트]          # 반드시 저장소 루트
source scripts/protocol.sh

# 한 번 실행
.venv/bin/python run_comparison.py $PROTOCOL --model yolov8s-world.pt \
    --w-bits 4 --a-bits 5 --conditions naive,brecq,brecq+PM,brecq+GPM --seed 0 --device 4

# 대기열: runs/<이름>/queue.tsv에 "태그<TAB>인자"를 한 줄씩 쓰고 GPU마다 작업자를 띄운다
scripts/worker.sh runs/<이름> 4 0-23 &
scripts/worker.sh runs/<이름> 5 24-47 &

# 메인 표 한 칸의 실제 형태(11-09~, runs/255·256): 기반과 기반+Ours를 묶고 calibration 밖 flip과 모델 저장을 켠다
export PDQ_PD_REDUCE=mean     # pdquant 조건이 있을 때. 작업자를 띄우기 전에
.venv/bin/python run_comparison.py $PROTOCOL --model yolov8l-world.pt --w-bits 4 --a-bits 8 \
    --conditions pdquant,pdquant+GPM --seed 1 --train-flip --save-model /data/taeho/cvpr2027_qmodels/r1102 --device 0
```

- **대기열 다루기:** 작업자가 돌고 있는 동안 `queue.tsv`를 고칠 때는 `flock <run_dir>/queue.lock`으로 잠근다(작업자도 같은 잠금을 쓴다). 맨 앞에 넣으면 다음에 비는 GPU가 먼저 가져간다. 진행은 `chain.log`(start/done), 실패는 `failed.log`(rc)에 남는다.

- **ODinW-13 평가:** `--odinw`면 COCO·LVIS 뒤에 ODinW-13도 잰다. `--odinw-only`면 LVIS를 건너뛰고 COCO AP(같은 모델인지 확인용)와 ODinW만 잰다. 결과는 `[ODinW-AP][<조건>] mean=... | <데이터셋>=...` 줄로 남는다. OWLv2(`run_owl.py`)도 `--odinw`를 받는다.
- **모델 저장/불러오기:** `--save-model <폴더>`는 빌드한 조건 모델을 `<모델>_w<W>a<A>_s<seed>_<조건>.pt`로 저장한다. `--load-model <폴더>`는 빌드 대신 같은 이름의 파일을 불러온다(같은 인자로 실행해야 이름이 맞는다). 첫·마지막 비트처럼 이름에 없는 인자를 바꿀 때는 폴더를 따로 쓴다(예: `.../fl16/`).
- **진단 훅:** `--post-build <스크립트>`를 주면 빌드 직후, 평가 전에 그 스크립트를 실행하고 끝난다. 스크립트에서 `models`, `fp`, `args`, `device`를 쓸 수 있다. 예: `scripts/examples/post_build_flip_check.py`.
- **segmentation:** `--yoloe-seg`는 YOLOE-seg를 mask 분기까지 양자화하고 LVIS mask AP를 잰다. `--yw-seg <경로>`는 YOLO-World에 YOLO-World-Seg mask 분기(`scripts/convert_yw_seg.py` 출력)를 붙인다.
- **calibration 데이터 교체:** `--calib-source`(coco / o365 / o365gqa, 10-28). 보호 진단 캐시가 소스별로 따로 생긴다(`_cso365` 등).
- **GPU 메모리:** m·L 모델은 한 run에 조건 2개까지만 묶는다. 20GB GPU에서 조건 3개는 메모리 한계에 걸린다.
- **호스트 RAM(서버 125GB, 연구실 공용):** 프로세스 하나가 양자화 중에는 약 13–14GB, **LVIS 평가 중에는 L이 20–27GB**까지 쓴다. L 평가가 여러 개 겹치면 OOM으로 죽는다(rc=137, `runs/253`, `256`에서 각 1–2건). L 작업은 대기열에서 서로 떨어뜨려 두고, 죽으면 저장 모델로 평가만 다시 돌린다(`--load-model`, 같은 인자). 호스트 RAM은 run마다 `mem.log`로 남긴다. 누수는 아니다(작업이 끝나면 돌아온다).
- **실행 위치:** 반드시 저장소 루트에서 실행한다. 하위 디렉터리(예: `scripts/`) 안에서 실행하면 `configs/`·가중치의 상대 경로가 어긋나고, ultralytics가 CLIP 가중치(338MB)를 그 디렉터리의 `weights/`에 다시 내려받는다.
- **리더보드:** `.venv/bin/python scripts/leaderboard/make_leaderboard.py`가 `runs/`(와 promptcal-ptq의 `runs/`)의 모든 로그로 `results/LEADERBOARD.html`과 `results/leaderboard.csv`를 다시 만든다. 10분마다 자동으로 만들려면 `nohup scripts/leaderboard/leaderboard_loop.sh 600 > /dev/null 2>&1 &`. 실험이 끝날 때마다 갱신하려면 `export PTQ_LEADERBOARD=1`(잠금 + 원자적 교체, 실패해도 실험에는 영향 없음).
- **모델 크기:** 로그의 "이론적 크기"는 양자화된 conv와 attention Linear weight의 합이다. 10-03 14:25 이전에 시작한 실행(`runs/158_final`, `159_gpm_all`, `154_protocol` 등)의 로그는 conv만이라 s는 0.44, m은 0.49 MiB 작게 찍혀 있다. 리더보드는 이 차이를 더해서 맞추고(`make_leaderboard.py`의 `fix_size`), promptcal-ptq의 `figures/make_pareto.py`는 로그 대신 상수를 쓴다.
- **소요 시간 (RTX 4000 Ada):** s 조건 하나 빌드에 약 20분, m은 약 23분이다. COCO와 LVIS 평가는 조건당 약 5분이다. PD-Quant는 PD 손실과 DC 때문에 더 걸려서, 조건 2개 묶은 작업 하나가 S·v2·YOLOE 약 1–3시간, m 약 4–5시간, L 약 6–7시간이다(`runs/256` `chain.log`, 11-11).

---

## 5. 표 재현

### 지금 쓰는 방법: 결과 고정본 (11-06~)

```bash
nice .venv/bin/python scripts/leaderboard/export_final.py     # 로그 1,000여 개를 읽어 1–2분
```

- `runs/`의 모든 로그를 리더보드와 같은 함수로 모아, **11-02 규칙 결과만** 논문 표 단위로 `documents/RESULTS_FINAL.md`에 쓰고 seed별 원자료와 칸별 출처(log, 저장 모델)를 `documents/results_final.csv`에 쓴다. md는 손으로 고치지 않는다.
- 머리에 생성 시각과 코드 커밋, 커밋 안 된 파일이 찍힌다. §7에 표마다 읽은 run 목록이 있다.
- 담는 것: 메인 표(6개 YOLO 모델 × W8A8/W4A8/W4A5/W4A4 × BRECQ / QDrop / PD-Quant ± Ours), 표 1(S, 모든 비교 방법 × W8A8–W4A4), OWLv2·GDINO, ablation, 그 밖의 현재 규칙 결과, 이전 규칙만 있는 묶음 목록.
- 그림(`figures/src/bits_curve.py`, `mcheck_scatter.py`)도 같은 집계 함수를 불러 같은 수치를 쓴다.

### 10월 표 생성기 (옛 규칙, 기록용)

아래는 10-08~10-28의 표 1~6을 만든 경로다. **11-02 이전 규칙 결과라 논문 수치로 쓰지 않는다.** 순서는 **대기열 복사 → 작업자 실행 → 표 생성기**다. 표 생성기는 아래 run 디렉터리 이름을 그대로 읽으므로 이름을 바꾸지 않는다.

| 표 | 내용 | 대기열 (`scripts/queues/`) → run 디렉터리 | 생성기 (`scripts/tables/`) |
|---|---|---|---|
| 1 | YOLOv8s-World 주 결과 (W8A8~W4A5, 6 seed) | `s_tables123.tsv` → `runs/158_final`<br>`s_seed01_w4.tsv` → `runs/154_protocol`<br>`s_gpm_w8a8_qdrop.tsv` → `runs/159_gpm_all` | `table1_main.py` |
| 2·3 | 구성요소 분해(G/P/M), 보호 기준 대조(P/H/R), QDrop 결합 | 표 1과 같은 run | `table23_ablation_plugin.py` |
| 4 | 일반화: YOLOv8s-WorldV2, YOLOv8m-World (6 seed) | `gen_v2_m.tsv` → seed 0·1의 28줄은 `runs/154_protocol`, 나머지 42줄은 `runs/160_gen6` | `table4_generalization.py` |
| 4 보충 | V2 W8A8에서 M을 뺀 GP | `v2_w8a8_gp.tsv` → `runs/162_v2w8` | (리더보드에서 확인) |
| 5 | YOLOE-v8s (W8A8/W4A8/W4A5, 6 seed) | `gen_yoloe.tsv` → `runs/164_yoloe` | `table5_yoloe.py` |
| 6 | ODinW-13 (YOLO-World-S, YOLOE-v8s/11s, W4A8·W4A5, 6 seed) | `odinw_yw.tsv` → `runs/201_odinw_yw`(W4A8 s2는 `runs/200_odinw_check`), `odinw_yoloe.tsv` → `runs/202_odinw_yoloe` | (RESULTS §4c, 생성기 없음) |
| W4A4 | s·m W4A4, s ablation (2 seed) | `w4a4.tsv` → `runs/203_w4a4`, `w4a4_m_abl.tsv` → `runs/205_w4a4_m_abl` | (RESULTS §4d) |
| PD-Quant | s W4A8·W4A5 (DC 버그 있던 판, 폐기) | `pdquant.tsv` → `runs/206_pdquant` | – |
| 그림 | 크기 대 정확도 | 표 1과 같은 run | promptcal-ptq의 `figures/make_pareto.py` (이 저장소에는 없다) |

```bash
mkdir -p runs/164_yoloe && cp scripts/queues/gen_yoloe.tsv runs/164_yoloe/queue.tsv
.venv/bin/python scripts/make_protect_cache.py yoloe-v8s-seg.pt      # 캐시가 이미 있으면 바로 끝난다
scripts/worker.sh runs/164_yoloe 4 0-23 &                            # GPU마다 하나씩
.venv/bin/python scripts/tables/table5_yoloe.py                      # 다 끝난 뒤
```

- 전체 180줄은 RTX 4000 Ada 5장(일부 구간은 7장)으로 합계 약 44시간 걸렸다. 가장 큰 `runs/158_final`(58줄)이 약 18시간이다.
- 다시 돌리려면 `<run_dir>/taken.log`에서 해당 태그를 지운다. 실패한 작업은 `failed.log`에 남는다.
- 같은 GPU 종류에서, 대기열 줄의 조건 묶음을 그대로 돌리면 로그의 수치가 bit 단위로 재현된다(묶음을 바꾸면 `brecq+GPM`이 달라질 수 있다, §3). GPU 종류가 다르면 소수 둘째 자리에서 달라질 수 있으므로, 한 비교의 seed는 모두 같은 GPU 종류에서 돌린다.
- 통계: 모든 차이는 같은 seed끼리의 짝 차이 평균 ± 95% 신뢰구간이고, 표마다 Holm 보정을 한다. s의 W4A5 seed 0은 기준선(BRECQ)이 붕괴하는 seed라 표에서 따로 보고한다.

---

## 6. 주의 (과거 사고)

- `--device N`과 nvidia-smi 번호는 `CUDA_DEVICE_ORDER=PCI_BUS_ID`일 때만 같다(`run_comparison.py`가 기본값으로 넣는다).
- 16bit activation에 LSQ를 걸면 step이 발산한다. `LSQ_MAX_BITS`가 막고 있으니 지우지 말 것(runs/155).
- 보호 진단 캐시는 프로토콜(`_la16_aqattn_cls`)별로 따로 저장된다. 프로토콜을 바꾸면 새 캐시가 생긴다. 네 모델의 캐시는 git에 들어 있다.
- YOLOE의 LVIS 평가는 `agnostic_nms=False`여야 한다. ultralytics의 `YOLOE.predict`가 클래스 무관 NMS를 강제해서, 그대로 두면 FP LVIS AP가 25.94가 아니라 11.0으로 나온다(`predict_lvis_results`에서 처리).
- YOLOE의 텍스트 프롬프트는 `a/b` 동의어 표기에서 첫 이름만 쓴다(`model_zoo.prompt_names`). YOLO-World는 전체 문자열을 쓴다.
- 10-02 이전 로그(`runs/154_protocol/old_lsqbug/`, `runs/153` 이하)는 프로토콜이 달라 지금 표와 섞으면 안 된다.
- ODinW 어노테이션 id는 0부터 시작하는데, pycocotools는 id 0인 GT를 "매칭 안 됨"으로 처리한다. GT를 그대로 예측으로 넣어도 AP가 100이 아니게 된다. `core/odinw.py`가 1부터 다시 매긴다.
- 저장 모델을 다른 GPU로 불러올 때는 `map_location`을 그 GPU로 줘야 한다(일반 텐서 속성은 `.to()`로 안 옮겨진다). `load_qmodel`이 처리한다.
- 같은 칸을 두 run에서 돌린 경우(예: S W4A6 BRECQ, `runs/234`와 `247`)가 있다. 집계 코드는 칸마다 하나만 고르지만, 직접 로그를 모아 그림을 그릴 때는 (모델, 비트, seed, 기반)마다 가장 늦은 로그만 쓴다(`mcheck_scatter.py`).
- PD-Quant 작업자를 `PDQ_PD_REDUCE` 없이 띄우면 조용히 `sum`으로 돈다. 로그의 `PD mean` 줄로 확인한다.
- 공식 PD-Quant 저장소는 대조용 참고로만 읽었다. 그 코드를 실행하지 않는다. 내려받은 체크포인트는 `torch.load(..., weights_only=True)`로만 연다.
- 앞 작업이 끝나기를 기다리는 스크립트에서 `pgrep -f "<경로>"`를 쓰면, 그 스크립트를 띄운 셸 명령에도 같은 문자열이 들어 있어 영원히 기다린다(10-27 사고). PID로 기다린다(`while kill -0 <pid>`).

---

## 7. 알려진 한계와 남은 일

11-11 기준. 실험 쪽 남은 일의 전체 목록은 `documents/EXPERIMENTS_2026-11-09.md`, 초안 반영은 `documents/DRAFT_PATCH_2026-11-10.md` §10에 있다.

| 항목 | 상태 |
|---|---|
| PD-Quant seed 1·2 (메인 표 W8A8/W4A8/W4A4) | `runs/256`에서 진행 중, 11-12 오전 끝 예상. W4A5·W4A6은 seed 0만(seed 1·2 추가 여부 미정) |
| ablation 빈칸 (S W4A8 PM, W4A5 M·GM·PM의 seed 1·2) | `runs/256` 대기열 맨 뒤 |
| 정성 그림 | 11-03판은 이전 규칙 모델. 현재 규칙(`r1102`) 모델로 검출을 다시 만들어야 한다 |
| GDINO LVIS flip | 정의가 다른 모델과 달라 고정본에서 비움. W4A5·W4A8 seed 0만 |
| OWLv2 W4A8 | QDrop(25.45)이 BRECQ + Ours(21.12)보다 높다. 같은 기반끼리만 비교한다 |
| s W4A5 seed 0에서 BRECQ·QDrop 기준선이 붕괴(COCO 19.3 / 9.8) | 원인 미상. 12.cv2 재구성 실패 가설은 기각됨 |
| Combined 1단계와 BRECQ 기준선의 재구성 단위가 다름 | Combined의 1단계 BRECQ는 neck을 conv 단위로, `brecq` 조건은 전부 block 단위로 재구성한다(`run_comparison.py`의 `build`). 표 1의 Combined−BRECQ 차이에 이 차이가 섞여 있다. 단위를 맞춘 재측정은 미실행 |
| `brecq+GPM`이 조건 묶음에 따라 달라짐 | s W4A8 seed 0에서 LVIS AP 23.47(대기열 묶음) vs 23.71(단독). 원인 미상, G가 들어간 BRECQ 재구성에서 갈라짐(`docs/HISTORY.md` §14) |
| V2 W8A8에서 M이 flip을 늘림(+2.05%p) | COCO 어휘 검사로도, 일반 어휘 검사로도 사전에 못 잡는다. GP는 flip을 줄인다(−0.65%p) |
| m에서 M이 해로운 이유 | 가설만 있음(m은 앞쪽 backbone 보호가 없음). 검증 실험 미실행 |
| W8A8 이득 | 섞여 있다. S·YOLOE는 작은 이득(+0.04 ~ +0.4), v2·L은 세 기반 모두 −0.0 ~ −0.3, M은 BRECQ·QDrop −0.0 ~ −0.1 / PD-Quant +0.1. 본문 주장은 W4 설정으로 한정한다 |
| 기준선 | RTN, AdaRound, BRECQ, QDrop, PD-Quant, BRECQ + SmoothQuant / AWQ. InlierQ(ICLR'26)·Reg-PTQ(CVPR'24)는 공식 코드가 없어 관련 연구로만 다룬다 |
| 빠진 측정 | 실제 지연 시간(지금은 fake-quant 시뮬레이션과 이론 크기만) |
| 모델 범위 | YOLO-World S/S-v2/M/L, YOLOE-v8s/11s(메인), OWLv2-B/16·Grounding DINO-T(확장), YOLOE-seg·YOLO-World-Seg-M(segmentation, W4A8만 3 seed) |
| W4A4 | 6개 모델 모두 동작(메인 표, 3 seed) |
