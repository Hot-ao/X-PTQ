# CVPR2027 — 저비트 PTQ(GPM) 실험 코드

**대상 독자:** 이 저장소를 넘겨받아 실험을 다시 돌리거나 코드를 고치는 사람. (2026-10-08 인수인계 기준, 2026-10-28 갱신: ODinW 평가, 모델 저장, M 결정 캐시, PD-Quant, W4A4. 바뀐 점만 보려면 [`docs/UPDATE_2026-10-28.md`](docs/UPDATE_2026-10-28.md))

이 README는 "어떻게 설치하고, 어떻게 돌리고, 표를 어떻게 다시 만드는가"를 다룬다. 다른 내용은 아래 문서에 있다. `docs/`의 다섯 문서는 원 저장소(promptcal-ptq)의 루트와 `docs/`에 있던 원본을 이 저장소 기준으로 옮겨 갱신한 판이고(2026-10-09), 앞으로는 이쪽을 고친다.

| 알고 싶은 것 | 문서 |
|---|---|
| 방법을 처음부터 쉽게 (수식·코드 포함) | [`docs/PRIMER.md`](docs/PRIMER.md) |
| 방법의 정의와 근거 실험 | [`docs/METHOD.md`](docs/METHOD.md) |
| 확정 결과 (표 1~6, W4A4) | [`docs/RESULTS.md`](docs/RESULTS.md) |
| 10-27~28 변경 사항 (코드·데이터·실험) | [`docs/UPDATE_2026-10-28.md`](docs/UPDATE_2026-10-28.md) |
| 왜 이렇게 됐는지 (시도와 기각 기록) | [`docs/HISTORY.md`](docs/HISTORY.md) |
| 논문 구성과 남은 일 | [`docs/PAPER_OUTLINE.md`](docs/PAPER_OUTLINE.md)(v1, 방법 중심), [`docs/PAPER_OUTLINE_v2_diagnosis.md`](docs/PAPER_OUTLINE_v2_diagnosis.md)(v2, 진단 중심) |
| 논문 관련 문서 (시나리오, 현황 정리, 변경 경위; **10-31부터 새 논문 문서는 모두 여기에 둔다**) | [`documents/`](documents/) — [`STATUS_2026-10-31.md`](documents/STATUS_2026-10-31.md), [`METHOD_SCENARIO.md`](documents/METHOD_SCENARIO.md), [`GDINO_CHANGES.md`](documents/GDINO_CHANGES.md) |
| 실험 현황 한눈에 | `results/LEADERBOARD.html`, `results/leaderboard.csv` |
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

**③ 가중치.** `yolov8s-world.pt`, `yolov8s-worldv2.pt`, `yolov8m-world.pt`, `yoloe-v8s-seg.pt`, `yoloe-11s-seg.pt`는 저장소 루트에 둔다(없으면 ultralytics가 처음 실행 때 내려받는다). YOLOE는 텍스트 인코더 `mobileclip_blt.ts`(약 600MB)도 루트에 내려받는다.

**④ 동작 확인.** 아래 한 줄이 10~15분쯤 걸려 끝나고 FP 수치가 맞으면 설치와 데이터가 정상이다.

```bash
source scripts/protocol.sh
.venv/bin/python run_comparison.py $PROTOCOL --model yolov8s-world.pt --w-bits 8 --a-bits 8 --conditions naive --seed 0 --device 0
```

| 모델 | FP COCO AP | FP LVIS AP |
|---|---|---|
| YOLOv8s-World | 36.80 | 25.89 |
| YOLOE-v8s | 31.29 | 25.94 |

**⑤ 기존 로그.** `runs/`는 git에 없다(전체 410GB). 표를 만드는 데 필요한 로그와 대기열만 묶은 `runs/logs_handoff_2026-10-08.tar.gz`(3.4MB)를 따로 받아 `runs/` 안에서 푼다. 그러면 §5의 표 생성기가 바로 돈다. 10-27~28의 run(200~206: ODinW, W4A4, PD-Quant)은 `runs/logs_handoff_2026-10-28.tar.gz`(0.7MB)에 따로 묶었다.

**⑥ 저장된 양자화 모델.** 10-27부터 빌드한 모델을 `/data/taeho/cvpr2027_qmodels/`(약 22GB)에 저장한다. `--load-model`로 빌드 없이 다른 평가를 돌릴 수 있다(§4). 같은 이름의 `.json`에 커밋·인자·COCO AP가 있다. 코드(특히 `quant/`)가 바뀌면 못 읽을 수 있으니 `.json`의 커밋을 확인한다.

---

## 1. 파일 구성

```
CVPR2027/
├── run_comparison.py   run_owl.py      진입점 (YOLO 계열 / OWLv2). 모든 대기열이 이 경로로 부른다
├── core/               진입점이 쓰는 보조 모듈 (진입점이 sys.path에 넣는다)
├── quant/              양자화기와 G·P·M, BRECQ/QDrop
├── configs/            데이터 설정, 보호 진단 캐시(protect_cache/), M 결정 캐시(mcheck_cache/), ODinW GLIP config(odinw13_glip/)
├── scripts/            실행(작업자·프로토콜), 대기열(queues/), 표 생성기(tables/), 리더보드(leaderboard/), 데이터 받기(data/)
├── figures/            논문 그림(pdf/png)과 그 생성 코드(src/)
├── docs/               입문서, 방법, 결과, 경위, 논문 개요
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
| `scripts/leaderboard/` | `make_leaderboard.py`(`runs/`의 모든 로그 → `results/`), 템플릿, 10분 자동 갱신 루프(`leaderboard_loop.sh`) |
| `scripts/examples/post_build_flip_check.py` | `--post-build` 훅 예시 |
| `figures/src/diag_concentration.py` | 여섯 모델 민감도 지도(`figures/sensitivity_maps.pdf`)와 진단 집중도 표 |
| `figures/src/make_method_overview.py` | 방법 개요 그림(`figures/method_overview.pdf`) |
| `docs/*.md` | 입문서, 방법, 결과, 경위, 논문 개요 (위 표). `docs/BASELINE_STATUS.md`는 10-02 프로토콜 이전의 기준선 충실도 기록 |
| `quant/fake_quant.py` | `QuantConv2d`: weight 출력 채널별 MSE, activation per-tensor MSE observer, 이전(`mig`), 채널별 비트 |
| `quant/quant_model.py` | conv 래핑, calibrate, 비트 지정(`set_first_last_bits`, `set_last_abits`, `set_conv_wbits` 등) |
| `quant/adaround.py`, `quant/brecq.py` | ③ AdaRound/BRECQ/QDrop 재구성. 16bit 이상 activation은 LSQ 학습 제외(`LSQ_MAX_BITS`) |
| `quant/fusion_quant.py` | ⓪ 텍스트 게이트 교환(`gate_commute_all`) |
| `quant/channel_graph.py`, `quant/migrate.py` | ② 공유 제약 scale 이전(채널 계보 추적, 공유 그룹, α 탐색) |
| `quant/attn_quant.py` | attention·contrastive matmul 8bit 양자화(`--attn-quant`) |
| `quant/pdquant.py` | PD-Quant(CVPR'23) 기준선의 PD 손실·DC(`optimize_brecq(pd=..., dc_bn_map=...)`가 사용)와 head 캡처 도우미. 검출기에 옮기며 정한 것은 파일 머리말 |
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
| 필수 환경 | `CUDA_DEVICE_ORDER=PCI_BUS_ID`. 한 비교의 모든 seed는 같은 GPU 종류(RTX 4000 Ada)에서 돌린다 |

---

## 3. 조건 이름

`<기반>+<접미사>`로 쓴다. 예: `brecq+GPM`. 기반은 `naive`, `brecq`, `qdrop`, `pdquant`, `adaround`, `combined` 중 하나다. `pdquant`(10-28)는 `qdrop`과 같은 설정에 PD-Quant의 PD 손실과 DC만 더한 것이다.

| 접미사 | 의미 |
|---|---|
| `G` | ⓪ 텍스트 게이트 교환. 보호 목록에서 C2fAttn cv2를 뺀다 |
| `P` | ① 누수 없는 진단으로 고른 conv를 W8로 보호(예산 1.5%) |
| `M` | ② 공유 제약 scale 이전 |
| `A` | M을 사전 검사로 켤지 정한다. M 끔/켬 두 후보를 빌드하고 calibration 밖 COCO flip을 비교해, M 쪽이 1.5배를 넘으면 끈다. 결정은 `configs/mcheck_cache/`에 **모델·비트·프로토콜별로** 저장되고, 다음부터는 저장된 결정대로 한 후보만 빌드한다(처음 한 번만 빌드 시간 약 2.2배). `--mcheck-refresh`면 다시 검사 |
| `R` / `H` | 같은 예산의 무작위 보호 / HAWQ식(출력 MSE) 보호 (대조군) |

**우리 방법 = `brecq+GPMA`**(= 규칙대로 M을 정한 GPM)를 모든 모델에 같은 명령으로 쓴다. G는 구조(C2fAttn 유무), P는 진단 캐시, M은 결정 캐시가 자동으로 정한다. 표 1~5의 대기열은 결정된 구성(`brecq+GPM` 또는 `brecq+GP`)을 직접 적었다. PM(`brecq+PM`)은 변형으로 함께 잰다.

| 모델 | 실제 구성 | M 결정 근거 |
|---|---|---|
| YOLOv8s-World | `brecq+GPM` (W8A8에서는 P가 효과 없어 G+M) | 검사 통과(M 켬) |
| YOLOv8s-WorldV2 | `brecq+GPM` | 검사 통과(M 켬) |
| YOLOv8m-World | `brecq+GP`. 보호 대상이 C2fAttn cv2뿐이라 `brecq+G`와 같다 | 검사 탈락(M을 켜면 flip 2~7배) |
| YOLOE-v8s (`yoloe-v8s-seg.pt`) | `brecq+GPM`. 텍스트 게이트가 없어 G가 아무것도 하지 않으므로 실제로는 PM | 검사 통과(M 켬) |
| YOLOE-11s (`yoloe-11s-seg.pt`) | `brecq+GPM` (실제로는 PM) | 검사 통과(M 켬) |
| YOLOv8m-World **W4A4** | `brecq+GPM` | W4A4 검사에서는 M 켬(W4A5와 반대). 그래서 결정을 비트별로 저장한다 |

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
```

- **ODinW-13 평가:** `--odinw`면 COCO·LVIS 뒤에 ODinW-13도 잰다. `--odinw-only`면 LVIS를 건너뛰고 COCO AP(같은 모델인지 확인용)와 ODinW만 잰다. 결과는 `[ODinW-AP][<조건>] mean=... | <데이터셋>=...` 줄로 남는다. OWLv2(`run_owl.py`)도 `--odinw`를 받는다.
- **모델 저장/불러오기:** `--save-model <폴더>`는 빌드한 조건 모델을 `<모델>_w<W>a<A>_s<seed>_<조건>.pt`로 저장한다. `--load-model <폴더>`는 빌드 대신 같은 이름의 파일을 불러온다(같은 인자로 실행해야 이름이 맞는다). 첫·마지막 비트처럼 이름에 없는 인자를 바꿀 때는 폴더를 따로 쓴다(예: `.../fl16/`).
- **진단 훅:** `--post-build <스크립트>`를 주면 빌드 직후, 평가 전에 그 스크립트를 실행하고 끝난다. 스크립트에서 `models`, `fp`, `args`, `device`를 쓸 수 있다. 예: `scripts/examples/post_build_flip_check.py`.
- **메모리:** m 모델은 한 run에 조건 2개까지만 묶는다. 20GB GPU에서 조건 3개는 메모리 한계에 걸린다.
- **실행 위치:** 반드시 저장소 루트에서 실행한다. 하위 디렉터리(예: `scripts/`) 안에서 실행하면 `configs/`·가중치의 상대 경로가 어긋나고, ultralytics가 CLIP 가중치(338MB)를 그 디렉터리의 `weights/`에 다시 내려받는다.
- **리더보드:** `.venv/bin/python scripts/leaderboard/make_leaderboard.py`가 `runs/`(와 promptcal-ptq의 `runs/`)의 모든 로그로 `results/LEADERBOARD.html`과 `results/leaderboard.csv`를 다시 만든다. 10분마다 자동으로 만들려면 `nohup scripts/leaderboard/leaderboard_loop.sh 600 > /dev/null 2>&1 &`. 실험이 끝날 때마다 갱신하려면 `export PTQ_LEADERBOARD=1`(잠금 + 원자적 교체, 실패해도 실험에는 영향 없음).
- **모델 크기:** 로그의 "이론적 크기"는 양자화된 conv와 attention Linear weight의 합이다. 10-03 14:25 이전에 시작한 실행(`runs/158_final`, `159_gpm_all`, `154_protocol` 등)의 로그는 conv만이라 s는 0.44, m은 0.49 MiB 작게 찍혀 있다. 리더보드는 이 차이를 더해서 맞추고(`make_leaderboard.py`의 `fix_size`), promptcal-ptq의 `figures/make_pareto.py`는 로그 대신 상수를 쓴다.
- **소요 시간 (RTX 4000 Ada):** s 조건 하나 빌드에 약 20분, m은 약 23분이다. COCO와 LVIS 평가는 조건당 약 5분이다.

---

## 5. 표 재현

순서는 **대기열 복사 → 작업자 실행 → 표 생성기**다. 표 생성기는 아래 run 디렉터리 이름을 그대로 읽으므로 이름을 바꾸지 않는다.

| 표 | 내용 | 대기열 (`scripts/queues/`) → run 디렉터리 | 생성기 (`scripts/tables/`) |
|---|---|---|---|
| 1 | YOLOv8s-World 주 결과 (W8A8~W4A5, 6 seed) | `s_tables123.tsv` → `runs/158_final`<br>`s_seed01_w4.tsv` → `runs/154_protocol`<br>`s_gpm_w8a8_qdrop.tsv` → `runs/159_gpm_all` | `table1_main.py` |
| 2·3 | 구성요소 분해(G/P/M), 보호 기준 대조(P/H/R), QDrop 결합 | 표 1과 같은 run | `table23_ablation_plugin.py` |
| 4 | 일반화: YOLOv8s-WorldV2, YOLOv8m-World (6 seed) | `gen_v2_m.tsv` → seed 0·1의 28줄은 `runs/154_protocol`, 나머지 42줄은 `runs/160_gen6` | `table4_generalization.py` |
| 4 보충 | V2 W8A8에서 M을 뺀 GP | `v2_w8a8_gp.tsv` → `runs/162_v2w8` | (리더보드에서 확인) |
| 5 | YOLOE-v8s (W8A8/W4A8/W4A5, 6 seed) | `gen_yoloe.tsv` → `runs/164_yoloe` | `table5_yoloe.py` |
| 6 | ODinW-13 (YOLO-World-S, YOLOE-v8s/11s, W4A8·W4A5, 6 seed) | `odinw_yw.tsv` → `runs/201_odinw_yw`(W4A8 s2는 `runs/200_odinw_check`), `odinw_yoloe.tsv` → `runs/202_odinw_yoloe` | (RESULTS §4c, 생성기 없음) |
| W4A4 | s·m W4A4, s ablation (2 seed) | `w4a4.tsv` → `runs/203_w4a4`, `w4a4_m_abl.tsv` → `runs/205_w4a4_m_abl` | (RESULTS §4d) |
| PD-Quant | s W4A8·W4A5, seed 0~2 (진행 중) | `pdquant.tsv` → `runs/206_pdquant` | – |
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
- 앞 작업이 끝나기를 기다리는 스크립트에서 `pgrep -f "<경로>"`를 쓰면, 그 스크립트를 띄운 셸 명령에도 같은 문자열이 들어 있어 영원히 기다린다(10-27 사고). PID로 기다린다(`while kill -0 <pid>`).

---

## 7. 알려진 한계와 남은 일

| 항목 | 상태 |
|---|---|
| s W4A5 seed 0에서 BRECQ·QDrop 기준선이 붕괴(COCO 19.3 / 9.8) | 원인 미상. 12.cv2 재구성 실패 가설은 기각됨 |
| Combined 1단계와 BRECQ 기준선의 재구성 단위가 다름 | Combined의 1단계 BRECQ는 neck을 conv 단위로, `brecq` 조건은 전부 block 단위로 재구성한다(`run_comparison.py`의 `build`). 표 1의 Combined−BRECQ 차이에 이 차이가 섞여 있다. 단위를 맞춘 재측정은 미실행 |
| `brecq+GPM`이 조건 묶음에 따라 달라짐 | s W4A8 seed 0에서 LVIS AP 23.47(대기열 묶음) vs 23.71(단독). 원인 미상, G가 들어간 BRECQ 재구성에서 갈라짐(`docs/HISTORY.md` §14) |
| V2 W8A8에서 M이 flip을 늘림(+2.05%p) | COCO 어휘 검사로도, 일반 어휘 검사로도 사전에 못 잡는다. GP는 flip을 줄인다(−0.65%p) |
| m에서 M이 해로운 이유 | 가설만 있음(m은 앞쪽 backbone 보호가 없음). 검증 실험 미실행 |
| W8A8 이득 | s에서만 유의. v2·m에서는 일반화되지 않음 |
| 기준선 | SmoothQuant·AWQ는 10-17에 추가(표 1). PD-Quant(CVPR'23)는 10-28 구현, 파일럿 진행 중. InlierQ(ICLR'26)·Reg-PTQ(CVPR'24)는 공식 코드가 없어 관련 연구로만 다룬다 |
| 빠진 측정 | 실제 지연 시간(지금은 fake-quant 시뮬레이션과 이론 크기만), m의 FP LVIS AP |
| 모델 범위 | YOLO-World S/v2/M, YOLOE-v8s/11s, OWLv2. 다음 후보는 YOLOE-seg(instance segmentation), YOLO-World-L, calibration 데이터 교체 |
| W4A4 | 9월에는 한계로 적었으나 GPM으로 동작함을 확인(RESULTS §4d, 2 seed) |
