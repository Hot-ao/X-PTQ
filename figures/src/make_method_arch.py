"""방법 그림 -- YOLO-World 논문 그림 문법판 (11-04). 사용: .venv/bin/python figures/src/make_method_arch.py
YOLO-World 논문 Fig. 3(전체 구조)의 배치와 이름(YOLO Backbone, Multi-scale Image Features, Vision-Language PAN,
Text Contrastive Head, Box Head, Region-Text Matching)을 그대로 두고, 그 위에 우리 단계 D/G/P/M/R이 어디에 작용하는지 표시한다.
아래 줄은 Fig. 4(RepVL-PAN)의 T-CSPLayer 상자 문법으로 G(게이트 교환)와 M(공유 scale 이전)을 펼친다.
  - 왼쪽 위 점선 영역(YOLO-World의 'Training: Online Vocabulary' 자리) = 누수 없는 진단. 막대는 실측값이다:
    configs/protect_cache의 YOLO-World-S 층별 COCO flip 증가(calibration 쪽). 초록 = G가 다루는 cv2, 보라 = 새 규칙 P 선택.
  - 입력/출력 사진 = val2017 446651, 출력 박스 = Ours W4A4 seed 0 검출(qual_s_w4a4/ours.json, 위치마다 최고 점수, >= 0.25).
    같은 이미지에서 BRECQ W4A4는 검출이 0개다(qual_grid 첫 블록과 같은 사진).
make_method_overview.py(기존 Fig. 2 초안, 세부 수치 포함)는 그대로 둔다. 출력: figures/method_arch.{pdf,png}"""
import json, os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
from PIL import Image

SRC = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.dirname(SRC); ROOT = os.path.dirname(FIG)
sys.path.insert(0, SRC)
plt.rcParams.update({'font.family': 'Liberation Serif', 'mathtext.fontset': 'stix', 'pdf.fonttype': 42})

# YOLO-World 그림의 색(모듈) + 우리 단계 배지 색
INK, MUTED, LINE = '#1d1d1f', '#5f6368', '#222222'
MOD = ('#e2f0d9', '#6a9f4b')       # YOLO 모듈(초록)
PAN = ('#fff2c6', '#d9a520')       # Vision-Language PAN(노랑)
TXE = ('#dce8f7', '#5b86c4')       # 텍스트 인코더(파랑)
PANEL = ('#f2f2f2', '#f2f2f2')     # 임베딩 패널(회색 면)
ORANGE = '#e8894a'                 # 텍스트 -> 이미지 화살표
D = '#8a6d0b'; G = '#2c7d4d'; P = '#6a3fa0'; M = '#1c7688'; R = '#59626c'; BAD = '#c0392b'
VOC = [('helmet', '#8cc474'), ('jeep', '#7aa7dc'), ('sunglasses', '#ee9a5a')]

fig = plt.figure(figsize=(7.0, 4.6))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 140); ax.set_ylim(0, 92); ax.axis('off')


def box(x, y, w, h, text='', c=MOD, fs=7, lw=0.8, r=0.7, tc=INK, ls='-', z=3, fc=None, ec=None, **kw):
    ax.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f'round,pad={r}', fc=fc or c[0],
                                ec=ec or c[1], lw=lw, ls=ls, zorder=z))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc, zorder=z + 1,
                linespacing=1.15, **kw)


def arrow(pts, c=LINE, lw=0.9, z=2, ls='-', ms=6, style='-|>'):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=lw, ls=ls, zorder=z, solid_capstyle='butt')
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle=style, mutation_scale=ms, color=c, lw=lw, ls=ls,
                                 shrinkA=0, shrinkB=0, zorder=z))


def fat(p0, p1, c='#b9b9b9'):
    """YOLO-World 그림의 속 빈 굵은 화살표."""
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='simple,head_width=1.1,head_length=0.9,tail_width=0.45',
                                 mutation_scale=7, fc='white', ec=c, lw=0.8, zorder=2))


def op(x, y, s, c=LINE, r=1.0, fs=7):
    ax.add_patch(Circle((x, y), r, fc='white', ec=c, lw=0.8, zorder=4))
    ax.text(x, y, s, ha='center', va='center', fontsize=fs, color=c, zorder=5)


def badge(x, y, s, c, r=1.35, fs=7):
    ax.add_patch(Circle((x, y), r, fc=c, ec='white', lw=0.8, zorder=7))
    ax.text(x, y - 0.05, s, ha='center', va='center', fontsize=fs, color='white', weight='bold', zorder=8,
            family='DejaVu Sans')


def photo(path, x, y, w, crop=None, z=3):
    im = Image.open(path).convert('RGB')
    if crop:
        im = im.crop(crop)
    h = w * im.height / im.width
    ax.imshow(im, extent=(x, x + w, y, y + h), zorder=z, interpolation='bilinear')
    ax.add_patch(Rectangle((x, y), w, h, fc='none', ec='#9a9a9a', lw=0.5, zorder=z + 0.1))
    return im, h


def emb_panel(x, y, w, h, title, labels=True):
    box(x, y, w, h, c=PANEL, lw=0, r=1.0, z=1)
    ax.text(x + w / 2, y + h + 1.4, title, ha='center', va='center', fontsize=7, color=INK)
    for i, (n, col) in enumerate(VOC):
        cy = y + h - 3.2 - i * 4.3
        ax.add_patch(Circle((x + 3.2, cy), 1.45, fc=col, ec='#4d4d4d', lw=0.5, zorder=3))
        if labels:
            ax.text(x + 5.6, cy, n, ha='left', va='center', fontsize=6.5, color=col if col != '#8cc474' else '#5f9a43')


# ================================================================ 위: 진단(D) + 텍스트 경로
# --- D: 왼쪽 위 점선 영역 (YOLO-World의 'Training: Online Vocabulary' 자리)
box(1, 64.2, 49, 26.8, c=('none', '#8f8f8f'), fc='white', lw=0.8, ls=(0, (3, 2)), r=0.6, z=1)
badge(3.6, 88.2, 'D', D)
ax.text(5.6, 88.2, 'Calibration side: Leak-free Diagnosis', ha='left', va='center', fontsize=7.5, color=D, style='italic')
tr = sorted(os.listdir('/data/taeho/coco_datasets/train2017'))
for k, f in enumerate(tr[:3][::-1]):
    photo(os.path.join('/data/taeho/coco_datasets/train2017', f), 3.0 + k * 1.3, 72.5 + k * 1.3, 9.0, z=3 + k)
ax.text(9.4, 70.3, 'calib. images + vocab.', ha='center', va='center', fontsize=5.6, color=MUTED)
box(16.5, 73.5, 13.0, 9.5, 'quantize one\nconv to W4,\nmeasure top-1\nflip', c=('#fbf1d3', D), fs=6.3, lw=0.8)
arrow([(13.6, 78.2), (16.5, 78.2)])
# 실측 층별 손상 (YOLO-World-S, calibration 쪽 COCO flip 증가 pp)
from diag_concentration import metrics
rows = json.load(open(os.path.join(ROOT, 'configs', 'protect_cache', 'yolov8s-world_n200_c256_img640_la16_aqattn_cls.json')))
gset = {f'{i}.cv2.conv' for i in (12, 15, 19, 22)}
psel = metrics(rows, gset)['psel']
bx, by, bw, bh, cap = 32.0, 71.5, 16.5, 11.5, 6.0
ax.plot([bx - 0.3, bx + bw], [by, by], color=MUTED, lw=0.5)
step = bw / len(rows)
for i, r in enumerate(rows):
    v = max(r['coco'], 0.0)
    col = G if r['name'] in gset else P if r['name'] in psel else '#b8b8b8'
    ax.add_patch(Rectangle((bx + i * step, by), step * 0.8, min(v, cap) / cap * bh, fc=col, ec='none', zorder=3))
    if v > cap:
        ax.text(bx + i * step + step * 0.4, by + bh + 0.9, f'{v:.0f} pp', ha='center', va='center', fontsize=5.3, color=G)
ax.text(bx + bw / 2, by - 1.4, 'per-conv damage (YOLO-World-S)', ha='center', va='center', fontsize=5.6, color=MUTED)
arrow([(29.5, 78.2), (31.6, 78.2)])
# 결정
ax.add_patch(Rectangle((3.0, 66.9), 1.4, 1.4, fc=G, ec='none')); ax.text(5.0, 67.6, 'text-gated fusion conv → G', fontsize=5.9, va='center', color=INK)
ax.add_patch(Rectangle((3.0, 64.9), 1.4, 1.4, fc=P, ec='none')); ax.text(5.0, 65.6, 'other top convs, 1.5% budget → P', fontsize=5.9, va='center', color=INK)
ax.text(48.5, 66.6, 'M: kept only if a calib.-side\nflip check approves it', fontsize=5.7, va='center', ha='right', color=M, linespacing=1.05)

# --- 텍스트 인코더 (FP, 오프라인)
box(54, 77.5, 14.5, 11.5, '', c=TXE, lw=0.8)
ax.text(61.25, 84.6, 'Text\nEncoder', ha='center', va='center', fontsize=8, color=INK, zorder=5, linespacing=1.1)
ax.text(61.25, 79.6, 'FP32 · run once offline', ha='center', va='center', fontsize=5.4, color=TXE[1], style='italic', zorder=5)
ax.text(66.6, 87.2, '❄', fontsize=8, ha='center', va='center', color='#2b4d7a', family='DejaVu Sans', zorder=6)
# 사용자 어휘 구름
cx, cy = 61.25, 69.0
for dx, dy, rr in ((-3.6, 0, 2.3), (-1.2, 1.4, 2.6), (2.0, 1.2, 2.5), (4.0, -0.3, 2.1), (0.6, -1.3, 2.4), (-2.6, -1.2, 2.0)):
    ax.add_patch(Circle((cx + dx, cy + dy), rr, fc='white', ec='#444', lw=0.6, zorder=2))
for dx, dy, rr in ((-3.6, 0, 2.3), (-1.2, 1.4, 2.6), (2.0, 1.2, 2.5), (4.0, -0.3, 2.1), (0.6, -1.3, 2.4), (-2.6, -1.2, 2.0)):
    ax.add_patch(Circle((cx + dx, cy + dy), rr - 0.35, fc='white', ec='none', zorder=2.5))
ax.text(cx, cy + 0.1, "User's\nVocabulary", ha='center', va='center', fontsize=6.0, zorder=3, linespacing=1.05)
fat((cx, 72.5), (cx, 77.2))

emb_panel(72, 74, 17.5, 13.5, 'Vocabulary Embeddings')
arrow([(68.5, 83.2), (72.0, 83.2)])

# ================================================================ 가운데: 양자화된 검출기
# 양자화 영역(점선): 백본 ~ 헤드. 텍스트 인코더는 밖(FP)
box(19.5, 25.5, 96.5, 37.5, c=('none', R), fc='#fafafa', lw=0.8, ls=(0, (4, 2)), r=0.8, z=0)
badge(22.0, 27.9, 'R', R, r=1.2, fs=6.3)
ax.text(23.8, 27.9, 'Quantized detector: W4, A4–A8 (convs, attention, region–text products); '
        'every block reconstructed by BRECQ / QDrop, unchanged', ha='left', va='center', fontsize=5.8, color=R)

# 입력 사진
IMG = '/data/taeho/coco_datasets/val2017/000000446651.jpg'
_, ih = photo(IMG, 1.5, 38.5, 16.0)
ax.text(9.5, 36.8, 'Input Image', ha='center', va='center', fontsize=7)

# 백본 + P
box(23.5, 38.5, 15.5, 11.5, 'YOLO\nBackbone', fs=8, lw=0.8)
badge(38.4, 49.6, 'P', P)
ax.text(31.25, 35.9, 'P: diagnosed convs keep W8\n(9 convs, 1.6% of weights)', ha='center', va='center', fontsize=5.6,
        color=P, linespacing=1.1)
arrow([(17.5, 44.2), (23.5, 44.2)])

# 다중 스케일 특징(쌓인 판)
ax.text(51.5, 60.0, 'Multi-scale\nImage Features', ha='center', va='center', fontsize=6.8, linespacing=1.1)
for k, (w, yy) in enumerate(((7.0, 52.0), (9.5, 45.0), (12.0, 37.0))):
    x0 = 51.5 - w / 2
    ax.add_patch(Polygon([(x0, yy), (x0 + w, yy), (x0 + w + 2.2, yy + 2.0), (x0 + 2.2, yy + 2.0)], closed=True,
                         fc='white', ec='#333', lw=0.6, zorder=3))
    ax.add_patch(Polygon([(x0, yy), (x0 + w, yy), (x0 + w, yy - 0.9), (x0, yy - 0.9)], closed=True, fc='white',
                         ec='#333', lw=0.6, zorder=3))
    ax.add_patch(Polygon([(x0 + w, yy), (x0 + w + 2.2, yy + 2.0), (x0 + w + 2.2, yy + 1.1), (x0 + w, yy - 0.9)],
                         closed=True, fc='white', ec='#333', lw=0.6, zorder=3))
arrow([(39.0, 44.2), (44.0, 44.2)])
for yy in (53.0, 46.0, 38.0):
    arrow([(60.5 if yy < 50 else 58.0, yy), (77.5, yy)])

# Vision-Language PAN + G
box(77.5, 31.0, 6.5, 32.0, '', c=PAN, lw=0.9, r=0.8)
ax.text(80.75, 47.0, 'Vision-Language PAN', rotation=-90, ha='center', va='center', fontsize=7.5, zorder=5)
badge(77.7, 62.6, 'G', G)
ax.text(75.8, 60.4, 'G in every\nT-CSPLayer\n(see below)', ha='right', va='center', fontsize=5.6, color=G, linespacing=1.05)
arrow([(80.75, 74.0), (80.75, 63.0)], c=ORANGE, lw=1.1)

# PAN 위쪽: 이미지 인식 임베딩 -> Region-Text Matching
emb_panel(96, 74, 15.5, 13.5, 'Image-aware Embeddings')
arrow([(84.0, 61.0), (89.0, 61.0), (89.0, 80.7), (96.0, 80.7)])
# 매칭 격자
gx, gy, cs = 120.5, 74.0, 3.5
ax.text(gx + 2 * cs, gy + 3 * cs + 1.4, 'Region-Text Matching', ha='center', va='center', fontsize=7)
hit = {(0, 0): VOC[0][1], (1, 1): VOC[1][1], (2, 2): VOC[2][1]}
for i in range(3):
    for j in range(4):
        ax.add_patch(Rectangle((gx + j * cs, gy + (2 - i) * cs), cs, cs, fc=hit.get((i, j), 'white'), ec='#222',
                               lw=0.6, zorder=3))
fat((112.5, 80.7), (119.5, 80.7))

# 헤드
box(92.5, 46.0, 18.0, 6.5, 'Text\nContrastive Head', fs=6.8, lw=0.8)
box(92.5, 36.0, 18.0, 6.0, 'Box Head', fs=6.8, lw=0.8)
arrow([(84.0, 49.25), (92.5, 49.25)]); arrow([(84.0, 39.0), (92.5, 39.0)])
# 객체 임베딩
box(119.5, 55.0, 18.0, 9.0, c=PANEL, lw=0, r=0.8, z=1)
for k in range(4):
    ax.add_patch(Rectangle((121.0 + k * 4.1, 59.5), 2.8, 2.8, fc='#dcd0ec', ec='#7b5aa6', lw=0.6, zorder=3))
ax.text(128.5, 57.0, 'Object Embeddings', ha='center', va='center', fontsize=6.6)
arrow([(101.5, 52.5), (101.5, 57.5), (119.5, 57.5)], ms=5)
fat((128.5, 64.4), (128.5, 73.6))

# 출력 사진 + Ours W4A4 검출
from qual_figure import top_per_location
dets = json.load(open('/data/taeho/cvpr2027_dets/qual_s_w4a4/ours.json'))['446651']
ox, oy, ow = 119.5, 33.5, 18.0
im, oh = photo(IMG, ox, oy, ow, z=3)
sx = ow / im.width
cls_col = {'helmet': VOC[0][1], 'jeep': VOC[1][1], 'sunglasses': VOC[2][1]}
from lvis import LVIS
from qual_figure import LVIS_ANN
names = {c['id'] - 1: c['name'] for c in LVIS(LVIS_ANN).dataset['categories']}
for b in top_per_location(dets, 0.25):
    n = names[b[5]]
    x1, y1, x2, y2 = b[:4]
    ax.add_patch(Rectangle((ox + x1 * sx, oy + oh - y2 * sx), (x2 - x1) * sx, (y2 - y1) * sx, fc='none',
                           ec=cls_col.get(n, '#dddddd'), lw=1.0, zorder=5))
ax.text(ox + ow / 2, oy - 1.4, 'Ours, W4A4 (BRECQ: 0 boxes)', ha='center', va='center', fontsize=5.8, color=INK)
arrow([(110.5, 39.0), (119.5, 39.0)], ms=5)
arrow([(110.5, 49.25), (114.5, 49.25), (114.5, 42.0), (119.5, 42.0)], ms=5)

# M: 생산 conv -> 소비 conv, 검출기 전체
ax.plot([23.5, 23.5, 110.5, 110.5], [31.6, 30.8, 30.8, 31.6], color=M, lw=0.9, zorder=2)
badge(26.0, 30.8, 'M', M, r=1.2, fs=6.3)
ax.text(27.8, 30.8, 'per-channel scale moved from every producer conv into all of its consumer convs (one shared s)',
        ha='left', va='center', fontsize=5.8, color=M, bbox=dict(fc='#fafafa', ec='none', pad=0.6), zorder=3)

# ================================================================ 아래: G, M 펼침 (YOLO-World Fig. 4 상자 문법)
ax.plot([1, 139], [24.2, 24.2], color='#d0d0d0', lw=0.6)
# --- G
box(1, 0.6, 91, 22.6, c=('none', G), fc='white', lw=0.8, ls=(0, (3, 2)), r=0.6, z=0)
badge(3.6, 20.8, 'G', G)
ax.text(5.6, 20.8, 'Gate commutation in the T-CSPLayer (C2f block)', ha='left', va='center', fontsize=7.5, color=G)
ax.text(90.5, 20.8, 'FP-equivalent rewrite · +0 bits', ha='right', va='center', fontsize=5.8, color=MUTED, style='italic')


def tcsp(x0, after):
    """Fig. 4의 T-CSPLayer: Split -> Dark Bottleneck -> Max-Sigmoid -> Concat -> cv2."""
    yc = 12.0
    arrow([(x0, yc), (x0 + 2.2, yc)])
    op(x0 + 3.2, yc, 'S', fs=6)
    ax.plot([x0 + 3.2, x0 + 3.2, x0 + 22.0], [yc + 1.0, 15.8, 15.8], color=LINE, lw=0.8)           # y0, y1 통과
    ax.text(x0 + 17.0, 16.6, '$y_0, y_1$', fontsize=6, color=MUTED, va='center')
    box(x0 + 5.5, yc - 1.9, 8.0, 3.8, 'Dark\nBottleneck', c=('white', '#333'), fs=5.6, lw=0.7, r=0.4)
    arrow([(x0 + 4.2, yc), (x0 + 5.5, yc)])
    ax.text(x0 + 14.8, yc + 0.9, '$y_2$', fontsize=6, color=MUTED, va='center')
    yt = 5.6                                                                                    # 텍스트 가지 높이
    if not after:
        ax.plot([x0 + 13.5, x0 + 22.0], [yc, yc], color=LINE, lw=0.8)
        ax.plot([x0 + 15.5, x0 + 15.5], [yc, yt + 1.9], color=LINE, lw=0.8)
        box(x0 + 13.0, yt - 1.9, 9.0, 3.8, 'Max-Sigmoid', c=MOD, fs=5.8, lw=0.7, r=0.4)
        arrow([(x0 + 17.5, 1.5), (x0 + 17.5, yt - 1.9)], c=ORANGE, lw=1.0, ms=5)
        ax.text(x0 + 18.4, 2.1, 'Text', fontsize=5.6, color=ORANGE, va='center')
        arrow([(x0 + 22.0, yt), (x0 + 23.3, yt), (x0 + 23.3, yc - 1.0)], c=ORANGE, lw=0.9, ms=5)
        ax.text(x0 + 24.0, 8.0, '$y_3 = p\\odot a$', fontsize=6, color=ORANGE, va='center')
        ax.plot([x0 + 22.0, x0 + 23.3], [15.8, 15.8], color=LINE, lw=0.8); ax.plot([x0 + 23.3, x0 + 23.3], [15.8, yc + 1.0], color=LINE, lw=0.8)
        ax.plot([x0 + 22.0, x0 + 22.3], [yc, yc], color=LINE, lw=0.8)
        op(x0 + 23.3, yc, 'C', fs=6)
        box(x0 + 26.0, yc - 2.4, 9.5, 4.8, 'cv2  1×1\none quantizer', c=('#fbe9e7', BAD), fs=5.8, lw=1.0, r=0.4)
        arrow([(x0 + 24.3, yc), (x0 + 26.0, yc)])
        arrow([(x0 + 35.5, yc), (x0 + 38.0, yc)])
        ax.text(x0 + 30.75, 6.0, 'gated $y_3$ (max 20.4) sets\nthe range for $y_0$–$y_2$ (≤ 9.3)',
                fontsize=5.4, color=BAD, ha='left', va='center', linespacing=1.1)
        ax.text(x0, 18.3, 'YOLO-World (original)', fontsize=6.4, ha='left', va='center', color=INK)
        return
    # 바꾼 뒤: 게이트를 conv 뒤로
    ax.plot([x0 + 13.5, x0 + 22.3], [yc, yc], color=LINE, lw=0.8)
    ax.plot([x0 + 22.0, x0 + 23.3, x0 + 23.3], [15.8, 15.8, yc + 1.0], color=LINE, lw=0.8)
    op(x0 + 23.3, yc, 'C', fs=6)
    box(x0 + 26.0, yc - 2.0, 9.5, 4.4, 'cv2$_{\\rm main}$  1×1\nquantizer A', c=('#dff1e3', G), fs=5.6, lw=1.0, r=0.4)
    arrow([(x0 + 24.3, yc), (x0 + 26.0, yc)])
    # 텍스트 가지: p = proj(y2) -> cv2_side -> (*) a
    ax.plot([x0 + 15.5, x0 + 15.5, x0 + 16.5], [yc, yt, yt], color=LINE, lw=0.8)
    ax.text(x0 + 15.0, 7.9, '$p$', fontsize=6, color=MUTED, ha='right', va='center')
    box(x0 + 16.5, yt - 2.0, 9.5, 4.0, 'cv2$_{\\rm side}$  1×1\nquantizer B', c=('#dff1e3', G), fs=5.6, lw=1.0, r=0.4)
    op(x0 + 28.6, yt, '$\\odot$', c=ORANGE, r=0.95, fs=6.5)
    arrow([(x0 + 26.0, yt), (x0 + 27.65, yt)])
    box(x0 + 31.0, yt - 3.6, 7.5, 3.2, 'gate $a$', c=MOD, fs=5.6, lw=0.7, r=0.4)
    ax.text(x0 + 34.75, yt - 2.0 - 2.4 + 0.15, '', fontsize=1)
    arrow([(x0 + 31.0, yt - 2.0), (x0 + 28.6, yt - 2.0), (x0 + 28.6, yt - 0.95)], c=ORANGE, lw=0.9, ms=5)
    arrow([(x0 + 39.8, yt - 2.0), (x0 + 38.5, yt - 2.0)], c=ORANGE, lw=1.0, ms=5)
    ax.text(x0 + 40.1, yt - 2.0, 'Text', fontsize=5.6, color=ORANGE, va='center')
    op(x0 + 37.5, yc, '+', r=0.95, fs=7)
    arrow([(x0 + 35.5, yc), (x0 + 36.55, yc)])
    arrow([(x0 + 29.55, yt), (x0 + 37.5, yt), (x0 + 37.5, yc - 0.95)], c=ORANGE, lw=0.9, ms=5)
    arrow([(x0 + 38.45, yc), (x0 + 40.5, yc)])
    ax.text(x0, 18.3, 'After G (ours): gate applied after the conv', fontsize=6.4, ha='left', va='center', color=G)


tcsp(2.5, after=False)
ax.plot([46.0, 46.0], [2.0, 19.0], color='#e0e0e0', lw=0.6)
tcsp(48.0, after=True)

# --- M
box(94, 0.6, 45, 22.6, c=('none', M), fc='white', lw=0.8, ls=(0, (3, 2)), r=0.6, z=0)
badge(96.6, 20.8, 'M', M)
ax.text(98.6, 20.8, 'Tied scale migration', ha='left', va='center', fontsize=7.5, color=M)
box(96.0, 9.5, 10.0, 5.6, 'Producer\nconv', c=('white', '#333'), fs=6.2, lw=0.7, r=0.4)
box(108.5, 10.5, 5.0, 3.6, '$\\div\\, s$', c=('#d8eef2', M), fs=7, lw=1.0, r=0.4)
arrow([(106.0, 12.3), (108.5, 12.3)])
box(119.0, 13.6, 18.0, 4.6, 'Consumer A:  $W_A\\,\\mathrm{diag}(s)$', c=('#d8eef2', M), fs=6, lw=0.8, r=0.4)
box(119.0, 6.4, 18.0, 4.6, 'Consumer B:  $W_B\\,\\mathrm{diag}(s)$', c=('#d8eef2', M), fs=6, lw=0.8, r=0.4)
arrow([(113.5, 12.3), (116.0, 12.3), (116.0, 15.9), (119.0, 15.9)])
arrow([(116.0, 12.3), (116.0, 8.7), (119.0, 8.7)])
ax.text(128.0, 12.3, 'same $s$', fontsize=6, color=M, ha='center', va='center', style='italic')
ax.text(111.0, 8.4, 'folded into the\nproducer’s requant', fontsize=5.3, color=MUTED, ha='center', va='center', linespacing=1.05)
ax.text(116.5, 3.2, '$\\mathrm{conv}(x, W) = \\mathrm{conv}(x/s,\\, W\\mathrm{diag}(s))$ · flattens channel ranges of $x$',
        fontsize=5.8, color=INK, ha='center', va='center')

for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(FIG, f'method_arch.{ext}'), dpi=300, facecolor='white')
print('저장:', os.path.join(FIG, 'method_arch.png'), '/ .pdf')
