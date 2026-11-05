"""방법 흐름 그림 (11-04). 사용: .venv/bin/python figures/src/make_method_flow.py
make_method_arch.py(위치 그림)와 짝을 이루는 순서 그림이다. 흐름:
  D 누수 없는 진단 -> G(구조로 정해진 T-CSPLayer cv2 4곳) + P(나머지 진단 상위 conv, weight만 W8, 1.5%)
  -> 후보 두 개: GP = calibrate + BRECQ, GPM = M(생산 conv 쪽 x/s, 소비 conv W*s) + calibrate + BRECQ
  -> calibration 밖 train2017 200장 + COCO 어휘의 top-1 flip으로 하나 선택(flip_GPM <= flip_GP 이면 GPM, rho = 1.0, seed마다)
막대는 실측(configs/protect_cache, YOLO-World-S 층별 COCO flip 증가). 근거: documents/FIGURE_METHOD_PLACEMENT_2026-11-04.md
출력: figures/method_flow.{pdf,png}"""
import json, os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from PIL import Image

SRC = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.dirname(SRC); ROOT = os.path.dirname(FIG)
sys.path.insert(0, SRC)
plt.rcParams.update({'font.family': 'Liberation Serif', 'mathtext.fontset': 'stix', 'pdf.fonttype': 42})

INK, MUTED, LINE = '#1d1d1f', '#5f6368', '#222222'
MOD = ('#e2f0d9', '#6a9f4b')
ORANGE = '#e8894a'
D = '#8a6d0b'; G = '#2c7d4d'; P = '#6a3fa0'; M = '#1c7688'; R = '#59626c'; OK = '#2a78d6'
DL, GL, PL, ML, RL = '#fbf1d3', '#dff1e3', '#ece2f6', '#d8eef2', '#eceef0'
# P 패널: backbone 블록별 conv(데이터 흐름 순서)와 보호 목록(+G 뒤 다시 채운 9개, yolov8s-world)
BLOCK_CONVS = [['0'], ['1'], ['2.cv1', '2.m.0.cv1', '2.m.0.cv2', '2.cv2'], ['3'],
               ['4.cv1', '4.m.0.cv1', '4.m.0.cv2', '4.m.1.cv1', '4.m.1.cv2', '4.cv2'], ['5'],
               ['6.cv1', '6.m.0.cv1', '6.m.0.cv2', '6.m.1.cv1', '6.m.1.cv2', '6.cv2'], ['7'],
               ['8.cv1', '8.m.0.cv1', '8.m.0.cv2', '8.cv2'], ['9.cv1', '9.cv2']]
PROTECTED = {'1', '2.cv1', '2.cv2', '2.m.0.cv1', '2.m.0.cv2', '4.cv1', '4.cv2', '4.m.1.cv1', '6.cv1'}
FAINT, STEM, CELL = '#dddcd6', '#8a8a8a', 1.0
LEGEND = ((5.5, 'W8 by P', P), (15.5, 'W4', FAINT), (22.0, 'W8 by protocol*', STEM))   # (패널 왼쪽부터 x, 글, 색)


def conv_cells(x, py, n):
    """블록(폭 3.0) 안 conv 사각형의 왼쪽 아래 좌표. 2열로 위에서부터, 하나면 가운데."""
    cols = 1 if n == 1 else 2
    x0 = x + (3.0 - cols * CELL - (cols - 1) * 0.3) / 2
    return [(x0 + (k % cols) * (CELL + 0.3), py + 12.9 - (k // cols) * (CELL + 0.3)) for k in range(n)]


def cell_color(c):
    return STEM if c == '0' else (P if c in PROTECTED else FAINT)


W, H = 7.0, 3.5
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 140); ax.set_ylim(0, 70); ax.axis('off')


def box(x, y, w, h, text='', fc='white', ec=LINE, fs=6.5, lw=0.8, r=0.6, tc=INK, ls='-', z=3, **kw):
    ax.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f'round,pad={r}', fc=fc, ec=ec,
                                lw=lw, ls=ls, zorder=z))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc, zorder=z + 1,
                linespacing=1.15, **kw)


def arrow(pts, c=LINE, lw=0.8, z=2, ms=5.5):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=lw, zorder=z, solid_capstyle='butt')
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle='-|>', mutation_scale=ms, color=c, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=z))


def op(x, y, s, c=LINE, r=0.9, fs=6.5):
    ax.add_patch(Circle((x, y), r, fc='white', ec=c, lw=0.7, zorder=4))
    ax.text(x, y, s, ha='center', va='center', fontsize=fs, color=c, zorder=5)


def badge(x, y, s, c, r=1.3, fs=6.8):
    ax.add_patch(Circle((x, y), r, fc=c, ec='white', lw=0.8, zorder=7))
    ax.text(x, y - 0.05, s, ha='center', va='center', fontsize=fs, color='white', weight='bold', zorder=8,
            family='DejaVu Sans')


def panel(x, y, w, h, c, cl, key, title, sub=''):
    box(x, y, w, h, fc=cl, ec=c, lw=0.9, r=0.8, z=0)
    badge(x + 2.4, y + h - 2.4, key, c)
    ax.text(x + 4.4, y + h - 2.4, title, ha='left', va='center', fontsize=7.2, color=c, weight='bold')
    if sub:
        ax.text(x + w - 1.2, y + h - 2.4, sub, ha='right', va='center', fontsize=5.6, color=MUTED, style='italic')


def thumbs(x, y, w, files, z=3):
    for k, f in enumerate(files[::-1]):
        im = Image.open(f).convert('RGB')
        cw, ch = im.size; tw = min(cw, ch * 4 // 3); th = tw * 3 // 4
        im = im.crop(((cw - tw) // 2, (ch - th) // 2, (cw - tw) // 2 + tw, (ch - th) // 2 + th))
        h = w * im.height / im.width
        xx, yy = x + k * 1.1, y + k * 1.1
        ax.imshow(im, extent=(xx, xx + w, yy, yy + h), zorder=z + k)
        ax.add_patch(Rectangle((xx, yy), w, h, fc='none', ec='#9a9a9a', lw=0.5, zorder=z + k + 0.1))


TRAIN = '/data/taeho/coco_datasets/train2017'
tr = sorted(os.listdir(TRAIN))
Y0 = 9.5                                   # 그림 바닥(이 아래는 잘라 냄)

# ---------------------------------------------------------------- 0. 입력
box(1, 50, 15, 9, 'YOLO-World\n(FP32)', fc=MOD[0], ec=MOD[1], fs=7)
thumbs(2.5, 33.5, 9.0, [os.path.join(TRAIN, f) for f in tr[:3]])
ax.text(8.5, 30.6, 'calibration images\n(256, COCO vocab.)', ha='center', va='center', fontsize=5.7, color=MUTED,
        linespacing=1.1)
ax.plot([16.6, 17.6, 17.6, 16.6], [54.5, 54.5, 37.5, 37.5], color=LINE, lw=0.8)
arrow([(17.6, 46.0), (20.0, 46.0)])

# ---------------------------------------------------------------- 1. D
panel(20, Y0 + 1, 26, 68.5 - Y0 - 1, D, DL, 'D', 'Leak-free diagnosis')
box(22, 55.5, 22, 6.5, 'quantize one conv to W4\n(rest W8A8), measure\nCOCO top-1 flip vs. FP',
    fc='white', ec=D, fs=5.7)
from diag_concentration import metrics
rows = json.load(open(os.path.join(ROOT, 'configs', 'protect_cache', 'yolov8s-world_n200_c256_img640_la16_aqattn_cls.json')))
gset = {f'{i}.cv2.conv' for i in (12, 15, 19, 22)}
psel = metrics(rows, gset)['psel']
bx, by, bw, bh, cap = 22.5, 32.0, 21.0, 15.0, 6.0
arrow([(33.0, 55.5), (33.0, 49.6)])
ax.plot([bx - 0.3, bx + bw], [by, by], color=MUTED, lw=0.5)
step = bw / len(rows)
for i, r in enumerate(rows):
    v = max(r['coco'], 0.0)
    col = G if r['name'] in gset else P if r['name'] in psel else '#b8b8b8'
    ax.add_patch(Rectangle((bx + i * step, by), step * 0.8, min(v, cap) / cap * bh, fc=col, ec='none', zorder=3))
    if v > cap:
        ax.text(bx + i * step + step * 0.4, by + bh + 1.0, f'12.cv2: {v:.0f} pp', ha='center', va='center',
                fontsize=5.4, color=G)
ax.text(bx + bw / 2, by - 1.5, 'flip increase per conv (YOLO-World-S)', ha='center', va='center',
        fontsize=5.3, color=MUTED)
ax.add_patch(Rectangle((22.3, 23.3), 1.4, 1.4, fc=G, ec='none'))
ax.text(24.3, 24.0, 'text-gated fusion convs → G', fontsize=5.8, va='center', color=INK)
ax.text(24.3, 21.7, '(all T-CSPLayer cv2, by structure)', fontsize=5.2, va='center', color=MUTED)
ax.add_patch(Rectangle((22.3, 16.7), 1.4, 1.4, fc=P, ec='none'))
ax.text(24.3, 17.4, 'other top convs → P', fontsize=5.8, va='center', color=INK)
ax.text(24.3, 15.1, '(greedy by damage, 1.5% budget)', fontsize=5.2, va='center', color=MUTED)

# ---------------------------------------------------------------- 2. G
GX, GY, GW, GH = 49, 37.5, 39, 31.0
panel(GX, GY, GW, GH, G, GL, 'G', 'Gate commutation', 'FP-equivalent · +0 bits')
ax.text(GX + 4.4, GY + GH - 5.0, 'in every T-CSPLayer (12, 15, 19, 22)', ha='left', va='center', fontsize=5.5, color=G)
yr = {'y0': 59.0, 'y1': 56.0, 'y2': 53.0}
for k, yy in yr.items():
    box(GX + 1.5, yy - 1.1, 4.2, 2.2, f'$y_{k[1]}$', fs=6.2, lw=0.6, r=0.3)
box(GX + 8.0, 51.6, 1.8, 8.8, fc='#e9eff8', ec='#3f5f96', lw=0.6, r=0.3)
ax.text(GX + 8.9, 56.0, 'C', fontsize=5.8, ha='center', va='center', zorder=5)
for k, yy in yr.items():
    arrow([(GX + 5.7, yy), (GX + 8.0, yy)], ms=4)
box(GX + 12.0, 53.5, 10.5, 5.0, 'cv2$_{\\rm main}$ 1×1\nquantizer A', fc='white', ec=G, fs=5.6, lw=1.0, r=0.4)
arrow([(GX + 9.8, 56.0), (GX + 12.0, 56.0)], ms=4)
op(GX + 30.5, 56.0, '+', fs=7)
arrow([(GX + 22.5, 56.0), (GX + 29.6, 56.0)], ms=4)
box(GX + 32.6, 54.4, 5.2, 3.2, 'SiLU', fs=5.6, lw=0.6, r=0.3)
arrow([(GX + 31.4, 56.0), (GX + 32.6, 56.0)], ms=4)
# 텍스트 분기: y2 -> proj 3x3 -> p -> cv2_side -> (*) a -> (+)
ax.plot([GX + 6.6, GX + 6.6], [53.0, 42.6], color=LINE, lw=0.7)
ax.plot([GX + 5.7, GX + 6.6], [53.0, 53.0], color=LINE, lw=0.7)
arrow([(GX + 6.6, 47.5), (GX + 8.0, 47.5)], ms=4)
box(GX + 8.0, 45.5, 7.0, 4.0, 'proj\n3×3', fs=5.4, lw=0.6, r=0.3)
ax.text(GX + 16.2, 48.5, '$p$', fontsize=6, color=MUTED, ha='center', va='center')
box(GX + 17.5, 45.0, 10.5, 5.0, 'cv2$_{\\rm side}$ 1×1\nquantizer B', fc='white', ec=G, fs=5.6, lw=1.0, r=0.4)
arrow([(GX + 15.0, 47.5), (GX + 17.5, 47.5)], ms=4)
op(GX + 30.5, 47.5, '$\\odot$', c=ORANGE, fs=6.5)
arrow([(GX + 28.0, 47.5), (GX + 29.6, 47.5)], ms=4)
arrow([(GX + 30.5, 48.4), (GX + 30.5, 55.1)], c=ORANGE, ms=4)
box(GX + 17.5, 41.4, 10.5, 2.6, 'gate $a(y_2,$ text$)$', fc=MOD[0], ec=MOD[1], fs=5.4, lw=0.6, r=0.3)
arrow([(GX + 6.6, 42.6), (GX + 17.5, 42.6)], ms=4)
arrow([(GX + 28.0, 42.6), (GX + 30.5, 42.6), (GX + 30.5, 46.6)], c=ORANGE, ms=4)
arrow([(GX + 22.75, 38.4), (GX + 22.75, 41.4)], c=ORANGE, ms=4)
ax.text(GX + 23.6, 38.9, 'text', fontsize=5.4, color=ORANGE, ha='left', va='center')
ax.text(GX + 34.2, 50.5, 'gate after\nthe conv', fontsize=5.3, color=G, ha='center', va='center', linespacing=1.05)

# ---------------------------------------------------------------- 2. P
PX, PY, PW, PH = 49, Y0 + 1, 39, 25.0
panel(PX, PY, PW, PH, P, PL, 'P', 'Protection')
ax.text(PX + 4.4, PY + PH - 5.0, 'YOLO Backbone (blocks 0–9)', ha='left', va='center', fontsize=5.5, color=P)
# 블록 안 작은 사각형 = conv 하나 (데이터 흐름 순서, 2열). 진한 보라 = P로 W8, 연한 회색 = W4, 회색 = 프로토콜 W8(stem)
for i, convs in enumerate(BLOCK_CONVS):
    x = PX + 1.6 + i * 3.6
    prot = any(c in PROTECTED for c in convs)
    box(x, PY + 9.0, 3.0, 7.6, '', fc='white', ec=P if prot else '#9a9a9a', lw=0.6, r=0.3)
    ax.text(x + 1.5, PY + 15.3, str(i), ha='center', va='center', fontsize=5.6, color=P if prot else INK, zorder=5)
    for (cx, cy), c in zip(conv_cells(x, PY, len(convs)), convs):
        ax.add_patch(Rectangle((cx, cy), CELL, CELL, fc=cell_color(c), ec='none', zorder=5))
for dx, lab, c in LEGEND:
    lx = PX + dx
    ax.add_patch(Rectangle((lx, PY + 6.0), CELL, CELL, fc=c, ec='none', zorder=5))
    ax.text(lx + CELL + 0.6, PY + 6.0 + CELL / 2, lab, ha='left', va='center', fontsize=5.0, color=INK)
ax.text(PX + PW / 2, PY + 3.7, 'activations stay low-bit (A4–A8)', ha='center', va='center', fontsize=5.6, color=INK)
ax.text(PX + PW / 2, PY + 1.6, '* stem is W8 by protocol (first layer), not by P', ha='center', va='center',
        fontsize=4.9, color=MUTED)
arrow([(46.0, 46.0), (49.0, 46.0)])
arrow([(46.0, 23.0), (49.0, 23.0)])

# ---------------------------------------------------------------- 3. 후보 두 개: GP / GPM
ax.plot([88.6, 89.6, 89.6, 88.6], [52.0, 52.0, 23.0, 23.0], color=LINE, lw=0.8)
CX = 91.0
arrow([(89.6, 37.5), (CX, 37.5), (CX, 52.0), (CX + 1.6, 52.0)])
arrow([(CX, 37.5), (CX, 23.0), (CX + 1.6, 23.0)])
ax.text(CX + 11.5, 63.6, 'two candidates', ha='center', va='center', fontsize=6.6, color=INK, weight='bold')
ax.text(CX + 11.5, 60.9, 'same seed · same calibration images', ha='center', va='center', fontsize=5.3, color=MUTED)
# GP
ax.text(CX + 1.6, 57.4, 'GP', fontsize=7, color=INK, weight='bold', va='center')
box(CX + 1.6, 48.5, 9.0, 7.0, 'no M', fc='white', ec='#9a9a9a', fs=5.8, ls=(0, (2, 1.5)), tc=MUTED, lw=0.6)
box(CX + 12.6, 48.5, 10.0, 7.0, 'calibrate\n+ BRECQ', fc=RL, ec=R, fs=6.0)
arrow([(CX + 10.6, 52.0), (CX + 12.6, 52.0)])
# GPM
ax.text(CX + 1.6, 33.8, 'GPM', fontsize=7, color=INK, weight='bold', va='center')
box(CX + 1.6, 16.0, 9.0, 15.0, '', fc=ML, ec=M, lw=0.9)
badge(CX + 3.4, 29.0, 'M', M, r=1.1, fs=6)
ax.text(CX + 6.1, 22.3, 'producer:\n$x \\,/\\, s$\n\nconsumers:\n$W\\,\\mathrm{diag}(s)$', ha='center',
        va='center', fontsize=5.3, color=INK, linespacing=1.05, zorder=5)
box(CX + 12.6, 19.5, 10.0, 7.0, 'calibrate\n+ BRECQ', fc=RL, ec=R, fs=6.0)
arrow([(CX + 10.6, 23.0), (CX + 12.6, 23.0)])
badge(CX + 22.2, 55.0, 'R', R, r=1.0, fs=5.5); badge(CX + 22.2, 26.0, 'R', R, r=1.0, fs=5.5)
ax.text(CX + 17.6, 46.6, 'BRECQ or QDrop', ha='center', va='center', fontsize=4.9, color=R)
ax.text(CX + 6.1, 14.6, 'shared $s$ per producer', ha='center', va='center', fontsize=4.9, color=M)

# ---------------------------------------------------------------- 4. 선택
KX = 116.5
box(KX, 25.5, 14.0, 23.5, '', fc='white', ec=OK, lw=1.0)
ax.text(KX + 7.0, 46.6, 'M check', ha='center', va='center', fontsize=6.8, color=OK, weight='bold', zorder=5)
thumbs(KX + 3.2, 37.0, 5.6, [os.path.join(TRAIN, f) for f in tr[256:259]], z=5)
ax.text(KX + 7.0, 32.6, 'held-out train\nimages (next 200)\n+ COCO vocab.', ha='center', va='center', fontsize=5.0,
        color=MUTED, linespacing=1.05, zorder=5)
ax.text(KX + 7.0, 27.9, 'top-1 flip vs. FP', ha='center', va='center', fontsize=5.4, color=INK, zorder=5)
arrow([(CX + 22.6, 52.0), (KX - 1.4, 52.0), (KX - 1.4, 42.0), (KX, 42.0)])
arrow([(CX + 22.6, 23.0), (KX - 1.4, 23.0), (KX - 1.4, 33.0), (KX, 33.0)])
ax.text(KX + 7.0, 21.6, '$\\mathrm{flip_{GPM}} \\leq \\mathrm{flip_{GP}}$\n→ GPM, else GP', ha='center', va='center',
        fontsize=5.6, color=OK, linespacing=1.15)
box(KX + 15.8, 31.5, 7.0, 12.0, 'Ours\n\nlow-bit\nmodel', fc=GL, ec=G, fs=6.0, lw=1.0)
arrow([(KX + 14.0, 37.5), (KX + 15.8, 37.5)])
ax.text(KX + 19.3, 29.6, 'per seed', ha='center', va='center', fontsize=5.0, color=MUTED)

# 데이터 띠(Y0 아래)는 11-04에 뺐다. 그 높이만큼 아래를 잘라 낸다(단위 크기 20/in 유지)
ax.set_ylim(Y0, 70); fig.set_size_inches(W, H * (70 - Y0) / 70)

for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(FIG, f'method_flow.{ext}'), dpi=300, facecolor='white')
print('저장:', os.path.join(FIG, 'method_flow.png'), '/ .pdf')
