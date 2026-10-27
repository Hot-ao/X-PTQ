"""방법 개요 그림 (논문 Fig. 2 초안). 사용: python figures/src/make_method_overview.py  (어디서 실행해도 된다)
(a) YOLO-World에서 G/P/M이 작용하는 위치, (b) C2fAttn의 게이트 교환, (c) 공유 제약 scale 이전.
구조와 수치의 근거는 docs/METHOD.md §2.2(분기별 activation 최대), §4.2, §4.4다. 데이터 파일은 읽지 않는다.
출력: figures/method_overview.png, method_overview.pdf"""
import os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 10-27: 코드는 figures/src/, 그림은 figures/
plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42})

INK, MUTED, LINE = '#22262b', '#5d6670', '#3a4048'
IMG = ('#e9eff8', '#3f5f96')      # 이미지 특징 경로 (면, 선)
TXT = ('#fdebd8', '#c2661c')      # 텍스트 경로
G = ('#dff1e3', '#2c7d4d')        # 게이트 교환
P = ('#ece2f6', '#6a3fa0')        # 보호
M = ('#d8eef2', '#1c7688')        # 이전
R = ('#eceef0', '#59626c')        # 재구성
BAD = '#c0392b'

fig = plt.figure(figsize=(14, 7.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 140); ax.set_ylim(0, 74); ax.axis('off')


def box(x, y, w, h, text='', c=IMG, fs=7.5, lw=1.0, r=0.6, tc=INK, weight='normal', ls='-', z=3, fc=None):
    ax.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f'round,pad={r}', fc=fc or c[0], ec=c[1],
                                lw=lw, ls=ls, zorder=z))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc, weight=weight, zorder=z + 1,
                linespacing=1.15)


def arrow(pts, c=LINE, lw=1.0, head=True, z=2, ls='-'):
    """pts를 잇는 꺾은선. 마지막 구간에 화살촉."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=lw, ls=ls, zorder=z, solid_capstyle='round')
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle='-|>' if head else '-', mutation_scale=8, color=c, lw=lw,
                                 ls=ls, shrinkA=0, shrinkB=0, zorder=z))


def op(x, y, sym, c=LINE, r=1.25):
    ax.add_patch(Circle((x, y), r, fc='white', ec=c, lw=1.0, zorder=4))
    ax.text(x, y, sym, ha='center', va='center', fontsize=9, color=c, zorder=5)


def badge(x, y, s, c, r=1.05, fs=6.5):
    ax.add_patch(Circle((x, y), r, fc=c[1], ec='white', lw=0.8, zorder=6))
    ax.text(x, y, s, ha='center', va='center', fontsize=fs, color='white', weight='bold', zorder=7)


def title(x, y, s):
    ax.text(x, y, s, ha='left', va='center', fontsize=9.5, weight='bold', color=INK)


# ---------------------------------------------------------------- 단계 띠 (빌드 순서)
STEPS = [('G', 'Gate commutation', 'FP-equivalent rewrite of C2fAttn · +0 bits', G),
         ('P', 'Protection', 'leak-free sensitivity → 3 convs keep W8 · +0.3% size', P),
         ('M', 'Tied scale migration', 'shared per-channel $s$ · kept only if pre-check passes', M),
         ('R', 'Block reconstruction', 'BRECQ / QDrop, unchanged', R)]
for i, (k, name, sub, c) in enumerate(STEPS):
    x = 2.5 + i * 34.4
    box(x, 67.2, 31.8, 5.6, c=c, lw=1.1, r=0.9)
    badge(x + 3.0, 70.0, k, c, r=1.7, fs=9)
    ax.text(x + 5.6, 71.0, name, ha='left', va='center', fontsize=8.5, weight='bold', color=INK, zorder=5)
    ax.text(x + 5.6, 68.75, sub, ha='left', va='center', fontsize=6.3, color=MUTED, zorder=5)
    if i < 3:
        arrow([(x + 32.0, 70.0), (x + 34.2, 70.0)], lw=1.2)

# ---------------------------------------------------------------- (a) 전체 구조
title(2.5, 64.3, '(a) Where each step acts in YOLO-World')
box(2.5, 50.0, 6.5, 8.0, 'Image', fs=7.5)
box(12.0, 47.6, 54.0, 13.2, c=('#f6f8fb', '#9aa7b8'), lw=0.9, z=1)
ax.text(13.2, 59.4, 'Backbone', fontsize=7.5, weight='bold', color=MUTED, va='center')
names = ['Conv', 'Conv', 'C2f', 'Conv', 'C2f', 'Conv', 'C2f', 'Conv', 'C2f', 'SPPF']
for i, n in enumerate(names):
    x = 13.2 + i * 5.2
    prot = i in (1, 2, 4)
    box(x, 50.0, 4.4, 7.4, f'{n}\n{i}', c=P if prot else IMG, fs=6.6, lw=1.4 if prot else 0.9)
    if prot:
        badge(x + 4.0, 57.2, 'P', P)
    if i < 9:
        arrow([(x + 4.4, 53.7), (x + 5.2, 53.7)], lw=0.8, head=False)
ax.text(24.2, 48.7, 'W8 kept for convs 1, 2.cv2, 4.cv1', fontsize=6.2, color=P[1], ha='center', va='center', style='italic')

box(69.0, 47.6, 38.0, 13.2, c=('#f6f8fb', '#9aa7b8'), lw=0.9, z=1)
ax.text(70.2, 59.4, 'Neck (text–image fusion)', fontsize=7.5, weight='bold', color=MUTED, va='center')
neck = [('C2fAttn', 12), ('C2fAttn', 15), ('ImgPool\nAttn', 16), ('C2fAttn', 19), ('C2fAttn', 22)]
attn_x = []
for i, (n, k) in enumerate(neck):
    x = 70.2 + i * 7.3
    g = n == 'C2fAttn'
    box(x, 50.0, 6.4, 7.4, f'{n}\n{k}', c=G if g else IMG, fs=6.3, lw=1.4 if g else 0.9)
    if g:
        badge(x + 6.0, 57.2, 'G', G); attn_x.append(x + 3.2)
    if i < 4:
        arrow([(x + 6.4, 53.7), (x + 7.3, 53.7)], lw=0.8, head=False)

box(110.0, 47.6, 21.0, 13.2, c=('#f6f8fb', '#9aa7b8'), lw=0.9, z=1)
ax.text(111.2, 59.4, 'Head', fontsize=7.5, weight='bold', color=MUTED, va='center')
box(111.2, 50.0, 8.4, 7.4, 'Box / embed\nconvs', fs=6.3, lw=0.9)
box(120.6, 50.0, 9.2, 7.4, 'Contrastive\n' + r'$\tau\cos(x, t_j)$', c=TXT, fs=6.3, lw=0.9)
arrow([(119.6, 53.7), (120.6, 53.7)], lw=0.8, head=False)
arrow([(9.0, 54.0), (12.0, 54.0)], lw=1.2); arrow([(66.0, 54.0), (69.0, 54.0)], lw=1.2)
arrow([(107.0, 54.0), (110.0, 54.0)], lw=1.2); arrow([(131.0, 54.0), (133.6, 54.0)], lw=1.2)
ax.text(134.1, 54.0, 'boxes,\nclass\nscores', fontsize=6.8, color=INK, va='center', ha='left', linespacing=1.1)

# M: 생산 conv -> 소비 conv 쌍 전체
ax.plot([12.0, 12.0, 131.0, 131.0], [61.4, 62.2, 62.2, 61.4], color=M[1], lw=1.1)
badge(14.0, 62.2, 'M', M)
ax.text(16.0, 62.25, 'scale moved from producer convs to their consumer convs', fontsize=6.4, color=M[1],
        va='center', ha='left', bbox=dict(fc='white', ec='none', pad=0.8))

# 텍스트 경로
box(12.0, 39.8, 15.5, 5.0, 'Text prompts\n“person”, “dog”, …', c=TXT, fs=6.6)
box(31.0, 39.8, 17.5, 5.0, 'CLIP text encoder\n(full precision, offline)', c=TXT, fs=6.6, ls=(0, (3, 1.5)))
arrow([(27.5, 42.3), (31.0, 42.3)], c=TXT[1], lw=1.1)
ax.plot([48.5, 125.2], [42.3, 42.3], color=TXT[1], lw=1.1, zorder=2)
for x in attn_x + [125.2]:
    arrow([(x, 42.3), (x, 50.0)], c=TXT[1], lw=1.1, z=2)
ax.text(57.5, 43.5, r'text embeddings $t_j$', fontsize=6.8, color=TXT[1], ha='center', va='center')
ax.text(88.0, 41.0, 'guide', fontsize=6.2, color=TXT[1], ha='center', va='center', style='italic')
ax.text(118.0, 41.0, 'classifier', fontsize=6.2, color=TXT[1], ha='center', va='center', style='italic')

ax.plot([2.5, 137.5], [37.4, 37.4], color='#c9cfd6', lw=0.8)
ax.plot([95.2, 95.2], [2.0, 36.0], color='#c9cfd6', lw=0.8)

# ---------------------------------------------------------------- (b) 게이트 교환
title(2.5, 34.8, '(b) Gate commutation in C2fAttn')
ROW = {'y0': 28.4, 'y1': 24.9, 'y2': 21.4, 'p': 15.0}


def stubs(x0):
    for k, lab in (('y0', '$y_0$'), ('y1', '$y_1$'), ('y2', '$y_2$')):
        box(x0, ROW[k] - 1.3, 5.0, 2.6, lab, fs=7.5, lw=0.9, r=0.4)
    box(x0, ROW['p'] - 1.3, 5.0, 2.6, '$p$', c=TXT, fs=7.5, lw=0.9, r=0.4)
    ax.text(x0 + 2.5, ROW['p'] - 2.4, r'$=\mathrm{proj}(y_2)$', fontsize=5.8, color=MUTED, ha='center', va='center')


def gate(x0):
    box(x0, 6.6, 11.0, 3.6, 'text gate $a_w$\n' + r'$\sigma(\max_j\langle y_2, g_j\rangle)$', c=TXT, fs=5.9, lw=0.9, r=0.4)


# 원래 구조
ax.text(3.0, 31.9, 'Original', fontsize=8, weight='bold', color=BAD, va='center')
stubs(3.5)
box(21.5, 13.0, 2.6, 17.2, c=IMG, lw=0.9, r=0.4)
ax.text(22.8, 21.6, 'Concat', rotation=90, fontsize=6.6, ha='center', va='center', color=INK, zorder=5)
for k in ('y0', 'y1', 'y2'):
    arrow([(8.5, ROW[k]), (21.5, ROW[k])], lw=0.9)
op(14.5, ROW['p'], r'$\odot$', c=TXT[1]); gate(9.0)
arrow([(8.5, ROW['p']), (13.25, ROW['p'])], c=TXT[1], lw=0.9)
arrow([(14.5, 10.2), (14.5, ROW['p'] - 1.25)], c=TXT[1], lw=0.9)
arrow([(15.75, ROW['p']), (21.5, ROW['p'])], c=TXT[1], lw=0.9)
ax.text(18.6, ROW['p'] + 1.1, '$y_3$', fontsize=7.5, color=TXT[1], ha='center', va='center')
box(27.6, 19.0, 9.4, 5.2, 'cv2  1×1\none quantizer', c=('#fbe9e7', BAD), fs=6.6, lw=1.3)
arrow([(24.1, 21.6), (27.6, 21.6)], lw=0.9)
box(39.4, 20.0, 5.0, 3.2, 'SiLU', fs=6.6, lw=0.9, r=0.4)
arrow([(37.0, 21.6), (39.4, 21.6)], lw=0.9); arrow([(44.4, 21.6), (46.4, 21.6)], lw=0.9)
# 분기별 입력 범위 (METHOD §2.2: activation 최대 7.7 / 7.0 / 9.3 / 20.4)
bx, by, sc = 28.6, 6.6, 0.33
ax.plot([bx - 0.8, bx + 9.0], [by, by], color=MUTED, lw=0.7)
for i, (lab, v) in enumerate((('$y_0$', 7.7), ('$y_1$', 7.0), ('$y_2$', 9.3), ('$y_3$', 20.4))):
    c = BAD if i == 3 else IMG[1]
    ax.add_patch(Rectangle((bx + i * 2.2, by), 1.5, v * sc, fc=c, ec='none', alpha=0.9 if i == 3 else 0.55, zorder=3))
    ax.text(bx + i * 2.2 + 0.75, by - 1.0, lab, fontsize=6.2, ha='center', va='center', color=c)
    ax.text(bx + i * 2.2 + 0.75, by + v * sc + 0.75, f'{v:.1f}', fontsize=5.8, ha='center', va='center', color=c)
ax.text(38.4, 12.2, 'max |x| per branch\n(block 12): the gated\ntext branch sets the\nshared range', fontsize=5.9, color=BAD,
        ha='left', va='center', linespacing=1.2)
arrow([(32.3, 14.6), (32.3, 19.0)], c=BAD, lw=0.8, ls=(0, (2, 1.5)))

# 바꾼 뒤
ax.text(49.5, 31.9, 'After gate commutation (ours)', fontsize=8, weight='bold', color=G[1], va='center')
arrow([(45.6, 31.9), (48.6, 31.9)], c=MUTED, lw=1.0)
X = 50.0
stubs(X)
box(X + 9.5, 19.6, 2.6, 10.6, c=IMG, lw=0.9, r=0.4)
ax.text(X + 10.8, 24.9, 'Concat', rotation=90, fontsize=6.6, ha='center', va='center', color=INK, zorder=5)
for k in ('y0', 'y1', 'y2'):
    arrow([(X + 5.0, ROW[k]), (X + 9.5, ROW[k])], lw=0.9)
box(X + 15.0, 22.3, 10.6, 5.2, 'cv2_main  1×1\nquantizer A', c=G, fs=6.6, lw=1.3)
arrow([(X + 12.1, 24.9), (X + 15.0, 24.9)], lw=0.9)
box(X + 9.5, 12.4, 10.6, 5.2, 'cv2_side  1×1\nquantizer B', c=G, fs=6.6, lw=1.3)
arrow([(X + 5.0, ROW['p']), (X + 9.5, ROW['p'])], c=TXT[1], lw=0.9)
op(X + 25.0, ROW['p'], r'$\odot$', c=TXT[1]); gate(X + 19.5)
arrow([(X + 20.1, ROW['p']), (X + 23.75, ROW['p'])], c=TXT[1], lw=0.9)
arrow([(X + 25.0, 10.2), (X + 25.0, ROW['p'] - 1.25)], c=TXT[1], lw=0.9)
op(X + 31.0, 24.9, r'$\oplus$')
arrow([(X + 25.6, 24.9), (X + 29.75, 24.9)], lw=0.9)
arrow([(X + 26.25, ROW['p']), (X + 31.0, ROW['p']), (X + 31.0, 23.65)], c=TXT[1], lw=0.9)
box(X + 34.6, 23.3, 5.0, 3.2, 'SiLU', fs=6.6, lw=0.9, r=0.4)
arrow([(X + 32.25, 24.9), (X + 34.6, 24.9)], lw=0.9); arrow([(X + 39.6, 24.9), (X + 41.6, 24.9)], lw=0.9)
ax.text(X + 33.0, 17.6, 'gate applied\nafter the conv:\nun-gated $p$\nis quantized', fontsize=5.9, color=G[1], ha='left',
        va='center', linespacing=1.2)
ax.text(X + 20.3, 20.4, 'text branch no longer shares\na range with $y_0$–$y_2$', fontsize=5.9, color=G[1], ha='center',
        va='center', linespacing=1.2)
ax.text(47.5, 3.0, r'$\mathrm{cv2}\,(\,[\,y_0, y_1, y_2,\; p \odot a_w\,]\,)\;=\;W_{\mathrm{main}}\,[\,y_0, y_1, y_2\,] + b'
        r'\;+\;\sum_h a_{w,h} \odot (W_h\, p_h)$      exact in full precision',
        fontsize=7.6, color=INK, ha='center', va='center')

# ---------------------------------------------------------------- (c) 공유 제약 scale 이전
title(97.5, 34.8, '(c) Tied scale migration')
box(98.0, 23.2, 10.5, 5.2, 'Producer\nconv', fs=6.8, lw=0.9)
box(111.6, 23.9, 5.4, 3.8, r'$\div\, s$', c=M, fs=8, lw=1.3, r=0.5)
arrow([(108.5, 25.8), (111.6, 25.8)], lw=0.9)
box(123.0, 27.4, 14.5, 5.0, 'Consumer A\n' + r'$W_A\,\mathrm{diag}(s)$', c=M, fs=6.6, lw=1.1)
box(123.0, 19.2, 14.5, 5.0, 'Consumer B\n' + r'$W_B\,\mathrm{diag}(s)$', c=M, fs=6.6, lw=1.1)
arrow([(117.0, 25.8), (119.6, 25.8), (119.6, 29.9), (123.0, 29.9)], lw=0.9)
arrow([(119.6, 25.8), (119.6, 21.7), (123.0, 21.7)], lw=0.9)
ax.text(130.2, 25.8, 'same $s$', fontsize=6.4, color=M[1], ha='center', va='center', style='italic')
ax.text(114.3, 21.9, 'folded into the\nproducer’s requant', fontsize=5.7, color=MUTED, ha='center', va='center',
        linespacing=1.2)
ax.text(117.5, 16.0, r'$\mathrm{conv}(x,\,W)\;=\;\mathrm{conv}(x/s,\;W\,\mathrm{diag}(s))$', fontsize=7.6, color=INK,
        ha='center', va='center')
# 채널별 범위: 이전 전(들쭉날쭉) -> 후(평평). 개념도이며 실측값이 아니다.
before = [1.2, 4.6, 0.9, 2.0, 5.6, 1.5, 0.7, 3.0]
after = [3.6, 4.4, 3.3, 3.9, 4.6, 3.7, 3.2, 4.1]
for x0, vals, lab in ((99.0, before, 'channel ranges of $x$'), (121.5, after, 'after $x/s$')):
    ax.plot([x0 - 0.6, x0 + 14.2], [5.4, 5.4], color=MUTED, lw=0.7)
    for q in range(1, 5):
        ax.plot([x0 - 0.6, x0 + 14.2], [5.4 + q * 1.5, 5.4 + q * 1.5], color='#c9cfd6', lw=0.5, ls=(0, (2, 2)), zorder=1)
    for i, v in enumerate(vals):
        ax.add_patch(Rectangle((x0 + i * 1.75, 5.4), 1.2, v, fc=M[1], ec='none', alpha=0.7, zorder=3))
    ax.text(x0 + 6.8, 13.4, lab, fontsize=6.2, color=MUTED, ha='center', va='center')
arrow([(114.6, 8.4), (119.6, 8.4)], c=M[1], lw=1.2)
ax.text(117.5, 3.0, 'one per-tensor activation grid (dashed) is used more evenly', fontsize=5.9, color=MUTED,
        ha='center', va='center')

for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(HERE, f'method_overview.{ext}'), dpi=220, facecolor='white')
print('저장:', os.path.join(HERE, 'method_overview.png'), '/ .pdf')
