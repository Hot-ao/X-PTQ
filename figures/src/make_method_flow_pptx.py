"""방법 흐름 그림의 편집 가능한 PPT판 (11-04). figures/method_flow.png(make_method_flow.py)와 같은 배치를
PowerPoint 기본 도형(둥근 사각형, 원, 꺾은선 화살표, 글상자)으로 다시 그린다. 패널마다 그룹으로 묶어 두었다.
사용(저장소 루트, python-pptx 필요. .venv에는 없어 scratch에 따로 설치해 썼다):
  PYTHONPATH=<python-pptx 경로> .venv/bin/python figures/src/make_method_flow_pptx.py
좌표는 make_method_flow.py의 그림 단위(가로 140 x 세로 70, y는 아래가 0)를 그대로 쓰고 슬라이드(13.33 x 6.67 in)로 옮긴다.
글자 크기는 그림 크기(7 in) 대비 슬라이드 배율(x1.9)만큼 키운다. 막대는 실측(configs/protect_cache).
출력: figures/method_flow.pptx"""
import json, os, sys
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.oxml.ns import qn
from lxml import etree

SRC = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.dirname(SRC); ROOT = os.path.dirname(FIG)
sys.path.insert(0, SRC)
TMP = os.path.join(FIG, '.pptx_assets'); os.makedirs(TMP, exist_ok=True)

SW, SH = 13.333, 6.667
S = SW / 140.0                       # in per unit
FS = SW / 7.0                        # 글자 배율
FONT = 'Times New Roman'
INK, MUTED, LINE = '1D1D1F', '5F6368', '222222'
MODF, MODE = 'E2F0D9', '6A9F4B'
ORANGE = 'E8894A'
D, G, P, M, R, OK = '8A6D0B', '2C7D4D', '6A3FA0', '1C7688', '59626C', '2A78D6'
DL, GL, PL, ML, RL = 'FBF1D3', 'DFF1E3', 'ECE2F6', 'D8EEF2', 'ECEEF0'
# P 패널: backbone 블록별 conv(데이터 흐름 순서)와 보호 목록. make_method_flow.py와 같은 값
BLOCK_CONVS = [['0'], ['1'], ['2.cv1', '2.m.0.cv1', '2.m.0.cv2', '2.cv2'], ['3'],
               ['4.cv1', '4.m.0.cv1', '4.m.0.cv2', '4.m.1.cv1', '4.m.1.cv2', '4.cv2'], ['5'],
               ['6.cv1', '6.m.0.cv1', '6.m.0.cv2', '6.m.1.cv1', '6.m.1.cv2', '6.cv2'], ['7'],
               ['8.cv1', '8.m.0.cv1', '8.m.0.cv2', '8.cv2'], ['9.cv1', '9.cv2']]
PROTECTED = {'1', '2.cv1', '2.cv2', '2.m.0.cv1', '2.m.0.cv2', '4.cv1', '4.cv2', '4.m.1.cv1', '6.cv1'}
FAINT, STEM, CELL = 'DDDCD6', '8A8A8A', 1.0
LEGEND = ((5.5, 'W8 by P', P), (15.5, 'W4', FAINT), (22.0, 'W8 by protocol*', STEM))


def conv_cells(x, py, n):
    cols = 1 if n == 1 else 2
    x0 = x + (3.0 - cols * CELL - (cols - 1) * 0.3) / 2
    return [(x0 + (k % cols) * (CELL + 0.3), py + 12.9 - (k // cols) * (CELL + 0.3)) for k in range(n)]


def cell_color(c):
    return STEM if c == '0' else (P if c in PROTECTED else FAINT)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)
slide = prs.slides.add_slide(prs.slide_layouts[6])
CUR = [slide.shapes]


def E(u):
    return Emu(int(round(u * S * 914400)))


def X(x):
    return E(x)


def Y(y):                              # 그림 y(아래 0) -> 슬라이드 top. 데이터 띠(Y0 아래)를 뺀 만큼 세로 가운데로
    return E(70 - y + Y0 / 2)


def rgb(h):
    return RGBColor.from_string(h)


def _line(shape, color, w, dash=None):
    if color is None:
        shape.line.fill.background()
        return
    shape.line.color.rgb = rgb(color); shape.line.width = Pt(w * FS * 0.75)
    if dash:
        shape.line.dash_style = MSO_LINE_DASH_STYLE.DASH


def runs_to(tf, lines, fs, color, align, bold=False, italic=False):
    """lines: 줄 목록. 줄은 문자열이거나 (글자, 옵션) 조각 목록. 옵션: i(기울임) b(굵게) sub(아래첨자) c(색)."""
    tf.clear()
    for li, ln in enumerate(lines):
        p = tf.paragraphs[0] if li == 0 else tf.add_paragraph()
        p.alignment = align
        segs = [(ln, {})] if isinstance(ln, str) else ln
        for t, o in segs:
            r = p.add_run(); r.text = t
            f = r.font; f.name = FONT; f.size = Pt(fs * FS); f.color.rgb = rgb(o.get('c', color))
            f.bold = o.get('b', bold); f.italic = o.get('i', italic)
            if o.get('sub'):
                r._r.get_or_add_rPr().set('baseline', '-25000')


def text(x, y, lines, fs=6, color=INK, ha='center', w=None, h=None, bold=False, italic=False, name=None):
    """(x, y)는 글자 기준점(matplotlib의 va=center). ha: left/center/right."""
    lines = lines if isinstance(lines, list) else [lines]
    n = len(lines)
    w = w or max(len(l) if isinstance(l, str) else sum(len(t) for t, _ in l) for l in lines) * fs * 0.16 + 1.5
    h = h or n * fs * 0.21 + 0.6
    x0 = x - w / 2 if ha == 'center' else x if ha == 'left' else x - w
    tb = CUR[0].add_textbox(X(x0), Y(y + h / 2), E(w), E(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for side in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'):
        setattr(tf, side, 0)
    al = {'center': PP_ALIGN.CENTER, 'left': PP_ALIGN.LEFT, 'right': PP_ALIGN.RIGHT}[ha]
    runs_to(tf, lines, fs, color, al, bold, italic)
    if name:
        tb.name = name
    return tb


def box(x, y, w, h, lines=None, fc='FFFFFF', ec=LINE, fs=6.5, lw=0.8, r=0.6, tc=INK, dash=False, bold=False, name=None):
    shp = CUR[0].add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, X(x), Y(y + h), E(w), E(h))
    shp.adjustments[0] = min(0.5, r / min(w, h))
    if fc is None:
        shp.fill.background()
    else:
        shp.fill.solid(); shp.fill.fore_color.rgb = rgb(fc)
    _line(shp, ec, lw, dash)
    shp.shadow.inherit = False
    tf = shp.text_frame
    for side in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'):
        setattr(tf, side, 0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.word_wrap = True
    if lines:
        runs_to(tf, lines if isinstance(lines, list) else [lines], fs, tc, PP_ALIGN.CENTER, bold)
    if name:
        shp.name = name
    return shp


def rect(x, y, w, h, fc, name=None):
    shp = CUR[0].add_shape(MSO_SHAPE.RECTANGLE, X(x), Y(y + h), E(w), E(h))
    shp.fill.solid(); shp.fill.fore_color.rgb = rgb(fc); shp.line.fill.background(); shp.shadow.inherit = False
    if name:
        shp.name = name
    return shp


def circle(x, y, rr, s, fc='FFFFFF', ec=LINE, tc=LINE, fs=6.5, bold=False, lw=0.7):
    shp = CUR[0].add_shape(MSO_SHAPE.OVAL, X(x - rr), Y(y + rr), E(2 * rr), E(2 * rr))
    shp.fill.solid(); shp.fill.fore_color.rgb = rgb(fc); _line(shp, ec, lw); shp.shadow.inherit = False
    tf = shp.text_frame
    for side in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'):
        setattr(tf, side, 0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.word_wrap = False
    runs_to(tf, [s], fs, tc, PP_ALIGN.CENTER, bold)
    return shp


def badge(x, y, s, c, r=1.3, fs=6.8):
    return circle(x, y, r, s, fc=c, ec='FFFFFF', tc='FFFFFF', fs=fs, bold=True, lw=0.8)


def poly(pts, c=LINE, lw=0.8, head=True, name=None):
    """꺾은선(열린 freeform). head면 끝에 삼각 화살촉."""
    sx, sy = X(pts[0][0]), Y(pts[0][1])
    b = CUR[0].build_freeform(sx, sy, scale=1.0)
    b.add_line_segments([(X(px), Y(py)) for px, py in pts[1:]], close=False)
    shp = b.convert_to_shape()
    shp.fill.background(); _line(shp, c, lw); shp.shadow.inherit = False
    if head:
        ln = shp.line._get_or_add_ln()
        te = etree.SubElement(ln, qn('a:tailEnd')); te.set('type', 'triangle'); te.set('w', 'med'); te.set('len', 'med')
    if name:
        shp.name = name
    return shp


def photo(path, x, y, w, name):
    im = Image.open(path).convert('RGB')
    cw, ch = im.size; tw = min(cw, ch * 4 // 3); th = tw * 3 // 4
    im = im.crop(((cw - tw) // 2, (ch - th) // 2, (cw - tw) // 2 + tw, (ch - th) // 2 + th)).resize((400, 300))
    fp = os.path.join(TMP, name + '.png'); im.save(fp)
    h = w * 0.75
    pic = CUR[0].add_picture(fp, X(x), Y(y + h), E(w), E(h))
    pic.line.color.rgb = rgb('9A9A9A'); pic.line.width = Pt(0.5)
    return pic


def thumbs(x, y, w, files, tag):
    for k, f in enumerate(files[::-1]):
        photo(f, x + k * 1.1, y + k * 1.1, w, f'{tag}{k}')


class group:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.g = CUR[0].add_group_shape(); self.g.name = self.name; CUR.append(self.g.shapes)

    def __exit__(self, *a):
        CUR.pop()


def panel(x, y, w, h, c, cl, key, title, sub=''):
    box(x, y, w, h, fc=cl, ec=c, lw=0.9, r=0.8, name=f'{key} panel')
    badge(x + 2.4, y + h - 2.4, key, c)
    text(x + 4.4, y + h - 2.4, title, fs=7.2, color=c, ha='left', bold=True)
    if sub:
        text(x + w - 1.2, y + h - 2.4, sub, fs=5.6, color=MUTED, ha='right', italic=True)


TRAIN = '/data/taeho/coco_datasets/train2017'
tr = sorted(os.listdir(TRAIN))
Y0 = 9.5
sub = lambda t: (t, {'sub': True})
it = lambda t: (t, {'i': True})

# ---------------------------------------------------------------- 0. 입력
with group('Inputs'):
    box(1, 50, 15, 9, ['YOLO-World', '(FP32)'], fc=MODF, ec=MODE, fs=7)
    thumbs(2.5, 33.5, 9.0, [os.path.join(TRAIN, f) for f in tr[:3]], 'calib')
    text(8.5, 30.6, ['calibration images', '(256, COCO vocab.)'], fs=5.7, color=MUTED)
    poly([(16.6, 54.5), (17.6, 54.5), (17.6, 37.5), (16.6, 37.5)], head=False)
    poly([(17.6, 46.0), (20.0, 46.0)])

# ---------------------------------------------------------------- 1. D
with group('D: Leak-free diagnosis'):
    panel(20, Y0 + 1, 26, 68.5 - Y0 - 1, D, DL, 'D', 'Leak-free diagnosis')
    box(22, 55.5, 22, 6.5, ['quantize one conv to W4', '(rest W8A8), measure', 'COCO top-1 flip vs. FP'], ec=D, fs=5.7)
    poly([(33.0, 55.5), (33.0, 49.6)])
    from diag_concentration import metrics
    rows = json.load(open(os.path.join(ROOT, 'configs', 'protect_cache', 'yolov8s-world_n200_c256_img640_la16_aqattn_cls.json')))
    gset = {f'{i}.cv2.conv' for i in (12, 15, 19, 22)}
    psel = metrics(rows, gset)['psel']
    bx, by, bw, bh, cap = 22.5, 32.0, 21.0, 15.0, 6.0
    with group('per-conv damage bars (measured)'):
        poly([(bx - 0.3, by), (bx + bw, by)], c=MUTED, lw=0.5, head=False)
        step = bw / len(rows)
        for i, r in enumerate(rows):
            v = max(r['coco'], 0.0)
            if v <= 0:
                continue
            col = G if r['name'] in gset else P if r['name'] in psel else 'B8B8B8'
            rect(bx + i * step, by, step * 0.8, max(min(v, cap) / cap * bh, 0.05), col, name=r['name'])
            if v > cap:
                text(bx + i * step + step * 0.4, by + bh + 1.0, f'12.cv2: {v:.0f} pp', fs=5.4, color=G)
    text(bx + bw / 2, by - 1.5, 'flip increase per conv (YOLO-World-S)', fs=5.3, color=MUTED)
    rect(22.3, 23.3, 1.4, 1.4, G); text(24.3, 24.0, 'text-gated fusion convs → G', fs=5.8, ha='left')
    text(24.3, 21.7, '(all T-CSPLayer cv2, by structure)', fs=5.2, color=MUTED, ha='left')
    rect(22.3, 16.7, 1.4, 1.4, P); text(24.3, 17.4, 'other top convs → P', fs=5.8, ha='left')
    text(24.3, 15.1, '(greedy by damage, 1.5% budget)', fs=5.2, color=MUTED, ha='left')

# ---------------------------------------------------------------- 2. G
GX, GY, GW, GH = 49, 37.5, 39, 31.0
with group('G: Gate commutation'):
    panel(GX, GY, GW, GH, G, GL, 'G', 'Gate commutation', 'FP-equivalent · +0 bits')
    text(GX + 4.4, GY + GH - 5.0, 'in every T-CSPLayer (12, 15, 19, 22)', fs=5.5, color=G, ha='left')
    yr = {'0': 59.0, '1': 56.0, '2': 53.0}
    for k, yy in yr.items():
        box(GX + 1.5, yy - 1.1, 4.2, 2.2, [[it('y'), sub(k)]], fs=6.2, lw=0.6, r=0.3)
        poly([(GX + 5.7, yy), (GX + 8.0, yy)])
    box(GX + 8.0, 51.6, 1.8, 8.8, ['C'], fc='E9EFF8', ec='3F5F96', fs=5.8, lw=0.6, r=0.3)
    box(GX + 12.0, 53.5, 10.5, 5.0, [[('cv2', {}), sub('main'), (' 1×1', {})], 'quantizer A'], ec=G, fs=5.6, lw=1.0, r=0.4)
    poly([(GX + 9.8, 56.0), (GX + 12.0, 56.0)])
    circle(GX + 30.5, 56.0, 0.9, '+', fs=7)
    poly([(GX + 22.5, 56.0), (GX + 29.6, 56.0)])
    box(GX + 32.6, 54.4, 5.2, 3.2, ['SiLU'], fs=5.6, lw=0.6, r=0.3)
    poly([(GX + 31.4, 56.0), (GX + 32.6, 56.0)])
    poly([(GX + 5.7, 53.0), (GX + 6.6, 53.0), (GX + 6.6, 42.6), (GX + 17.5, 42.6)])
    poly([(GX + 6.6, 47.5), (GX + 8.0, 47.5)])
    box(GX + 8.0, 45.5, 7.0, 4.0, ['proj', '3×3'], fs=5.4, lw=0.6, r=0.3)
    text(GX + 16.2, 48.5, [[it('p')]], fs=6, color=MUTED)
    box(GX + 17.5, 45.0, 10.5, 5.0, [[('cv2', {}), sub('side'), (' 1×1', {})], 'quantizer B'], ec=G, fs=5.6, lw=1.0, r=0.4)
    poly([(GX + 15.0, 47.5), (GX + 17.5, 47.5)])
    circle(GX + 30.5, 47.5, 0.9, '⊙', ec=ORANGE, tc=ORANGE, fs=6.5)
    poly([(GX + 28.0, 47.5), (GX + 29.6, 47.5)])
    poly([(GX + 30.5, 48.4), (GX + 30.5, 55.1)], c=ORANGE)
    box(GX + 17.5, 41.4, 10.5, 2.6, [[('gate ', {}), it('a'), ('(', {}), it('y'), sub('2'), (', text)', {})]],
        fc=MODF, ec=MODE, fs=5.4, lw=0.6, r=0.3)
    poly([(GX + 28.0, 42.6), (GX + 30.5, 42.6), (GX + 30.5, 46.6)], c=ORANGE)
    poly([(GX + 22.75, 38.4), (GX + 22.75, 41.4)], c=ORANGE)
    text(GX + 23.6, 38.9, 'text', fs=5.4, color=ORANGE, ha='left')
    text(GX + 34.2, 50.5, ['gate after', 'the conv'], fs=5.3, color=G)

# ---------------------------------------------------------------- 2. P
PX, PY, PW, PH = 49, Y0 + 1, 39, 25.0
with group('P: Protection'):
    panel(PX, PY, PW, PH, P, PL, 'P', 'Protection')
    text(PX + 4.4, PY + PH - 5.0, 'YOLO Backbone (blocks 0–9)', fs=5.5, color=P, ha='left')
    for i, convs in enumerate(BLOCK_CONVS):         # 블록 하나 = 그룹(테두리 + 번호 + conv 사각형)
        x = PX + 1.6 + i * 3.6
        prot = any(c in PROTECTED for c in convs)
        with group(f'block {i}'):
            box(x, PY + 9.0, 3.0, 7.6, fc='FFFFFF', ec=P if prot else '9A9A9A', lw=0.6, r=0.3, name=f'block {i} frame')
            text(x + 1.5, PY + 15.3, str(i), fs=5.6, color=P if prot else INK)
            for (cx, cy), c in zip(conv_cells(x, PY, len(convs)), convs):
                rect(cx, cy, CELL, CELL, cell_color(c), name=f'{c} ({"W8 protocol" if c == "0" else "W8 P" if c in PROTECTED else "W4"})')
    for dx, lab, c in LEGEND:
        rect(PX + dx, PY + 6.0, CELL, CELL, c, name=f'legend {lab}')
        text(PX + dx + CELL + 0.6, PY + 6.0 + CELL / 2, lab, fs=5.0, ha='left')
    text(PX + PW / 2, PY + 3.7, 'activations stay low-bit (A4–A8)', fs=5.6)
    text(PX + PW / 2, PY + 1.6, '* stem is W8 by protocol (first layer), not by P', fs=4.9, color=MUTED)
poly([(46.0, 46.0), (49.0, 46.0)])
poly([(46.0, 23.0), (49.0, 23.0)])

# ---------------------------------------------------------------- 3. 후보 두 개
CX = 91.0
with group('Two candidates: GP / GPM'):
    poly([(88.6, 52.0), (89.6, 52.0), (89.6, 23.0), (88.6, 23.0)], head=False)
    poly([(89.6, 37.5), (CX, 37.5), (CX, 52.0), (CX + 1.6, 52.0)])
    poly([(CX, 37.5), (CX, 23.0), (CX + 1.6, 23.0)])
    text(CX + 11.5, 63.6, 'two candidates', fs=6.6, bold=True)
    text(CX + 11.5, 60.9, 'same seed · same calibration images', fs=5.3, color=MUTED)
    text(CX + 1.6, 57.4, 'GP', fs=7, bold=True, ha='left')
    box(CX + 1.6, 48.5, 9.0, 7.0, ['no M'], ec='9A9A9A', fs=5.8, tc=MUTED, lw=0.6, dash=True)
    box(CX + 12.6, 48.5, 10.0, 7.0, ['calibrate', '+ BRECQ'], fc=RL, ec=R, fs=6.0)
    poly([(CX + 10.6, 52.0), (CX + 12.6, 52.0)])
    text(CX + 1.6, 33.8, 'GPM', fs=7, bold=True, ha='left')
    box(CX + 1.6, 16.0, 9.0, 15.0, ['producer:', [it('x'), (' / ', {}), it('s')], '', 'consumers:',
                                    [it('W'), (' diag(', {}), it('s'), (')', {})]], fc=ML, ec=M, fs=5.3, lw=0.9)
    badge(CX + 3.4, 29.0, 'M', M, r=1.1, fs=6)
    box(CX + 12.6, 19.5, 10.0, 7.0, ['calibrate', '+ BRECQ'], fc=RL, ec=R, fs=6.0)
    poly([(CX + 10.6, 23.0), (CX + 12.6, 23.0)])
    badge(CX + 22.2, 55.0, 'R', R, r=1.0, fs=5.5); badge(CX + 22.2, 26.0, 'R', R, r=1.0, fs=5.5)
    text(CX + 17.6, 46.6, 'BRECQ or QDrop', fs=4.9, color=R)
    text(CX + 6.1, 14.6, [[('shared ', {}), it('s'), (' per producer', {})]], fs=4.9, color=M)

# ---------------------------------------------------------------- 4. 선택
KX = 116.5
with group('M check and output'):
    box(KX, 25.5, 14.0, 23.5, None, ec=OK, lw=1.0)
    text(KX + 7.0, 46.6, 'M check', fs=6.8, color=OK, bold=True)
    thumbs(KX + 3.2, 37.0, 5.6, [os.path.join(TRAIN, f) for f in tr[256:259]], 'check')
    text(KX + 7.0, 32.6, ['held-out train', 'images (next 200)', '+ COCO vocab.'], fs=5.0, color=MUTED)
    text(KX + 7.0, 27.9, 'top-1 flip vs. FP', fs=5.4)
    poly([(CX + 22.6, 52.0), (KX - 1.4, 52.0), (KX - 1.4, 42.0), (KX, 42.0)])
    poly([(CX + 22.6, 23.0), (KX - 1.4, 23.0), (KX - 1.4, 33.0), (KX, 33.0)])
    text(KX + 7.0, 21.6, [[('flip', {}), sub('GPM'), (' ≤ flip', {}), sub('GP')], '→ GPM, else GP'], fs=5.6, color=OK)
    box(KX + 15.8, 31.5, 7.0, 12.0, ['Ours', '', 'low-bit', 'model'], fc=GL, ec=G, fs=6.0, lw=1.0)
    poly([(KX + 14.0, 37.5), (KX + 15.8, 37.5)])
    text(KX + 19.3, 29.6, 'per seed', fs=5.0, color=MUTED)

out = os.path.join(FIG, 'method_flow.pptx')
prs.save(out)
print('저장:', out)
