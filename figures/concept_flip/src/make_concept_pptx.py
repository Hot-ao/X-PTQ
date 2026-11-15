"""개념 그림 A/B/C의 편집 가능한 PPT판 (11-13). make_concept.py와 같은 좌표·같은 실측 점수를
PowerPoint 기본 도형(다각형 칸, 원, 삼각형, 화살표, 막대, 글상자)으로 다시 그린다. 패널마다 그룹으로 묶는다.
슬라이드 1 = A, 2 = B, 3 = C. 캡션 초안은 슬라이드 노트.
사용(저장소 루트, python-pptx 필요. .venv에는 없어 scratch에 따로 설치해 썼다):
  PYTHONPATH=<python-pptx 경로> .venv/bin/python figures/concept_flip/src/make_concept_pptx.py
출력: figures/concept_flip/concept_flip.pptx
"""
import json, os, sys
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.oxml.ns import qn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib
matplotlib.use("Agg")
from make_concept import FEAT, E_DOM, E_REST, E, SMALL, LARGE, LIM, OUT

FONT = "Arial"
C = lambda h: RGBColor.from_string(h.lstrip("#").upper())
INK, MUTED, AXIS, GRID = "0B0B0B", "6B6A64", "C3C2B7", "E1E0D9"
BLUE, RED, LBLUE = "2A78D6", "D03B3B", "9FB9DC"
CELL_OK, CELL_BAD, CELL = "DBE8F8", "F6D9D6", "F3F2EE"
EMU = 914400

CAPTIONS = {
    "A": ("Figure X. Why low-bit PTQ flips decisions on an open vocabulary (illustration). Each cell is the region whose "
          "top-1 class is the nearest text embedding. (a) With a calibration-size vocabulary, a quantization error e leaves "
          "the region feature in the helmet cell. (b) On the LVIS vocabulary, semantically close classes (cap, hat, visor) "
          "shrink the cell, and an error of the same norm crosses into cap: the top-1 flips although the output error is "
          "unchanged. (c) The error is dominated by one layer l*, the one with the largest flip increase Delta_l (Sec. 3.2). "
          "Removing that component with G or P keeps the decision. Coordinates are illustrative."),
    "B": ("Figure X. A measured flip. Class scores at the region where FP32 predicts helmet (YOLO-World-S, W4A8, 1,203 "
          "LVIS classes, LVIS minival image 450559). BRECQ lowers helmet (0.87 -> 0.64) and raises a group of similar "
          "classes (cap 0.08 -> 0.65, hat, baseball cap), so cap overtakes helmet by 0.007. With our method the top-1 "
          "stays helmet (0.88)."),
    "C": ("Figure X. Same error, different decision. (a, b) Illustration: an error of the same norm keeps the top-1 on a "
          "calibration-size vocabulary but flips it on the LVIS vocabulary, where similar classes crowd the decision cell. "
          "(c) Measured at a helmet region (YOLO-World-S, W4A8): BRECQ raises a group of similar classes until cap "
          "overtakes helmet, while our method keeps the top-1. Minimizing output error alone does not guarantee the "
          "decision; we locate the layers whose error flips decisions (Sec. 3.2) and treat them (Sec. 4)."),
}


# ---------- 기본 도형
def text(shapes, x, y, w, h, runs, size=11, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, bold=False):
    """runs: str 또는 [(글자, {'i':1,'sub':1,'sup':1,'color':hex,'b':1}), ...]"""
    tb = shapes.add_textbox(Emu(int(x)), Emu(int(y)), Emu(int(w)), Emu(int(h)))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]; p.alignment = align
    if isinstance(runs, str):
        runs = [(runs, {})]
    for t, o in runs:
        r = p.add_run(); r.text = t
        f = r.font; f.name = FONT; f.size = Pt(size); f.bold = bool(o.get("b", bold)); f.italic = bool(o.get("i"))
        f.color.rgb = C(o.get("color", color))
        if o.get("sub"): r._r.get_or_add_rPr().set("baseline", "-25000")
        if o.get("sup"): r._r.get_or_add_rPr().set("baseline", "30000")
    return tb


def poly(shapes, pts, fill=None, line=None, lw=0.75):
    pts = [(int(x), int(y)) for x, y in pts]
    fb = shapes.build_freeform(pts[0][0], pts[0][1], scale=1.0)
    fb.add_line_segments(pts[1:], close=True)
    sh = fb.convert_to_shape()
    if fill: sh.fill.solid(); sh.fill.fore_color.rgb = C(fill)
    else: sh.fill.background()
    if line: sh.line.color.rgb = C(line); sh.line.width = Pt(lw)
    else: sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def shape(shapes, kind, cx, cy, w, h, fill=None, line=None, lw=1.0, dash=None):
    sh = shapes.add_shape(kind, Emu(int(cx - w / 2)), Emu(int(cy - h / 2)), Emu(int(w)), Emu(int(h)))
    if fill: sh.fill.solid(); sh.fill.fore_color.rgb = C(fill)
    else: sh.fill.background()
    if line:
        sh.line.color.rgb = C(line); sh.line.width = Pt(lw)
        if dash: sh.line.dash_style = dash
    else: sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def arrow(shapes, p, q, color, lw=1.5, dash=None):
    cn = shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Emu(int(p[0])), Emu(int(p[1])), Emu(int(q[0])), Emu(int(q[1])))
    cn.line.color.rgb = C(color); cn.line.width = Pt(lw)
    if dash: cn.line.dash_style = dash
    ln = cn.line._get_or_add_ln()
    te = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(te)
    return cn


def rect(shapes, x, y, w, h, fill=None, line=None, lw=0.75):
    sh = shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(int(x)), Emu(int(y)), Emu(int(w)), Emu(int(h)))
    if fill: sh.fill.solid(); sh.fill.fore_color.rgb = C(fill)
    else: sh.fill.background()
    if line: sh.line.color.rgb = C(line); sh.line.width = Pt(lw)
    else: sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


# ---------- Voronoi 칸(상자 다각형을 이등분선 반평면으로 잘라 낸다)
def clip(polyg, a, b):
    """polyg를 {x : |x-a| <= |x-b|} 쪽으로 자른다."""
    n = b - a; c = (np.dot(b, b) - np.dot(a, a)) / 2           # n.x <= c
    out = []
    for i in range(len(polyg)):
        P, Q = polyg[i], polyg[(i + 1) % len(polyg)]
        inP, inQ = np.dot(n, P) <= c, np.dot(n, Q) <= c
        if inP: out.append(P)
        if inP != inQ:
            t = (c - np.dot(n, P)) / np.dot(n, Q - P)
            out.append(P + t * (Q - P))
    return out


def voronoi(anchors):
    box = [np.array(v, float) for v in ((LIM[0], LIM[2]), (LIM[1], LIM[2]), (LIM[1], LIM[3]), (LIM[0], LIM[3]))]
    cells = {}
    for n, a in anchors.items():
        pg = box
        for m, b in anchors.items():
            if m != n:
                pg = clip(pg, np.array(a, float), np.array(b, float))
                if not pg: break
        cells[n] = pg
    return cells


# ---------- 결정 경계 패널
class Panel:
    def __init__(self, shapes, x, y, w):
        self.x, self.y, self.w = x, y, w
        self.h = w * (LIM[3] - LIM[2]) / (LIM[1] - LIM[0])
        self.s = shapes

    def P(self, v):
        return (self.x + (v[0] - LIM[0]) / (LIM[1] - LIM[0]) * self.w,
                self.y + (LIM[3] - v[1]) / (LIM[3] - LIM[2]) * self.h)

    def cells(self, anchors, ok="helmet", bad=None):
        for n, pg in voronoi(anchors).items():
            fill = CELL_OK if n == ok else (CELL_BAD if n == bad else CELL)
            poly(self.s, [self.P(v) for v in pg], fill=fill, line=AXIS, lw=0.75)
        rect(self.s, self.x, self.y, self.w, self.h, line=AXIS, lw=0.75)
        for n, v in anchors.items():
            if LIM[0] < v[0] < LIM[1] and LIM[2] < v[1] < LIM[3]:
                cx, cy = self.P(v)
                shape(self.s, MSO_SHAPE.ISOSCELES_TRIANGLE, cx, cy, 0.09 * EMU, 0.08 * EMU, fill=MUTED)
                text(self.s, cx - 0.7 * EMU, cy + 0.07 * EMU, 1.4 * EMU, 0.2 * EMU, n, size=10,
                     color=INK if n in (ok, bad) else MUTED, align=PP_ALIGN.CENTER, bold=n in (ok, bad))

    def dot(self, v, color=INK, hollow=False):
        cx, cy = self.P(v)
        if hollow:
            shape(self.s, MSO_SHAPE.OVAL, cx, cy, 0.11 * EMU, 0.11 * EMU, fill="FFFFFF", line=color, lw=1.75)
        else:
            shape(self.s, MSO_SHAPE.OVAL, cx, cy, 0.11 * EMU, 0.11 * EMU, fill=color)

    def ball(self, r):
        cx, cy = self.P(FEAT)
        rw = r / (LIM[1] - LIM[0]) * self.w
        shape(self.s, MSO_SHAPE.OVAL, cx, cy, 2 * rw, 2 * rw, line=MUTED, lw=1.0, dash=MSO_LINE_DASH_STYLE.DASH)

    def arrow(self, a, b, color, lw=1.6, dash=None):
        arrow(self.s, self.P(a), self.P(b), color, lw, dash)

    def title(self, t):
        text(self.s, self.x, self.y - 0.32 * EMU, self.w, 0.25 * EMU, t, size=13)

    def foot(self, runs, color):
        text(self.s, self.x, self.y + self.h + 0.08 * EMU, self.w, 0.25 * EMU, runs, size=12, color=color,
             align=PP_ALIGN.CENTER)


def panel_a(group, x, y, w):
    p = Panel(group.shapes, x, y, w)
    p.cells(SMALL); p.ball(np.linalg.norm(E)); p.arrow(FEAT, FEAT + E, BLUE); p.dot(FEAT); p.dot(FEAT + E, BLUE, True)
    p.title("(a) calibration-size vocabulary"); p.foot("top-1 kept", BLUE)
    return p


def panel_b(group, x, y, w):
    p = Panel(group.shapes, x, y, w)
    p.cells(LARGE, bad="cap"); p.ball(np.linalg.norm(E)); p.arrow(FEAT, FEAT + E, RED); p.dot(FEAT); p.dot(FEAT + E, RED, True)
    p.title("(b) open vocabulary (LVIS)")
    p.foot([("same ‖", {}), ("e", {"i": 1}), ("‖  →  top-1 flips", {})], RED)
    return p


def panel_c(group, x, y, w):
    p = Panel(group.shapes, x, y, w)
    p.cells(LARGE, bad="cap")
    q = FEAT.copy()
    for e in E_REST:
        p.arrow(q, q + e, MUTED, lw=1.0); q = q + e
    p.arrow(q, q + E_DOM, RED, lw=1.6, dash=MSO_LINE_DASH_STYLE.DASH)
    cx, cy = p.P(q + E_DOM)
    text(group.shapes, cx - 0.08 * EMU, cy - 0.11 * EMU, 0.16 * EMU, 0.2 * EMU, "✕", size=12, color=RED, align=PP_ALIGN.CENTER, bold=True)
    lx, ly = p.P(q + E_DOM / 2 + np.array([-0.16, 0.16]))
    text(group.shapes, lx - 0.2 * EMU, ly - 0.12 * EMU, 0.4 * EMU, 0.25 * EMU,
         [("e", {"i": 1, "color": RED}), ("ℓ*", {"i": 1, "sub": 1, "color": RED})], size=13, align=PP_ALIGN.CENTER)
    p.dot(FEAT); p.dot(FEAT + sum(E_REST), BLUE, True)
    p.title("(c) remove the diagnosed component")
    p.foot([("ℓ*", {"i": 1}), (" = layer with the largest Δ", {}), ("ℓ", {"i": 1, "sub": 1}), ("  →  G / P", {})], BLUE)
    return p


def legend(shapes, x, y, items, gap=0.25):
    cx = x
    for kind, label in items:
        if kind == "feat": shape(shapes, MSO_SHAPE.OVAL, cx, y, 0.11 * EMU, 0.11 * EMU, fill=INK)
        elif kind in ("kept", "flip"):
            shape(shapes, MSO_SHAPE.OVAL, cx, y, 0.11 * EMU, 0.11 * EMU, fill="FFFFFF", line=BLUE if kind == "kept" else RED, lw=1.75)
        elif kind == "emb": shape(shapes, MSO_SHAPE.ISOSCELES_TRIANGLE, cx, y, 0.09 * EMU, 0.08 * EMU, fill=MUTED)
        elif kind == "ball":
            cn = shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Emu(int(cx - 0.12 * EMU)), Emu(int(y)), Emu(int(cx + 0.12 * EMU)), Emu(int(y)))
            cn.line.color.rgb = C(MUTED); cn.line.width = Pt(1.0); cn.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        tb = text(shapes, cx + 0.15 * EMU, y - 0.1 * EMU, 2.2 * EMU, 0.2 * EMU, label, size=11)
        cx += (0.15 + 0.085 * sum(len(t) for t, _ in (label if isinstance(label, list) else [(label, {})])) + gap) * EMU


# ---------- 점수 막대
def load_scores(n=6):
    d = json.load(open(os.path.join(OUT, "data", "scores_450559_helmet.json")))
    short = {"cap/cap headwear": "cap (headwear)", "baseball cap/jockey cap/golf cap": "baseball cap",
             "gasmask/respirator/gas helmet": "gas mask"}
    return d, [short.get(c, c) for c in d["classes"][:n]], {k: v[:n] for k, v in d["scores"].items()}


def bars(shapes, x, y, w, h, key, ttl, show_labels, label_w=1.4 * EMU, size=11):
    d, cls, sc = load_scores()
    v = np.array(sc[key]); top = int(v.argmax()); flipped = cls[top] != "helmet"
    x0 = x + (label_w if show_labels else 0); W = w - (label_w if show_labels else 0)
    xmax = 1.22
    text(shapes, x0, y - 0.48 * EMU, W, 0.22 * EMU, ttl, size=size + 2)
    text(shapes, x0, y - 0.25 * EMU, W, 0.2 * EMU, cls[top].split(" (")[0] + ("  ✗" if flipped else "  ✓"), size=size,
         color=RED if flipped else BLUE)
    row = h / len(cls)
    for gx in (0.5, 1.0):
        X = x0 + gx / xmax * W
        cn = shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Emu(int(X)), Emu(int(y)), Emu(int(X)), Emu(int(y + h)))
        cn.line.color.rgb = C(GRID); cn.line.width = Pt(0.75)
    for i, c in enumerate(cls):
        cy = y + (i + 0.5) * row
        col = (RED if flipped else BLUE) if i == top else (LBLUE if c == "helmet" else AXIS)
        bw = max(v[i], 0.004) / xmax * W
        rect(shapes, x0, cy - 0.31 * row, bw, 0.62 * row, fill=col)
        text(shapes, x0 + bw + 0.04 * EMU, cy - 0.1 * EMU, 0.5 * EMU, 0.2 * EMU, f"{v[i]:.2f}", size=size - 1,
             color=INK if i == top else MUTED, anchor=MSO_ANCHOR.MIDDLE)
        if show_labels:
            text(shapes, x, cy - 0.1 * EMU, label_w - 0.08 * EMU, 0.2 * EMU, c, size=size, color=MUTED, align=PP_ALIGN.RIGHT,
                 anchor=MSO_ANCHOR.MIDDLE)
    for X0, Y0, X1, Y1 in ((x0, y, x0, y + h), (x0, y + h, x0 + W, y + h)):
        cn = shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Emu(int(X0)), Emu(int(Y0)), Emu(int(X1)), Emu(int(Y1)))
        cn.line.color.rgb = C(AXIS); cn.line.width = Pt(1.0)
    for gx, lab in ((0, "0"), (0.5, ".5"), (1.0, "1")):
        X = x0 + gx / xmax * W
        text(shapes, X - 0.2 * EMU, y + h + 0.05 * EMU, 0.4 * EMU, 0.2 * EMU, lab, size=size - 1, color=MUTED, align=PP_ALIGN.CENTER)
    return d


def main():
    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    LEG = [("feat", "FP region feature"), ("kept", "quantized (top-1 kept)"), ("flip", "quantized (top-1 flipped)"),
           ("emb", "text embedding"), ("ball", [("error ball ‖", {}), ("e", {"i": 1}), ("‖", {})])]

    # 슬라이드 1: A
    s = prs.slides.add_slide(blank)
    w = 4.0 * EMU; y = 1.6 * EMU
    for i, f in enumerate((panel_a, panel_b, panel_c)):
        g = s.shapes.add_group_shape(); g.name = f"A panel {'abc'[i]}"
        f(g, (0.45 + i * 4.22) * EMU, y, w)
    g = s.shapes.add_group_shape(); g.name = "A legend"
    legend(g.shapes, 1.6 * EMU, y + 3.43 * EMU + 0.75 * EMU, LEG)
    s.notes_slide.notes_text_frame.text = CAPTIONS["A"]

    # 슬라이드 2: B
    s = prs.slides.add_slide(blank)
    for i, (k, t) in enumerate((("fp", "FP32"), ("brecq", "BRECQ"), ("ours", "Ours"))):
        g = s.shapes.add_group_shape(); g.name = f"B {t}"
        x = (0.9 + (0 if i == 0 else 1.8 + 3.3 * i)) * EMU
        d = bars(g.shapes, x, 2.0 * EMU, (5.1 if i == 0 else 3.3) * EMU - (0.3 * EMU if i else 0), 3.4 * EMU, k, t, i == 0,
                 label_w=1.8 * EMU, size=13)
    text(s.shapes, 2.7 * EMU, 6.05 * EMU, 9 * EMU, 0.25 * EMU,
         f"YOLO-World-S W4A8, LVIS vocabulary ({d['n_vocab']} classes), scores at the FP helmet anchor", size=11, color=MUTED)
    s.notes_slide.notes_text_frame.text = CAPTIONS["B"]

    # 슬라이드 3: C
    s = prs.slides.add_slide(blank)
    w = 3.55 * EMU; y = 1.9 * EMU
    for i, f in enumerate((panel_a, panel_b)):
        g = s.shapes.add_group_shape(); g.name = f"C panel {'ab'[i]}"
        f(g, (0.35 + i * 3.75) * EMU, y, w)
    text(s.shapes, 7.4 * EMU, 1.05 * EMU, 5 * EMU, 0.25 * EMU, "(c) measured: helmet region, W4A8", size=13)
    for i, (k, t) in enumerate((("fp", "FP32"), ("brecq", "BRECQ"), ("ours", "Ours"))):
        g = s.shapes.add_group_shape(); g.name = f"C bars {t}"
        x = (7.4 + (0 if i == 0 else 1.2 + 1.6 * i)) * EMU
        bars(g.shapes, x, y, (2.5 if i == 0 else 1.3) * EMU, 3.0 * EMU, k, t, i == 0, label_w=1.2 * EMU, size=10)
    g = s.shapes.add_group_shape(); g.name = "C legend"
    legend(g.shapes, 0.5 * EMU, y + w * 3.0 / 3.5 + 0.7 * EMU, LEG[:4])
    s.notes_slide.notes_text_frame.text = CAPTIONS["C"]

    for sl in prs.slides:                      # 테마 기본 효과(그림자) 제거: 색과 선은 모두 명시했다
        for st in list(sl.shapes._spTree.iter(qn("p:style"))):
            st.getparent().remove(st)
    f = os.path.join(OUT, "concept_flip.pptx")
    prs.save(f); print("->", f)


if __name__ == "__main__":
    main()
