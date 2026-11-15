"""그림 1 1단판(fig1_voronoi_1col)의 편집 가능한 PPT (11-13). make_fig1_voronoi.py의 one_column()과 같은 좌표·같은
실측 값(flip 3.94 / 17.49 / 10.69%, 반경 비율 sqrt(0.0373/0.0702))을 PowerPoint 기본 도형으로 다시 그린다.
슬라이드 = 그림 크기의 2배(6.6 x 3.1 in). 패널·범례마다 그룹. 캡션 초안은 노트.
사용(저장소 루트, python-pptx 필요. .venv에는 없어 scratch에 따로 설치해 썼다):
  PYTHONPATH=<python-pptx 경로> .venv/bin/python figures/concept_flip/src/make_fig1_1col_pptx.py [출력 경로]
출력(기본): documents/figure/fig1/fig1_voronoi_1col.pptx
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.oxml.ns import qn
import make_concept_pptx as cp
from make_concept_pptx import text, poly, shape, arrow, rect, voronoi, C, EMU
from make_concept_pptx import INK, MUTED, AXIS, BLUE, RED, CELL_OK, CELL_BAD, CELL
import make_fig1_voronoi as fv
from make_concept import LIM

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "documents", "figure", "fig1", "fig1_voronoi_1col.pptx")
SW, SH = 6.6, 3.1
CAPTION = ("Figure 1. Cells mark the region features whose top-1 class is each text embedding (layout illustrative). "
           "(a, b) YOLO-World-S quantized by BRECQ at W4A8 flips the top-1 of 3.9% of confident predictions on an 80-class "
           "vocabulary but 17.5% on the 1,203-class LVIS vocabulary, where similar classes crowd the cell. (c) Our method "
           "keeps the text embeddings (the same 8-bit constants as BRECQ) and lowers the region-embedding error ||e||^2 to "
           "0.53x (circle radii to scale; 500 LVIS minival images), reducing the flip to 10.7%.\n\n"
           "Sources: flip = leaderboard seed 0 (LVIS minival); ||e||^2 = figures/concept_flip/data/w4a8_emb_flip_n500.json.")


class P1(cp.Panel):
    def cells_min(self, anchors, bad=None, show=("helmet", "cap")):
        for n, pg in voronoi(anchors).items():
            fill = CELL_OK if n == "helmet" else (CELL_BAD if n == bad else CELL)
            poly(self.s, [self.P(v) for v in pg], fill=fill, line=AXIS, lw=0.75)
        rect(self.s, self.x, self.y, self.w, self.h, line=AXIS, lw=0.75)
        for n, v in anchors.items():
            if LIM[0] < v[0] < LIM[1] and LIM[2] < v[1] < LIM[3]:
                cx, cy = self.P(v)
                shape(self.s, MSO_SHAPE.ISOSCELES_TRIANGLE, cx, cy, 0.075 * EMU, 0.065 * EMU, fill=MUTED)
                if n in show:
                    text(self.s, cx - 0.4 * EMU, cy + 0.06 * EMU, 0.8 * EMU, 0.18 * EMU, n, size=10.5, color=INK,
                         align=PP_ALIGN.CENTER)

    def dot(self, v, color=INK, hollow=False):
        cx, cy = self.P(v); d = 0.085 * EMU
        if hollow:
            shape(self.s, MSO_SHAPE.OVAL, cx, cy, d, d, fill="FFFFFF", line=color, lw=1.5)
        else:
            shape(self.s, MSO_SHAPE.OVAL, cx, cy, d, d, fill=color)

    def ball(self, r, color=MUTED, dash=MSO_LINE_DASH_STYLE.DASH, lw=1.0):
        cx, cy = self.P(fv.FEAT)
        rw = r / (LIM[1] - LIM[0]) * self.w
        shape(self.s, MSO_SHAPE.OVAL, cx, cy, 2 * rw, 2 * rw, line=color, lw=lw, dash=dash)

    def head(self, t):
        text(self.s, self.x, self.y - 0.27 * EMU, self.w, 0.22 * EMU, t, size=12.5)

    def foot2(self, t, color):
        text(self.s, self.x, self.y + self.h + 0.05 * EMU, self.w, 0.22 * EMU, t, size=12, color=color,
             align=PP_ALIGN.CENTER)


def main():
    prs = Presentation(); prs.slide_width = Inches(SW); prs.slide_height = Inches(SH)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    w = 2.07 * EMU; y = 0.42 * EMU; gap = 0.1 * EMU; x0 = 0.05 * EMU
    E, FEAT = fv.E, fv.FEAT
    specs = [
        ("(a) 80 classes", fv.SMALL, None, E, BLUE, f"flip {fv.FLIP['coco']:.1f}%", MUTED, False),
        ("(b) 1,203 classes", fv.LARGE, "cap", E, RED, f"flip {fv.FLIP['lvis']:.1f}%", RED, False),
        ("(c) + Ours", fv.LARGE, "cap", E * fv.SCALE_G, BLUE, f"flip {fv.FLIP['ours']:.1f}%", BLUE, True),
    ]
    for i, (ttl, anchors, bad, err, col, ft, fcol, ref) in enumerate(specs):
        g = s.shapes.add_group_shape(); g.name = f"panel {'abc'[i]}"
        p = P1(g.shapes, x0 + i * (w + gap), y, w)
        p.cells_min(anchors, bad=bad)
        if ref:                                     # (b)의 오차 원(비교용, 옅은 점선)
            p.ball(fv.R, color=AXIS, dash=MSO_LINE_DASH_STYLE.ROUND_DOT, lw=0.75)
        p.ball(float(np.linalg.norm(err)))
        p.arrow(FEAT, FEAT + err, col, lw=1.4)
        p.dot(FEAT); p.dot(FEAT + err, col, True)
        p.head(ttl); p.foot2(ft, fcol)
    # 범례(3열 x 2행)
    g = s.shapes.add_group_shape(); g.name = "legend"
    items = [("feat", "FP feature"), ("kept", "quantized, kept"), ("ball", [("error ‖", {}), ("e", {"i": 1}), ("‖", {})]),
             ("flip", "quantized, flipped"), ("emb", "text embedding")]
    cols = [1.25, 2.85, 4.7]; rows = [2.62, 2.86]
    pos = [(cols[0], rows[0]), (cols[0], rows[1]), (cols[2], rows[0]), (cols[1], rows[0]), (cols[1], rows[1])]
    for (kind, lab), (lx, ly) in zip(items, pos):
        cx, cy = lx * EMU, ly * EMU; d = 0.085 * EMU
        if kind == "feat": shape(g.shapes, MSO_SHAPE.OVAL, cx, cy, d, d, fill=INK)
        elif kind in ("kept", "flip"):
            shape(g.shapes, MSO_SHAPE.OVAL, cx, cy, d, d, fill="FFFFFF", line=BLUE if kind == "kept" else RED, lw=1.5)
        elif kind == "emb": shape(g.shapes, MSO_SHAPE.ISOSCELES_TRIANGLE, cx, cy, 0.075 * EMU, 0.065 * EMU, fill=MUTED)
        else:
            cn = g.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Emu(int(cx - 0.11 * EMU)), Emu(int(cy)),
                                        Emu(int(cx + 0.11 * EMU)), Emu(int(cy)))
            cn.line.color.rgb = C(MUTED); cn.line.width = Pt(1.0); cn.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        text(g.shapes, cx + 0.14 * EMU, cy - 0.09 * EMU, 1.6 * EMU, 0.18 * EMU, lab, size=10.5)
    s.notes_slide.notes_text_frame.text = CAPTION
    for st in list(s.shapes._spTree.iter(qn("p:style"))):   # 테마 기본 그림자 제거
        st.getparent().remove(st)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    prs.save(OUT); print("->", OUT)


if __name__ == "__main__":
    main()
