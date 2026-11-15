"""그림 1 1단판 과장판(11-15): 논문 그림(fig1_voronoi_1col)의 대비를 키운 편집 가능한 PPT.

make_fig1_1col_pptx.py(11-13, 실측 반경 비율판)와 다른 점
  - 레이아웃: 11-15 사용자가 PowerPoint에서 고친 논문판을 따른다(제목 줄 오른쪽에 flip, 한 줄 범례, "(c) Ours").
  - (a) person을 멀리 두어 오차 원 전체가 helmet 칸 안에 들어가게 했다(11-13판은 원이 person 칸 경계를 0.16 넘었다).
  - (b) 비슷한 클래스(cap, football helmet, hat, baseball cap, gas mask, visor)를 helmet 쪽으로 CROWD배 당겨 칸을 좁혔다.
  - (c) 오차 반경을 BRECQ의 SCALE_X배로 줄였다(실측 sqrt(0.0373/0.0702)=0.73 대신 과장). 원은 helmet 칸 안에 있지만
    (a)보다 경계에 가깝게 두었다(실측 flip 10.7%가 (a)의 3.9%보다 높다는 것과 어긋나지 않게).
  - 칸 모양, 위치, 반경은 모두 설명용(개념도)이다. 실측 값은 패널 위 flip(리더보드 seed 0)뿐이다.
사용(저장소 루트, python-pptx 필요):
  PYTHONPATH=<python-pptx 경로> .venv/bin/python figures/concept_flip/src/make_fig1_1col_x_pptx.py [출력 경로]
출력(기본): documents/figure/fig1/fig1_voronoi_1col_x.pptx
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
from make_concept_pptx import text, shape, C, EMU, INK, MUTED, AXIS, BLUE, RED
from make_concept import LIM
import make_fig1_voronoi as fv
from make_fig1_1col_pptx import P1

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "documents", "figure", "fig1", "fig1_voronoi_1col_x.pptx")
SW, SH = 6.6, 2.42
FEAT, E = fv.FEAT, fv.E
CROWD = 0.78          # (b)(c) 비슷한 클래스를 helmet 쪽으로 당기는 비율
SCALE_X = 0.5         # (c) 오차 반경 / BRECQ 오차 반경(과장, 실측 0.73)
SMALL = dict(fv.SMALL, person=(1.45, -1.1))
SIMILAR = ("cap", "football helmet", "hat", "baseball cap", "gas mask", "visor")
LARGE = dict(SMALL, **{n: tuple(CROWD * np.array(fv.LARGE[n])) for n in SIMILAR})
# 11-15: 80 -> 1,203이면 클래스가 15배라 (b)(c)에 이름 없는 클래스를 촘촘히 더한다(떨린 육각 격자, 고정 seed).
# helmet 주변 HOLE 안에는 넣지 않아 helmet 칸 모양은 비슷한 클래스 6개가 정한다.
EXTRA_STEP, HOLE, MIN_GAP = 0.5, 1.0, 0.33


def extra_anchors(base):
    rng = np.random.default_rng(0); out = {}; k = 0
    for j, y in enumerate(np.arange(LIM[2] - 0.3, LIM[3] + 0.3, EXTRA_STEP * 0.866)):
        for x in np.arange(LIM[0] - 0.3 + (EXTRA_STEP / 2) * (j % 2), LIM[1] + 0.3, EXTRA_STEP):
            v = np.array([x, y]) + rng.uniform(-0.12, 0.12, 2)
            if np.linalg.norm(v) < HOLE: continue
            if any(np.linalg.norm(v - np.array(b)) < MIN_GAP for b in list(base.values()) + list(out.values())): continue
            out[f"c{k}"] = tuple(v); k += 1
    return out


LARGE.update(extra_anchors(LARGE))
CAPTION = ("Figure 1 (exaggerated schematic). Cells, positions and circle radii are illustrative; only the flip values "
           "are measured (YOLO-World-S W4A8, BRECQ vs. Ours, leaderboard seed 0, LVIS minival).")


class PX(P1):
    """P1과 같고, cap 라벨만 삼각형 위에 둔다(당겨진 cap 쪽으로 양자화 점이 와서 아래 라벨과 겹친다)."""
    def cells_min(self, anchors, bad=None, show=("helmet", "cap")):
        from make_concept_pptx import voronoi, poly, rect, CELL_OK, CELL_BAD, CELL
        for n, pg in voronoi(anchors).items():
            if not pg: continue                       # 상자 밖 클래스(칸 없음)
            fill = CELL_OK if n == "helmet" else (CELL_BAD if n == bad else CELL)
            poly(self.s, [self.P(v) for v in pg], fill=fill, line=AXIS, lw=0.75)
        rect(self.s, self.x, self.y, self.w, self.h, line=AXIS, lw=0.75)
        for n, v in anchors.items():
            if LIM[0] + 0.08 < v[0] < LIM[1] - 0.08 and LIM[2] + 0.08 < v[1] < LIM[3] - 0.08:   # 가장자리에서 잘리는 삼각형은 뺀다
                cx, cy = self.P(v)
                shape(self.s, MSO_SHAPE.ISOSCELES_TRIANGLE, cx, cy, 0.075 * EMU, 0.065 * EMU, fill=MUTED)
                if n in show:
                    dy = -0.25 * EMU if n == "cap" else 0.06 * EMU
                    text(self.s, cx - 0.4 * EMU, cy + dy, 0.8 * EMU, 0.18 * EMU, n, size=10.5, color=INK,
                         align=PP_ALIGN.CENTER)


def margin(anchors, r):
    """FP 특징에서 helmet 칸 경계까지 최소 거리 - 반경(양수면 원이 칸 안)."""
    h = np.array(anchors["helmet"], float); m = []
    for n, b in anchors.items():
        if n == "helmet": continue
        d = np.array(b, float) - h; u = d / np.linalg.norm(d)
        m.append(float(np.dot((h + b) / 2 - FEAT, u)))
    return min(m) - r


def nearest(anchors, p):
    return min(anchors, key=lambda n: np.sum((np.array(anchors[n]) - p) ** 2))


def main():
    R = fv.R
    specs = [("(a) 80 classes", SMALL, None, E, BLUE, f"flip {fv.FLIP['coco']:.1f}%", MUTED, False),
             ("(b) 1,203 classes", LARGE, "cap", E, RED, f"flip {fv.FLIP['lvis']:.1f}%", RED, False),
             ("(c) Ours", LARGE, "cap", E * SCALE_X, BLUE, f"flip {fv.FLIP['ours']:.1f}%", BLUE, True)]
    # 그림이 말하는 것이 기하적으로 맞는지 확인
    assert nearest(SMALL, FEAT + E) == "helmet" and margin(SMALL, R) > 0.1
    assert nearest(LARGE, FEAT + E) == "cap" and margin(LARGE, R) < -0.2
    assert nearest(LARGE, FEAT + E * SCALE_X) == "helmet" and 0 < margin(LARGE, R * SCALE_X) < margin(SMALL, R)
    print(f"원과 helmet 칸 경계 여유: (a) {margin(SMALL, R):+.2f}  (b) {margin(LARGE, R):+.2f}  (c) {margin(LARGE, R * SCALE_X):+.2f}")

    prs = Presentation(); prs.slide_width = Inches(SW); prs.slide_height = Inches(SH)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    w = 2.1 * EMU; y = 0.3 * EMU; gap = 0.085 * EMU; x0 = 0.04 * EMU
    for i, (ttl, anchors, bad, err, col, ft, fcol, ref) in enumerate(specs):
        g = s.shapes.add_group_shape(); g.name = f"panel {'abc'[i]}"
        p = PX(g.shapes, x0 + i * (w + gap), y, w)
        p.cells_min(anchors, bad=bad)
        if ref:                                       # (b)의 오차 원(비교용, 옅은 점선)
            p.ball(R, color=AXIS, dash=MSO_LINE_DASH_STYLE.ROUND_DOT, lw=0.75)
        p.ball(float(np.linalg.norm(err)))
        p.arrow(FEAT, FEAT + err, col, lw=1.4)
        p.dot(FEAT); p.dot(FEAT + err, col, True)
        text(g.shapes, p.x, y - 0.27 * EMU, w, 0.22 * EMU, ttl, size=11.5)
        text(g.shapes, p.x, y - 0.27 * EMU, w, 0.22 * EMU, ft, size=11.5, color=fcol, align=PP_ALIGN.RIGHT)
    # 범례(한 줄)
    g = s.shapes.add_group_shape(); g.name = "legend"
    ly = (y + p.h + 0.17 * EMU)
    items = [("feat", "FP feature", 0.16), ("flip", "quantized, flipped", 1.22), ("kept", "quantized, kept", 2.74),
             ("emb", "text embedding", 4.1), ("ball", [("error ‖", {}), ("e", {"i": 1}), ("‖", {})], 5.55)]
    for kind, lab, lx in items:
        cx = lx * EMU; d = 0.085 * EMU
        if kind == "feat": shape(g.shapes, MSO_SHAPE.OVAL, cx, ly, d, d, fill=INK)
        elif kind in ("kept", "flip"):
            shape(g.shapes, MSO_SHAPE.OVAL, cx, ly, d, d, fill="FFFFFF", line=BLUE if kind == "kept" else RED, lw=1.5)
        elif kind == "emb": shape(g.shapes, MSO_SHAPE.ISOSCELES_TRIANGLE, cx, ly, 0.075 * EMU, 0.065 * EMU, fill=MUTED)
        else:
            cn = g.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Emu(int(cx - 0.11 * EMU)), Emu(int(ly)),
                                        Emu(int(cx + 0.11 * EMU)), Emu(int(ly)))
            cn.line.color.rgb = C(MUTED); cn.line.width = Pt(1.0); cn.line.dash_style = MSO_LINE_DASH_STYLE.DASH
            cx += 0.05 * EMU
        text(g.shapes, cx + 0.12 * EMU, ly - 0.1 * EMU, 1.5 * EMU, 0.2 * EMU, lab, size=11)
    s.notes_slide.notes_text_frame.text = CAPTION
    for st in list(s.shapes._spTree.iter(qn("p:style"))):   # 테마 기본 그림자 제거
        st.getparent().remove(st)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    prs.save(OUT); print("->", OUT)


if __name__ == "__main__":
    main()
