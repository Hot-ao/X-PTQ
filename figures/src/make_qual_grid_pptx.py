"""qual_grid(부록 비트별 정성 비교)의 편집 가능한 PPT (11-15).
qual_grid.py를 같은 인자로 그대로 돌려 matplotlib 그림을 만든 뒤(파일 저장은 막는다), 각 패널의 사진 위치, 박스(Rectangle),
라벨(배경 있는 글자), 제목, 행 이름을 화면 좌표에서 읽어 PowerPoint 기본 도형으로 옮긴다. 그래서 배치는 PDF와 같다.
슬라이드 = 그림 크기(7.0 in 폭). 패널마다 그룹. 박스 색은 qual_figure.py(OK 파랑 / BAD 빨강 / OTHER 초록).
사용(저장소 루트, python-pptx 필요):
  PYTHONPATH=<python-pptx 경로> .venv/bin/python figures/src/make_qual_grid_pptx.py [출력 경로]
출력(기본): documents/figure/qual/qual_grid.pptx
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "documents", "figure", "qual", "qual_grid.pptx")
PICKS = "446651,153632,237071"          # 논문 그림과 같은 이미지(11-16 최종. 이전: 229997, 469174, 478393, 381971 등)
BASES = "brecq,qdrop,pdquant"            # 블록별 기준 PTQ(11-15, 각 + Ours, 모두 seed 0)
CROP = ""                               # 자르기 없음(이전 478393:0,130,480,530)
CAPTION = ("비트별 정성 비교 (YOLO-World-S, LVIS 어휘, seed 0). 위부터 BRECQ, QDrop, PD-Quant를 기준 PTQ로 쓴 예시다. "
           "왼쪽은 FP32, 오른쪽은 기준 PTQ(위 행)와 + Ours(아래 행)의 W4A8, W4A5, W4A4 결과다. "
           "파랑은 FP와 같은 판정, 빨강은 바뀐 판정, 빨간 점선(missed)은 그 위치를 놓친 것, 초록은 그 밖의 검출이다. "
           "BRECQ는 W4A8에서 helmet을 cap으로 바꾸고 W4A5부터 놓친다. QDrop은 W4A8·W4A5에서 knife를 fork로 바꾸고 W4A4에서 놓친다. "
           "PD-Quant는 W4A8에서는 맞히지만 W4A5와 W4A4에서 tennis ball을 apple과 lime으로 바꾼다. "
           "Ours를 더하면 세 경우 모두 세 비트에서 FP 판정을 지킨다.\n"
           "생성: figures/src/make_qual_grid_pptx.py (qual_grid.py --pick " + PICKS + " --bases " + BASES + "과 같은 배치).")

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure as mfig
import matplotlib.transforms
from matplotlib.colors import to_hex
from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.oxml.ns import qn

EMU_IN = 914400
CAP = {}


def fake_savefig(self, *a, **k):        # qual_grid.py가 figures/qual/qual_grid.*를 덮어쓰지 않게 하고 그림만 잡는다
    CAP["fig"] = self


def rgb(c):
    return RGBColor.from_string(to_hex(c)[1:].upper())


def main():
    mfig.Figure.savefig = fake_savefig
    import qual_grid
    sys.argv = ["qual_grid.py", "--pick", PICKS, "--bases", BASES] + (["--crop", CROP] if CROP else [])
    qual_grid.main()
    fig = CAP["fig"]
    fig.canvas.draw(); rnd = fig.canvas.get_renderer()
    dpi = fig.dpi; FW, FH = fig.get_size_inches(); HPX = FH * dpi

    def box(bb):                         # display bbox(px, 원점 왼쪽 아래) -> EMU (x, y, w, h), 원점 왼쪽 위
        return (Emu(int(bb.x0 / dpi * EMU_IN)), Emu(int((HPX - bb.y1) / dpi * EMU_IN)),
                Emu(int(bb.width / dpi * EMU_IN)), Emu(int(bb.height / dpi * EMU_IN)))

    prs = Presentation(); prs.slide_width = Emu(int(FW * EMU_IN)); prs.slide_height = Emu(int(FH * EMU_IN))
    s = prs.slides.add_slide(prs.slide_layouts[6])

    def tbox(shapes, bb, txt, size, color="0B0B0B", fill=None, align=PP_ALIGN.LEFT, rot=0, bold=False):
        x, y, w, h = box(bb)
        if rot:                          # 90도 회전 글자: 회전 전 상자는 가로·세로가 바뀐다
            cx, cy = x + w // 2, y + h // 2; w, h = h, w; x, y = cx - w // 2, cy - h // 2
        t = shapes.add_textbox(x, y, w, h)
        if rot:
            t.rotation = rot
        tf = t.text_frame; tf.word_wrap = False
        tf.margin_left = tf.margin_right = Emu(int(1.0 / 72 * EMU_IN)); tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]; p.alignment = align
        r = p.add_run(); r.text = txt; r.font.size = Pt(size); r.font.name = "Arial"; r.font.bold = bold
        r.font.color.rgb = RGBColor.from_string(color)
        if fill is not None:
            t.fill.solid(); t.fill.fore_color.rgb = fill
        return t

    n = 0
    for ax in fig.axes:
        if not ax.images:                # 자리만 잡은 빈 축(FP32 열 바깥 틀)
            continue
        ax.apply_aspect()
        bb = ax.get_window_extent(rnd)
        g = s.shapes.add_group_shape(); n += 1
        title = ax.get_title(loc="left"); ylab = ax.get_ylabel()
        g.name = f"panel {n}" + (f" {title}" if title else "") + (f" {ylab}" if ylab else "")
        im = ax.images[0]
        tmp = os.path.join(os.path.dirname(OUT), f".qg_{n}.png")
        from PIL import Image
        import numpy as np
        Image.fromarray(np.asarray(im.get_array())).save(tmp)
        x, y, w, h = box(bb)
        pic = g.shapes.add_picture(tmp, x, y, w, h); pic.name = "image"; os.remove(tmp)
        for p in ax.patches:             # 검출 박스
            pb = matplotlib.transforms.Bbox.intersection(p.get_window_extent(rnd), bb)   # 자른 사진 밖은 버린다
            if pb is None or pb.width < 1 or pb.height < 1:
                continue
            fb = p.get_window_extent(rnd)

            def style(ln):
                ln.color.rgb = rgb(p.get_edgecolor()); ln.width = Pt(p.get_linewidth())
                if p.get_linestyle() not in ("-", "solid"):
                    ln.dash_style = MSO_LINE_DASH_STYLE.DASH
            if all(abs(a - b) < 0.5 for a, b in zip(fb.extents, pb.extents)):
                x, y, w, h = box(pb)
                sh = g.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
                sh.fill.background(); style(sh.line)
                sh.name = "box " + to_hex(p.get_edgecolor())
                continue
            # 사진 밖으로 나간 박스: matplotlib처럼 사진 안에 남는 변만 선으로 그린다
            E = lambda px, py: (Emu(int(px / dpi * EMU_IN)), Emu(int((HPX - py) / dpi * EMU_IN)))
            sides = []
            if fb.x0 >= bb.x0 - 0.5: sides.append(((fb.x0, pb.y0), (fb.x0, pb.y1)))
            if fb.x1 <= bb.x1 + 0.5: sides.append(((fb.x1, pb.y0), (fb.x1, pb.y1)))
            if fb.y0 >= bb.y0 - 0.5: sides.append(((pb.x0, fb.y0), (pb.x1, fb.y0)))
            if fb.y1 <= bb.y1 + 0.5: sides.append(((pb.x0, fb.y1), (pb.x1, fb.y1)))
            for a, b in sides:
                cn = g.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, *E(*a), *E(*b))
                style(cn.line); cn.name = "box side " + to_hex(p.get_edgecolor())
        for t in ax.texts:               # 라벨(배경 상자 포함)
            bp = t.get_bbox_patch()
            lb = bp.get_window_extent(rnd) if bp is not None else t.get_window_extent(rnd)
            fill = rgb(bp.get_facecolor()) if bp is not None else None
            tb = tbox(g.shapes, lb, t.get_text(), t.get_fontsize(), color=to_hex(t.get_color())[1:].upper(), fill=fill)
            tb.name = "label " + t.get_text()
        if title:
            lt = ax._left_title             # draw()는 loc="left" 제목을 쓴다(ax.title은 가운데 제목, 비어 있음)
            tbox(g.shapes, lt.get_window_extent(rnd), title, lt.get_fontsize()).name = "title"
        if ylab:
            yl = ax.yaxis.label
            tbox(g.shapes, yl.get_window_extent(rnd), ylab, yl.get_fontsize(), rot=270,
                 align=PP_ALIGN.CENTER).name = "row name"
    s.notes_slide.notes_text_frame.text = CAPTION
    for st in list(s.shapes._spTree.iter(qn("p:style"))):   # 테마 기본 그림자·윤곽 제거
        st.getparent().remove(st)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    prs.save(OUT); print(f"-> {OUT} (패널 {n}개)")


if __name__ == "__main__":
    main()
