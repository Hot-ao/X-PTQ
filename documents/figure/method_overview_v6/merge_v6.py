"""Insert tsm_numeric_v6 (user-edited) into the TSM panel of method_overview_v6_base.pptx, unchanged, as one group.
Only a uniform scale k is applied (group transform + font sizes + line widths), so it looks the same as the original.
Usage: python3 merge_v5.py  (from this folder)"""
import re, sys, zipfile, shutil, os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "../method_overview_v3/tsm/tsm_numeric_v6.pptx")
BASE = os.path.join(HERE, "method_overview_v6_base.pptx")
OUT = os.path.join(HERE, "method_overview_v6.pptx")
EMU = 914400
X, Y, Wt = 6.92, 0.36, 3.5                        # target box (in): T.x + 0.12, T.y + 0.3, width 3.5

src = zipfile.ZipFile(SRC)
sw, sh = map(int, re.search(r'<p:sldSz cx="(\d+)" cy="(\d+)"', src.read("ppt/presentation.xml").decode()).groups())
k = Wt * EMU / sw
xml = src.read("ppt/slides/slide1.xml").decode("utf-8")
body = re.search(r"</p:grpSpPr>(.*)</p:spTree>", xml, re.S).group(1)
body = re.sub(r'(<p:cNvPr id=")(\d+)"', lambda m: f'{m.group(1)}{int(m.group(2)) + 5000}"', body)
body = re.sub(r'(<a:(?:rPr|defRPr|endParaRPr)\b[^>]*?\bsz=")(\d+)"', lambda m: f'{m.group(1)}{max(100, round(int(m.group(2)) * k))}"', body)
body = re.sub(r'(<a:ln\b[^>]*?\bw=")(\d+)"', lambda m: f'{m.group(1)}{round(int(m.group(2)) * k)}"', body)
ox, oy, cx, cy = round(X * EMU), round(Y * EMU), round(sw * k), round(sh * k)
grp = (f'<p:grpSp><p:nvGrpSpPr><p:cNvPr id="4999" name="TSM v6 (tsm_numeric_v6.pptx)"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
       f'<p:grpSpPr><a:xfrm><a:off x="{ox}" y="{oy}"/><a:ext cx="{cx}" cy="{cy}"/><a:chOff x="0" y="0"/><a:chExt cx="{sw}" cy="{sh}"/></a:xfrm></p:grpSpPr>'
       f'{body}</p:grpSp>')

zin = zipfile.ZipFile(BASE)
tmp = OUT + ".tmp"
with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for it in zin.infolist():
        if it.filename.endswith("/"):
            continue
        data = zin.read(it.filename)
        if it.filename == "ppt/slides/slide1.xml":
            s = data.decode("utf-8")
            if 'xmlns:a16' not in s and 'a16:' in grp:
                pass  # a16 is declared inline on each element in the source, nothing to add
            s = s.replace("</p:spTree>", grp + "</p:spTree>", 1)
            data = s.encode("utf-8")
        zout.writestr(it, data)
shutil.move(tmp, OUT)
print(f"k = {k:.3f}, group at ({X}, {Y}) in, {cx/EMU:.2f} x {cy/EMU:.2f} in ->", OUT)
