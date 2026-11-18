"""bits_curve_models.csv -> bits_curve_models.pptx (11-16).

논문 그림 bits_curve(YOLO-World-S)와 같은 모양의 꺾은선 차트를 모델마다 한 장씩 만든다(PowerPoint에서 편집 가능한 차트).
- 데이터: 같은 폴더의 bits_curve_models.csv (seed 0, 리더보드에서 추출. 열: <기법>_lvis_ap, fp_lvis_ap, ours_config ...)
- 계열: Ours (on BRECQ / QDrop / PD-Quant), BRECQ, QDrop, PD-Quant, FP32 점선, 범례용 "baseline collapsed" 빈 계열
- 속 빈 점: 기준선(BRECQ, QDrop, PD-Quant)의 LVIS AP < 15 (논문 그림과 같은 규칙)
- 값 라벨: Ours (on BRECQ)만. AP는 위(FP 점선과 2 AP 안이면 오른쪽), flip은 아래
- --metric lvis_flip이면 같은 모양으로 LVIS flip(%) 곡선(y 0-100, FP선·속 빈 점 없음)
- 범례: 플롯 아래(선을 가리지 않게). 논문 그림처럼 축 안에 넣으려면 PowerPoint에서 옮긴다
- y축 상한: 그 모델의 최댓값(FP 포함) + 3을 5 단위로 올림

사용:
  python make_bits_curve_pptx.py [--csv bits_curve_models.csv] [--out bits_curve_models.pptx]
         [--models "YOLO-World-S,YOLO-World-L,YOLOE-v8s"] [--metric lvis_ap|lvis_flip]
필요: python3(표준 라이브러리만), node + pptxgenjs.
  pptxgenjs 위치는 환경변수 PPTXGENJS_DIR(node_modules가 든 폴더)로 줄 수 있다.
  기본으로 이 파일 옆, 저장소의 tools/pptxgen, 현재 폴더 순으로 찾는다. 없으면: npm install pptxgenjs
"""
from __future__ import annotations
import argparse, csv, json, math, os, re, shutil, subprocess, sys, tempfile, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
BITS = ["W8A8", "W4A8", "W4A6", "W4A5", "W4A4"]
COLLAPSE = 15.0
# (csv 열 접두어, 범례 이름, 색, 선 굵기 EMU, 대시, 속 빈 점 판정 대상)
SERIES = [("ours", "Ours (on BRECQ)", "2A78D6", 25400, "solid", False),
          ("qdrop+ours", "Ours (on QDrop)", "2A78D6", 20320, "dash", False),
          ("pdquant+ours", "Ours (on PD-Quant)", "2A78D6", 20320, "sysDot", False),
          ("brecq", "BRECQ", "EB6834", 16510, "solid", True),
          ("qdrop", "QDrop", "1BAF7A", 16510, "solid", True),
          ("pdquant", "PD-Quant", "9A6BC9", 16510, "solid", True)]
INK, MUTED = "0B0B0B", "6B6A64"


def load(path, metric):
    by = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        by.setdefault(r["model"], {})[r["setting"]] = r
    out = []
    for model, rows in by.items():
        miss = [b for b in BITS if b not in rows]
        if miss:
            sys.exit(f"{model}: {miss} 칸이 csv에 없다")
        val = lambda k, b: (float(rows[b][f"{k}_{metric}"]) if rows[b].get(f"{k}_{metric}") not in (None, "") else None)
        ser = {k: [val(k, b) for b in BITS] for k, *_ in SERIES}
        hollow = {k: [i for i, v in enumerate(ser[k]) if v is not None and v < COLLAPSE] if h and metric == "lvis_ap" else []
                  for k, _, _, _, _, h in SERIES}
        fp = float(rows["W8A8"]["fp_lvis_ap"]) if metric == "lvis_ap" else None
        cfg = {b: rows[b]["ours_config"] for b in BITS}
        out.append(dict(model=model, ser=ser, hollow=hollow, fp=fp, cfg=cfg))
    return out


JS = r"""
const pptxgen = require("pptxgenjs");
const spec = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));
(async () => {
  const pres = new pptxgen();
  pres.defineLayout({ name: "FIG", width: 3.4, height: 2.55 });
  pres.layout = "FIG";
  pres.title = spec.title;
  for (const c of spec.charts) {
    const s = pres.addSlide();
    s.addChart(pres.charts.LINE, c.series.map(x => ({ name: x.name, labels: spec.labels, values: x.values })), {
      x: 0.05, y: 0.05, w: 3.3, h: 2.45,
      chartColors: c.series.map(x => x.color), lineSize: 1.25,
      lineDataSymbol: "circle", lineDataSymbolSize: 4, lineDataSymbolLineSize: 0.75,
      showTitle: true, title: c.title, titleFontSize: 7, titleColor: "0B0B0B", titleFontFace: "Arial",
      showLegend: true, legendPos: "b", legendFontSize: 5.5, legendFontFace: "Arial", legendColor: "0B0B0B",
      catAxisLabelColor: "6B6A64", catAxisLabelFontSize: 6.5, catAxisLabelFontFace: "Arial",
      valAxisLabelColor: "6B6A64", valAxisLabelFontSize: 6.5, valAxisLabelFontFace: "Arial",
      valAxisMinVal: 0, valAxisMaxVal: c.ymax, valAxisMajorUnit: c.ymax > 40 ? 10 : 5,
      showValAxisTitle: true, valAxisTitle: spec.ylabel, valAxisTitleFontSize: 6.5,
      valAxisTitleColor: "0B0B0B", valAxisTitleFontFace: "Arial",
      valGridLine: { color: "E1E0D9", size: 0.5 }, catGridLine: { style: "none" },
    });
    s.addNotes(c.notes);
  }
  await pres.writeFile({ fileName: spec.out });
})();
"""


def find_node_modules():
    cands = [os.environ.get("PPTXGENJS_DIR"), HERE, os.path.join(HERE, "..", "..", "..", "tools", "pptxgen"),
             "/home/taeho/CVPR2027/tools/pptxgen", os.getcwd()]
    for c in cands:
        if not c:
            continue
        c = os.path.abspath(c)
        nm = c if os.path.basename(c) == "node_modules" else os.path.join(c, "node_modules")
        if os.path.isdir(os.path.join(nm, "pptxgenjs")):
            return nm
    sys.exit("pptxgenjs를 찾지 못했다. PPTXGENJS_DIR=<node_modules가 든 폴더>를 주거나 npm install pptxgenjs")


# ---- 차트 XML 후처리: 계열별 선 모양, 흰 테두리 마커, 속 빈 점, 값 라벨, 범례 위치 ----
TXT = lambda sz, col: (f'<c:txPr><a:bodyPr/><a:lstStyle/><a:p><a:pPr><a:defRPr sz="{sz}"><a:solidFill><a:srgbClr val="{col}"/>'
                       f'</a:solidFill><a:latin typeface="Arial"/></a:defRPr></a:pPr><a:endParaRPr lang="en-US"/></a:p></c:txPr>')
FLAGS = lambda val, ser: (f'<c:showLegendKey val="0"/><c:showVal val="{val}"/><c:showCatName val="0"/><c:showSerName val="{ser}"/>'
                          f'<c:showPercent val="0"/><c:showBubbleSize val="0"/>')
NOLBL = '<c:dLbls>' + FLAGS(0, 0) + '</c:dLbls>'
ML = lambda x, y, w=None, h=None, inner=False: ('<c:layout><c:manualLayout>' + ('<c:layoutTarget val="inner"/>' if inner else '') +
     f'<c:xMode val="edge"/><c:yMode val="edge"/><c:x val="{x}"/><c:y val="{y}"/>' +
     (f'<c:w val="{w}"/><c:h val="{h}"/>' if w else '') + '</c:manualLayout></c:layout>')


def lbl(i, pos, val, ser, col=INK, fmt="0.0"):
    return (f'<c:dLbl><c:idx val="{i}"/><c:numFmt formatCode="{fmt}" sourceLinked="0"/><c:spPr><a:noFill/><a:ln><a:noFill/></a:ln>'
            f'</c:spPr>{TXT(620, col)}<c:dLblPos val="{pos}"/>{FLAGS(val, ser)}</c:dLbl>')


def marker(col, kind):
    if kind == "none":
        return '<c:marker><c:symbol val="none"/></c:marker>'
    fill, edge, w = (col, "FFFFFF", 9525) if kind == "fill" else ("FFFFFF", col, 11430)
    return (f'<c:marker><c:symbol val="circle"/><c:size val="4"/><c:spPr><a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
            f'<a:ln w="{w}"><a:solidFill><a:srgbClr val="{edge}"/></a:solidFill></a:ln></c:spPr></c:marker>')


def fix_series(ser, st):
    col, w, dash, kind, hollow, dl = st["color"], st["w"], st["dash"], st["marker"], st["hollow"], st["dl"]
    ser = ser.replace('<c:invertIfNegative val="0"/>', '')
    ser = re.sub(r'<c:dLbls>.*?</c:dLbls>', '', ser, count=1, flags=re.S)
    ln = (f'<a:ln w="{w}" cap="flat"><a:solidFill><a:srgbClr val="{col}"/></a:solidFill><a:prstDash val="{dash}"/><a:round/></a:ln>'
          if w else '<a:ln><a:noFill/></a:ln>')
    ser = re.sub(r'<c:spPr>.*?</c:spPr>', f'<c:spPr>{ln}</c:spPr>', ser, count=1, flags=re.S)
    dpts = ''.join(f'<c:dPt><c:idx val="{i}"/>{marker(col, "hollow")}<c:bubble3D val="0"/></c:dPt>' for i in hollow)
    ser = re.sub(r'<c:marker>.*?</c:marker>', lambda _: marker(col, kind) + dpts + dl, ser, count=1, flags=re.S)
    if st.get("empty"):
        ser = re.sub(r'(<c:ptCount val="\d+"/>)(?:<c:pt idx="\d+"><c:v></c:v></c:pt>)+', r'\1', ser)
    return ser


def post(src, dst, charts):
    z = zipfile.ZipFile(src)
    out = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
    names = sorted((n for n in z.namelist() if re.fullmatch(r"ppt/charts/chart\d+\.xml", n)),
                   key=lambda n: int(re.search(r"(\d+)\.xml", n).group(1)))
    assert len(names) == len(charts), (names, len(charts))
    styles = dict(zip(names, charts))
    for it in z.infolist():
        if it.filename.endswith("/"):
            continue
        data = z.read(it.filename)
        if it.filename in styles:
            c = styles[it.filename]
            x = data.decode()
            sers = re.findall(r'<c:ser>.*?</c:ser>', x, re.S)
            assert len(sers) == len(c["styles"]), it.filename
            first = sers[0]
            assert c["check"] in first, (it.filename, c["check"])   # pptxgenjs 차트 번호 = 슬라이드 순서 확인
            for s, st in zip(sers, c["styles"]):
                x = x.replace(s, fix_series(s, st), 1)
            t0, t1 = x.index('<c:title>'), x.index('</c:title>')
            x = x[:t0] + x[t0:t1].replace('<c:layout/>', ML(0.13, 0.015)) + x[t1:]
            x = x.replace('<c:plotArea><c:layout/>', '<c:plotArea>' + ML(0.15, 0.11, 0.82, 0.62, inner=True), 1)
            hide = ''.join(f'<c:legendEntry><c:idx val="{i}"/><c:delete val="1"/></c:legendEntry>' for i in c["hide_legend"])
            leg = ('<c:legend><c:legendPos val="b"/>' + hide + ML(*c["legend"]) +
                   '<c:overlay val="0"/><c:spPr><a:noFill/><a:ln><a:noFill/></a:ln></c:spPr>' + TXT(560, INK) + '</c:legend>')
            x = re.sub(r'<c:legend>.*?</c:legend>', leg, x, count=1, flags=re.S)
            x = re.sub(r'<c:dispBlanksAs val="\w+"/>', '<c:dispBlanksAs val="gap"/>', x)
            data = x.encode()
        out.writestr(it, data)
    out.close()


def build(models, metric, out):
    ap = metric == "lvis_ap"
    spec = dict(labels=BITS, out=None, title=f"{'LVIS AP' if ap else 'LVIS flip'} by activation bits (seed 0)",
                ylabel="LVIS AP (deployment vocabulary)" if ap else "LVIS flip (%)", charts=[])
    post_charts = []
    for m in models:
        vals = [v for k, *_ in SERIES for v in m["ser"][k] if v is not None] + ([m["fp"]] if m["fp"] else [])
        ymax = 100 if not ap else int(math.ceil((max(vals) + 3) / 5.0) * 5)
        series = [dict(name=n, values=m["ser"][k], color=c) for k, n, c, *_ in SERIES]
        styles = [dict(color=c, w=w, dash=d, marker="fill", hollow=m["hollow"][k], dl=NOLBL) for k, _, c, w, d, _ in SERIES]
        ours = m["ser"]["ours"]
        # AP: 위(FP 점선과 2 AP 안이면 오른쪽). flip: Ours가 가장 아래라 아래
        lab = lambda i, v: ("b" if not ap else "r" if (m["fp"] is not None and v is not None and 0 < i and m["fp"] - v < 2.0) else "t")
        styles[0]["dl"] = '<c:dLbls>' + ''.join(lbl(i, lab(i, v), 1, 0) for i, v in enumerate(ours) if v is not None) + FLAGS(0, 0) + '</c:dLbls>'
        hide = []
        if ap:
            series.append(dict(name=f"FP32 {m['fp']:.1f}", values=[m["fp"]] * len(BITS), color=MUTED))
            styles.append(dict(color=MUTED, w=12700, dash="dash", marker="none", hollow=[],
                               dl='<c:dLbls>' + lbl(len(BITS) - 1, "t", 0, 1, MUTED) + FLAGS(0, 0) + '</c:dLbls>'))
            hide.append(len(series) - 1)
            series.append(dict(name=f"baseline collapsed (AP < {COLLAPSE:.0f})", values=[None] * len(BITS), color=MUTED))
            styles.append(dict(color=MUTED, w=None, dash="solid", marker="hollow", hollow=[], dl=NOLBL, empty=True))
        cfg = ", ".join(f"{b} {m['cfg'][b]}" for b in BITS)
        notes = (f"{m['model']}, seed 0, bits_curve_models.csv. Ours 구성(BRECQ seed 0 사전 검사, QDrop/PD-Quant에도 같이 씀): {cfg}. "
                 + (f"속 빈 점 = 기준선 LVIS AP < {COLLAPSE:.0f}. FP32 = {m['fp']:.2f}." if ap else ""))
        spec["charts"].append(dict(title=f"{m['model']} (seed 0)", series=series, ymax=ymax, notes=notes))
        post_charts.append(dict(styles=styles, hide_legend=hide, check=SERIES[0][1],
                                legend=(0.03, 0.83, 0.95, 0.16)))
    nm = find_node_modules()
    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, "raw.pptx")
        spec["out"] = raw
        sp, js = os.path.join(td, "spec.json"), os.path.join(td, "make.js")
        json.dump(spec, open(sp, "w"))
        open(js, "w").write(JS)
        env = dict(os.environ, NODE_PATH=nm)
        subprocess.run(["node", js, sp], check=True, env=env)
        post(raw, out, post_charts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(HERE, "bits_curve_models.csv"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--models", default=None, help="쉼표로 구분한 모델 이름(csv의 model 열). 기본: 전부, csv 순서")
    ap.add_argument("--metric", default="lvis_ap", choices=["lvis_ap", "lvis_flip"])
    a = ap.parse_args()
    models = load(a.csv, a.metric)
    if a.models:
        want = [s.strip() for s in a.models.split(",")]
        have = {m["model"]: m for m in models}
        bad = [w for w in want if w not in have]
        if bad:
            sys.exit(f"csv에 없는 모델: {bad}. 있는 모델: {list(have)}")
        models = [have[w] for w in want]
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.csv)),
                                "bits_curve_models.pptx" if a.metric == "lvis_ap" else "bits_curve_models_flip.pptx")
    build(models, a.metric, out)
    print(f"{out}: {len(models)}장 ({', '.join(m['model'] for m in models)})")


if __name__ == "__main__":
    main()
