// TSM numeric example (v3), editable PowerPoint version of tsm_numeric_v3.png (11-17).
// Same layout as tsm_numeric_v3.py: inputs -> tied statistics -> s -> outputs, with orthogonal arrows.
// Usage (repo root): NODE_PATH=tools/pptxgen/node_modules node documents/figure/method_overview_v3/tsm/make_tsm_numeric_v3.js
const pptxgen = require("pptxgenjs");
const path = require("path");
const HERE = __dirname;

// ---- data (alpha = 0.5) ----
const X = [[1, -16, 2, 6], [-2, 8, -1, -9]];
const W1 = [[2, 1, -2, 1], [1, -1, -1, -1]], W2 = [[1, 4, 2, -1], [-1, -2, 1, 1]];
const colAbsMax = M => M[0].map((_, j) => Math.max(...M.map(r => Math.abs(r[j]))));
const xmax = colAbsMax(X), wt = colAbsMax(W1).map((v, j) => Math.max(v, colAbsMax(W2)[j]));
const s = xmax.map((v, j) => Math.sqrt(v / wt[j]));
const div = (M, v) => M.map(r => r.map((x, j) => x / v[j])), mul = (M, v) => M.map(r => r.map((x, j) => x * v[j]));
const Xh = div(X, s), W1h = mul(W1, s), W2h = mul(W2, s);

// ---- style ----
const F = "Arial", INK = "222222", MUT = "6B6A64", GR = "8A8A8A", ACC = "B03A2E", HL = 1;
const K = 0.78;                       // inches per figure unit (matches the matplotlib layout)
const cs = 0.42, Wm = 4 * cs, Hm = 2 * cs;
const X0 = -0.75, Y0 = 3.75;          // figure-unit origin of the slide's top-left corner
const PX = u => (u - X0) * K, PY = u => (Y0 - u) * K;

const x0 = 0.0, x1 = x0 + Wm + 0.55, mx = x1 + Wm + 0.2, x2 = mx + 0.38, sx = x2 + Wm + 0.2, x3 = sx + 0.5;
const yXt = 3.0, yX = yXt - Hm / 2, yW1t = 1.72, yW2t = 0.42, yW = (yW1t + (yW2t - Hm)) / 2, yS = (yX + yW) / 2;
const SW = PX(x3 + Wm + 0.1), SH = PY(-0.45);

const pres = new pptxgen();
pres.defineLayout({ name: "TSM", width: SW, height: SH });
pres.layout = "TSM";
pres.title = "TSM numeric example v3";
const sl = pres.addSlide();
sl.background = { color: "FFFFFF" };

// ---- helpers ----
function hex(c) { return c.map(v => Math.round(v * 255).toString(16).padStart(2, "0")).join("").toUpperCase(); }
function heat(v, vmax) {
  const t = Math.min(Math.abs(v) / vmax, 1), c0 = [0.95, 0.94, 0.91], c1 = [0.96, 0.80, 0.45], c2 = [0.80, 0.36, 0.45];
  const lerp = (a, b, u) => a.map((x, i) => x + (b[i] - x) * u);
  return hex(t < 0.5 ? lerp(c0, c1, t / 0.5) : lerp(c1, c2, (t - 0.5) / 0.5));
}
const fmt = v => (Number.isInteger(v) ? String(v) : String(+v.toFixed(2)));
// math markup: _{..} subscript, ^{..} superscript, *..* italic
function math(src, sz, color = INK) {
  const runs = []; let i = 0, ital = false, buf = "";
  const push = (t, o = {}) => { if (t) runs.push({ text: t, options: Object.assign({ fontFace: F, fontSize: sz, color, italic: ital }, o) }); };
  while (i < src.length) {
    const ch = src[i];
    if ((ch === "_" || ch === "^") && src[i + 1] === "{") {
      push(buf); buf = ""; let d = 1, j = i + 2;
      while (j < src.length && d) { if (src[j] === "{") d++; if (src[j] === "}") d--; j++; }
      src.slice(i + 2, j - 1).split(/(\*[^*]+\*)/).forEach(p => {
        if (!p) return; const it = p.length > 2 && p.startsWith("*") && p.endsWith("*");
        push(it ? p.slice(1, -1) : p, { [ch === "_" ? "subscript" : "superscript"]: true, italic: it });
      });
      i = j; continue;
    }
    if (ch === "*") { push(buf); buf = ""; ital = !ital; i++; continue; }
    buf += ch; i++;
  }
  push(buf); return runs;
}
// label centred at (ux, uy) in figure units; va: "b" = text sits above uy, "t" = below, "m" = centred
function label(src, ux, uy, sz, color = INK, { ha = "c", va = "b", w = 2.2, bold = false } = {}) {
  const h = 0.3, wi = w * K;
  const x = ha === "c" ? PX(ux) - wi / 2 : ha === "l" ? PX(ux) : PX(ux) - wi;
  const y = va === "b" ? PY(uy) - h : va === "t" ? PY(uy) : PY(uy) - h / 2;
  const runs = math(src, sz, color); if (bold) runs.forEach(r => (r.options.bold = true));
  sl.addText(runs, { x, y, w: wi, h, margin: 0, align: { c: "center", l: "left", r: "right" }[ha], valign: { b: "bottom", t: "top", m: "middle" }[va], isTextBox: true });
}
function matrix(M, ux, utop, vmax, dashed = false, name = "m") {
  M.forEach((row, i) => row.forEach((v, j) => {
    sl.addText(fmt(v), {
      x: PX(ux + j * cs), y: PY(utop - i * cs), w: cs * K, h: cs * K, margin: 0, align: "center", valign: "middle",
      fontFace: F, fontSize: 9, color: INK, fill: { color: heat(v, vmax) }, line: { color: "FFFFFF", width: 1.2 },
      isTextBox: true, objectName: `${name}_${i}_${j}`,
    });
  }));
  sl.addShape(pres.shapes.RECTANGLE, { x: PX(ux), y: PY(utop), w: M[0].length * cs * K, h: M.length * cs * K,
    fill: { type: "none" }, line: { color: "777777", width: 0.9, dashType: dashed ? "dash" : "solid" }, objectName: `${name}_frame` });
  sl.addShape(pres.shapes.RECTANGLE, { x: PX(ux + HL * cs), y: PY(utop), w: cs * K, h: M.length * cs * K,
    fill: { type: "none" }, line: { color: ACC, width: 1.8 }, objectName: `${name}_highlight` });
}
function seg(a, b, color, width = 1, arrow = false) {   // axis-aligned segment in figure units
  const xa = PX(a[0]), ya = PY(a[1]), xb = PX(b[0]), yb = PY(b[1]);
  sl.addShape(pres.shapes.LINE, { x: Math.min(xa, xb), y: Math.min(ya, yb), w: Math.abs(xb - xa), h: Math.abs(yb - ya),
    flipH: xb < xa, flipV: yb < ya, line: { color, width, endArrowType: arrow ? "triangle" : undefined } });
}
function bracket(ux, y0, y1, side) {   // side=+1: opens left, -1: opens right
  seg([ux - side * 0.07, y0], [ux, y0], ACC); seg([ux, y0], [ux, y1], ACC); seg([ux, y1], [ux - side * 0.07, y1], ACC);
}

// ---- column 1: inputs ----
matrix(X, x0, yXt, 16, false, "x"); label("*x*", x0 + Wm / 2, yXt + 0.06, 10);
matrix(W1, x0, yW1t, 4, false, "W1"); label("*W*^{(1)}", x0 + Wm / 2, yW1t + 0.06, 10);
matrix(W2, x0, yW2t, 4, false, "W2"); label("*W*^{(2)}", x0 + Wm / 2, yW2t + 0.06, 10);
bracket(x0 + Wm + 0.1, yW1t, yW2t - Hm, +1); label("C(*g*)", x0 - 0.08, yW, 9.5, ACC, { ha: "r", va: "m", w: 0.8 });
// ---- column 2: tied statistics ----
matrix([xmax], x1, yX + cs / 2, 16, true, "xmax"); label("max|*x*_{*c*}|", x1 + Wm / 2, yX + cs / 2 + 0.06, 8.5, MUT);
matrix([wt], x1, yW + cs / 2, 4, true, "wmax"); label("max_{*j*} max|*W*^{(*j*)}_{:,*c*}|", x1 + Wm / 2, yW + cs / 2 + 0.06, 8.5, MUT, { w: 2.6 });
seg([x0 + Wm + 0.05, yX], [x1 - 0.04, yX], GR, 1, true);
seg([x0 + Wm + 0.12, yW], [x1 - 0.04, yW], GR, 1, true);
// ---- column 3: s (merge) ----
seg([x1 + Wm + 0.04, yX], [mx, yX], GR); seg([mx, yX], [mx, yW], GR); seg([x1 + Wm + 0.04, yW], [mx, yW], GR);
seg([mx, yS], [x2 - 0.04, yS], GR, 1, true);
matrix([s], x2, yS + cs / 2, 4, true, "s"); label("*s*", x2 + Wm / 2, yS + cs / 2 + 0.06, 11, ACC);
label("*s*_{*c*}(*α*)", x2 + Wm / 2, yS - cs / 2 - 0.08, 8.5, MUT, { va: "t" });
// ---- column 4: outputs (split) ----
seg([x2 + Wm + 0.04, yS], [sx, yS], ACC, 1.25); seg([sx, yX], [sx, yW], ACC, 1.25);
seg([sx, yX], [x3 - 0.04, yX], ACC, 1.25, true); label("÷*s*", (sx + x3) / 2, yX + 0.05, 9.5, ACC, { w: 0.6 });
seg([sx, yW], [x3 - 0.18, yW], ACC, 1.25, true); label("diag(*s*)", sx - 0.06, yW - 0.08, 9, ACC, { ha: "r", va: "t", w: 0.9 });
bracket(x3 - 0.1, yW1t, yW2t - Hm, -1);
matrix(Xh, x3, yXt, 16, false, "xs"); label("*x*/*s*", x3 + Wm / 2, yXt + 0.06, 10);
matrix(W1h, x3, yW1t, 8, false, "W1s"); label("*W*^{(1)}diag(*s*)", x3 + Wm / 2, yW1t + 0.06, 10);
matrix(W2h, x3, yW2t, 8, false, "W2s"); label("*W*^{(2)}diag(*s*)", x3 + Wm / 2, yW2t + 0.06, 10);
// ---- titles ----
label("Tied statistics", (x0 + x1 + Wm) / 2, 3.42, 11, INK, { bold: true, w: 3 });
label("TSM (shared *s*)", (x2 + x3 + Wm) / 2, 3.42, 11, INK, { bold: true, w: 3 });

pres.writeFile({ fileName: path.join(HERE, "tsm_numeric_v3.pptx") }).then(f => console.log("wrote", f));
