// TSM numeric example v4 (11-17): grouped horizontal layout from the user's sketch.
//   [ box: max|x_c| over x ]  [ box C(g): tied max over W^(1) W^(2) ]  s   ->   x/s, W^(j)diag(s) stacked on the right
// The top row (two statistics and s) is one line: the two statistics give s; s is then applied on the right.
// Usage (repo root): NODE_PATH=tools/pptxgen/node_modules node documents/figure/method_overview_v3/tsm/make_tsm_numeric_v4.js
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

// ---- style (inches) ----
const F = "Arial", INK = "222222", MUT = "6B6A64", GR = "9A9A9A", ACC = "B03A2E", BOX = "A8A8A8", HL = 1;
const cs = 0.34, MW = 4 * cs, MH = 2 * cs, PAD = 0.14, LBL = 0.24, GAP = 0.22;

const pres = new pptxgen();
const SW = 9.45, SH = 3.6;
pres.defineLayout({ name: "TSM4", width: SW, height: SH });
pres.layout = "TSM4";
pres.title = "TSM numeric example v4";
const sl = pres.addSlide();
sl.background = { color: "FFFFFF" };

// ---- helpers ----
const hex = c => c.map(v => Math.round(v * 255).toString(16).padStart(2, "0")).join("").toUpperCase();
function heat(v, vmax) {
  const t = Math.min(Math.abs(v) / vmax, 1), c0 = [0.95, 0.94, 0.91], c1 = [0.96, 0.80, 0.45], c2 = [0.80, 0.36, 0.45];
  const lerp = (a, b, u) => a.map((x, i) => x + (b[i] - x) * u);
  return hex(t < 0.5 ? lerp(c0, c1, t / 0.5) : lerp(c1, c2, (t - 0.5) / 0.5));
}
const fmt = v => (Number.isInteger(v) ? String(v) : String(+v.toFixed(2)));
function math(src, sz, color = INK) {   // _{..} subscript, ^{..} superscript, *..* italic
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
function label(src, x, y, w, sz, color = INK, align = "left") {   // text box whose top-left is (x, y)
  sl.addText(math(src, sz, color), { x, y, w, h: LBL, margin: 0, align, valign: "bottom", isTextBox: true });
}
function matrix(M, x, y, vmax, name, dashed = false) {
  M.forEach((row, i) => row.forEach((v, j) => sl.addText(fmt(v), {
    x: x + j * cs, y: y + i * cs, w: cs, h: cs, margin: 0, align: "center", valign: "middle",
    fontFace: F, fontSize: 9, color: INK, fill: { color: heat(v, vmax) }, line: { color: "FFFFFF", width: 1.2 },
    isTextBox: true, objectName: `${name}_${i}_${j}`,
  })));
  sl.addShape(pres.shapes.RECTANGLE, { x, y, w: M[0].length * cs, h: M.length * cs, fill: { type: "none" },
    line: { color: "7A7A7A", width: 0.9, dashType: dashed ? "dash" : "solid" }, objectName: `${name}_frame` });
  sl.addShape(pres.shapes.RECTANGLE, { x: x + HL * cs, y, w: cs, h: M.length * cs, fill: { type: "none" },
    line: { color: ACC, width: 1.8 }, objectName: `${name}_highlight` });
}
function group(x, y, w, h, name) {
  sl.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.05, fill: { color: "FAFAF8" },
    line: { color: BOX, width: 1, dashType: "dash" }, objectName: name });
}
function seg(xa, ya, xb, yb, color, width = 1, arrow = false) {
  sl.addShape(pres.shapes.LINE, { x: Math.min(xa, xb), y: Math.min(ya, yb), w: Math.abs(xb - xa), h: Math.abs(yb - ya),
    flipH: xb < xa, flipV: yb < ya, line: { color, width, endArrowType: arrow ? "triangle" : undefined } });
}

// ---- layout ----
const top = 0.3;
const yStatL = top + PAD, yStat = yStatL + LBL;          // statistics row (label, then cells)
const yMatL = yStat + cs + 0.16, yMat = yMatL + LBL;     // input matrices (label, then cells)
const gH = (yMat + MH + PAD) - top;                       // group box height

// group A: x and its statistic
const ax = 0.3, aW = MW + 2 * PAD;
group(ax, top, aW, gH, "group_x");
label("max|*x*_{*c*}|", ax + PAD, yStatL, MW, 9, MUT); matrix([xmax], ax + PAD, yStat, 16, "xmax", true);
label("*x*", ax + PAD, yMatL, MW, 10.5); matrix(X, ax + PAD, yMat, 16, "x");

// group B: consumers C(g) and their tied statistic
const bx = ax + aW + GAP, bW = 2 * MW + 0.24 + 2 * PAD;
group(bx, top, bW, gH, "group_W");
const w1x = bx + PAD, w2x = w1x + MW + 0.24, wsx = bx + (bW - MW) / 2;
label("max_{*j*} max|*W*^{(*j*)}_{:,*c*}|", wsx - 0.5, yStatL - 0.03, MW + 1.0, 9, MUT, "center"); matrix([wt], wsx, yStat, 4, "wmax", true);
label("*W*^{(1)}", w1x, yMatL, MW, 10.5); matrix(W1, w1x, yMat, 4, "W1");
label("*W*^{(2)}", w2x, yMatL, MW, 10.5); matrix(W2, w2x, yMat, 4, "W2");
label("C(*g*)", bx + PAD, yStatL, 0.6, 10, ACC, "left");

// s: on the statistics row, right of the two groups
// outer box: both statistics are inputs to s
sl.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: ax - 0.1, y: top - 0.1, w: bx + bW - ax + 0.2, h: gH + 0.2, rectRadius: 0.07,
  fill: { type: "none" }, line: { color: "C4C4C4", width: 0.9, dashType: "dash" }, objectName: "group_all" });
const sxL = bx + bW + 0.5;
seg(bx + bW + 0.12, yStat + cs / 2, sxL - 0.05, yStat + cs / 2, GR, 1, true);
label("*s*", sxL, yStatL - 0.05, MW, 11, ACC, "center"); matrix([s], sxL, yStat, 4, "s", true);
label("*s*_{*c*}(*α*)", sxL, yStat + cs + 0.02, MW, 8.5, MUT, "center");

// outputs: stacked on the right
const ox = sxL + MW + 0.62, oGap = 0.12;
const o1 = yStat + cs / 2 - MH / 2;                      // x/s centred on the s row
const o2 = o1 + MH + LBL + oGap, o3 = o2 + MH + LBL + oGap;
label("*x*/*s*", ox, o1 - LBL, MW, 10.5); matrix(Xh, ox, o1, 16, "xs");
label("*W*^{(1)}diag(*s*)", ox, o2 - LBL, MW + 0.3, 10.5); matrix(W1h, ox, o2, 8, "W1s");
label("*W*^{(2)}diag(*s*)", ox, o3 - LBL, MW + 0.3, 10.5); matrix(W2h, ox, o3, 8, "W2s");

// s -> outputs: split line, "÷s" to x/s, "diag(s)" to the weight bracket
const split = sxL + MW + 0.22, yS = yStat + cs / 2, yWmid = (o2 + o3 + MH) / 2;
seg(sxL + MW + 0.03, yS, split, yS, ACC, 1.25);
seg(split, yS, ox - 0.05, yS, ACC, 1.25, true);
sl.addText(math("÷*s*", 9.5, ACC), { x: split, y: yS - LBL - 0.01, w: ox - split - 0.05, h: LBL, margin: 0, align: "center", valign: "bottom", isTextBox: true });
seg(split, yS, split, yWmid, ACC, 1.25);
const brx = ox - 0.1;
seg(split, yWmid, brx - 0.06, yWmid, ACC, 1.25, true);
seg(brx, o2, brx + 0.06, o2, ACC); seg(brx, o2, brx, o3 + MH, ACC); seg(brx, o3 + MH, brx + 0.06, o3 + MH, ACC);
sl.addText(math("diag(*s*)", 9, ACC), { x: split - 0.75, y: yWmid + 0.03, w: 0.72, h: LBL, margin: 0, align: "right", valign: "top", isTextBox: true });

pres.writeFile({ fileName: path.join(HERE, "tsm_numeric_v4.pptx") }).then(f => console.log("wrote", f));
