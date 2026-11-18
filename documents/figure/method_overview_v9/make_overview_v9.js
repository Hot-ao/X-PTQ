// Method overview v9 (11-17). Changes from v8 (user feedback):
//  - whitespace left by removed text is closed: narrower diagnosis panel (small Q boxes), shorter canvas, tighter SWP
//  - calibration input = image stack + stacked vocabulary cards (class-name list, as in GLIP Fig. 2 / YOLO-World Fig. 2-3)
//  - bar chart: omitted layers are simply left out (no "⋯", no white masks); equal bar pitch; wavy break on the y axis only
//  - TGR: a_h above the × node, + directly above ×, straight vertical arrow; Before/After both end in the same z
//  - SWP: "backbone blocks" label starts above block 0, legend centred below
//  - TSM check: flips over many FP-confident regions of D_chk for each candidate (rate, not a single example)
// Build (from this folder): NODE_PATH=../../../tools/pptxgen/node_modules node make_overview_v9.js && python3 merge_v9.py
const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");
const HERE = __dirname;
const DIAG = JSON.parse(fs.readFileSync(path.join(HERE, "diag_yolov8s-world.json"), "utf8"));
const EQ = { diag: 3, G: 6, Gq: 7, mig: 9, tie: 11, check: 13 };   // manuscript numbers (11-17)

const W = 9.76, H = 3.44;
const F = "Arial", INK = "222222", MUT = "6B6A64", ARW = "8A8A8A", FRM = "777777", ACC = "B03A2E";
const FUS = "CC5C73", FUSL = "F6DCE2", PRT = "E3A33B", OK = "3B7D5B", OKL = "E3F0E8", OFF = "D9D9D9";

const pres = new pptxgen();
pres.defineLayout({ name: "OV9", width: W, height: H });
pres.layout = "OV9";
pres.title = "DeFiQ method overview v9";
const s = pres.addSlide();
s.background = { color: "FFFFFF" };

// ---- text: _{..} subscript, ^{..} superscript, *..* italic ----
function math(src, sz, color = INK, base = {}) {
  const runs = []; let i = 0, ital = false, buf = "";
  const push = (t, o = {}) => { if (t) runs.push({ text: t, options: Object.assign({ fontFace: F, fontSize: sz, color, italic: ital }, base, o) }); };
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
function mathLines(lines, sz, color = INK) {
  const out = [];
  lines.forEach((l, i) => { const r = math(l, sz, color); if (i < lines.length - 1) r[r.length - 1].options.breakLine = true; out.push(...r); });
  return out;
}
const txt = (t, sz, o = {}) => [{ text: t, options: Object.assign({ fontFace: F, fontSize: sz, color: INK }, o) }];
function label(x, y, w, h, runs, o = {}) {   // free text (axis labels, tags)
  s.addText(runs, Object.assign({ x, y, w, h, margin: 0, align: "center", valign: "middle", isTextBox: true, fit: "none" }, o));
}
// a box whose text is part of the shape itself
function box(x, y, w, h, runs, o = {}) {
  s.addText(runs, Object.assign({
    shape: o.shape || (o.round ? pres.shapes.ROUNDED_RECTANGLE : pres.shapes.RECTANGLE), x, y, w, h, margin: 0,
    align: "center", valign: o.valign || "middle", fit: "none",
    fill: o.fill === null ? { type: "none" } : { color: o.fill || "FFFFFF" },
    line: o.line === null ? { type: "none" } : { color: o.line || FRM, width: o.lw || 0.9, dashType: o.dash || "solid" },
  }, o.round ? { rectRadius: o.r || 0.04 } : {}));
}
const rect = (x, y, w, h, o = {}) => box(x, y, w, h, "", Object.assign({ fill: null }, o));
const circ = (cx, cy, d, t, col = FRM) => box(cx - d / 2, cy - d / 2, d, d, txt(t, 7.5, { bold: true, color: col === FRM ? INK : col }), { shape: pres.shapes.OVAL, line: col });
// one polyline (absolute inch coordinates); arrowhead at the end unless head === false
function line(pts, o = {}) {
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const x0 = Math.min(...xs), y0 = Math.min(...ys), w = Math.max(Math.max(...xs) - x0, 0.001), h = Math.max(Math.max(...ys) - y0, 0.001);
  s.addShape(pres.shapes.CUSTOM_GEOMETRY, {
    x: x0, y: y0, w, h, fill: { type: "none" },
    points: pts.map(([x, y]) => ({ x: x - x0, y: y - y0 })),
    line: { color: o.color || ARW, width: o.lw || 1.0, dashType: o.dash || "solid", endArrowType: o.head === false ? undefined : "triangle" },
  });
}
// wavy break mark centred at (cx, cy), width w
function wave(cx, cy, w, o = {}) {
  const n = 16, a = 0.018, pts = [];
  for (let k = 0; k <= n; k++) pts.push([cx - w / 2 + (w * k) / n, cy + a * Math.sin((2 * Math.PI * k) / n * 1.5)]);
  line(pts, { head: false, color: o.color || FRM, lw: o.lw || 0.7 });
}
function panel(x, y, w, h, title, sec, eq) {
  rect(x, y, w, h, { line: "A8A8A8", lw: 1.2, dash: "lgDash", round: true, r: 0.05 });
  label(x + 0.12, y + 0.05, w - 0.24, 0.2, [
    { text: title, options: { fontFace: F, fontSize: 9, bold: true, color: INK } },
    { text: "  " + sec, options: { fontFace: F, fontSize: 7, color: MUT } },
  ], { align: "left" });
  if (eq) label(x + w - 1.62, y + 0.05, 1.5, 0.2, txt(eq, 7, { color: MUT }), { align: "right" });
}
function img(x, y, w, h, f) { s.addImage({ path: path.join(HERE, "img", f), x, y, w, h }); rect(x, y, w, h, { line: "FFFFFF", lw: 0.75 }); }
const sq = (x, y, c, d = 0.085) => box(x, y, d, d, "", { fill: c, line: null });
const PH = 3.32;                                   // panel height (TSM panel sets it)

// =====================================================================
// (a) layer-wise diagnosis §3.2
const A = { x: 0.06, y: 0.06, w: 2.3, h: PH };
panel(A.x, A.y, A.w, A.h, "Layer-wise diagnosis", "§3.2", `Eq. (${EQ.diag})`);
// calibration input: images + vocabulary cards
const tY = A.y + 0.34;
img(A.x + 0.2, tY, 0.46, 0.34, "cal0.jpg");
img(A.x + 0.3, tY + 0.07, 0.46, 0.34, "cal1.jpg");
label(A.x + 0.2, tY + 0.43, 0.56, 0.14, math("*D*_{cal}", 7.5), { align: "center" });
const vX = A.x + 1.18, vW = 0.62, vH = 0.38;
[2, 1].forEach(k => rect(vX + k * 0.05, tY + 0.03 - k * 0.03 + 0.02, vW, vH, { fill: "FFFFFF", line: "B5B5B5", lw: 0.6, round: true, r: 0.03 }));
box(vX, tY + 0.05, vW, vH, mathLines(["person. bicycle.", "car. … toothbrush"], 6, INK), { line: FRM, lw: 0.7, round: true, r: 0.03 });
label(vX, tY + 0.45, vW + 0.1, 0.14, math("𝒱 (COCO, 80)", 7), { align: "center" });
// Q8 / Q8_{l->4} -> Eq. (3)
const qY = tY + 0.66, qH = 0.24, qW = 0.72;
const qx1 = A.x + 0.3, qx2 = A.x + A.w - 0.3 - qW;
box(qx1, qY, qW, qH, math("*Q*^{8}", 8.5));
box(qx2, qY, qW, qH, math("*Q*^{8}_{*ℓ*→4}", 8.5));
const fY = qY + qH + 0.18;
line([[qx1 + qW / 2, qY + qH], [qx1 + qW / 2, fY]]);
line([[qx2 + qW / 2, qY + qH], [qx2 + qW / 2, fY]]);
box(A.x + 0.12, fY, A.w - 0.24, 0.26, math("Δ_{*ℓ*} = Flip_{cal}(*Q*^{8}_{*ℓ*→4}) − Flip_{cal}(*Q*^{8})", 7.5));
// bars: informative layers only, equal pitch (others removed from the plot)
const keep = [...Array(14).keys()].concat([25, 26, 27, 31, 32, 41, 47]);
const ax = A.x + 0.36, base = A.y + PH - 0.2, pitch = 0.08;
const ph = base - (fY + 0.26 + 0.34);
const loMax = 6, loH = ph * 0.62, brkY = ph * 0.69, hiH0 = ph * 0.76, hiMin = 29, hiMax = 32;
const yOf = v => v <= loMax ? v / loMax * loH : hiH0 + (Math.min(v, hiMax) - hiMin) / (hiMax - hiMin) * (ph - hiH0);
const pw = keep.length * pitch + 0.06;
const bk = base - brkY;
line([[ax, base - ph - 0.03], [ax, bk - 0.035]], { head: false, color: FRM, lw: 0.6 });
line([[ax, bk + 0.035], [ax, base], [ax + pw, base]], { head: false, color: FRM, lw: 0.6 });
wave(ax, bk - 0.035, 0.1); wave(ax, bk + 0.035, 0.1);
[0, 2, 4, 6, 30].forEach(v => {
  line([[ax - 0.04, base - yOf(v)], [ax, base - yOf(v)]], { head: false, color: FRM, lw: 0.5 });
  label(ax - 0.3, base - yOf(v) - 0.07, 0.24, 0.14, txt(String(v), 6, { color: MUT }), { align: "right" });
});
label(ax - 0.3, base - ph - 0.19, 0.8, 0.14, math("Δ_{*ℓ*} (pp)", 6.5, MUT), { align: "left" });
keep.forEach((k, j) => {
  const r = DIAG[k], v = Math.max(r.delta, 0), hh = yOf(v);
  if (hh < 0.004) return;
  const col = r.kind === "fusion" ? FUS : r.kind === "swp" ? PRT : "C4C4C4";
  const bx = ax + 0.04 + j * pitch, bw = pitch * 0.74;
  if (v > loMax) {   // tall bar drawn in two pieces around the break
    box(bx, bk + 0.035, bw, brkY - 0.035, "", { fill: col, line: null });
    box(bx, base - hh, bw, hh - brkY - 0.035, "", { fill: col, line: null });
    wave(bx + bw / 2, bk - 0.035, 0.13); wave(bx + bw / 2, bk + 0.035, 0.13);
    label(bx + bw + 0.03, base - hh - 0.02, 0.6, 0.26, txt(`${r.name.replace(".conv", "")}\n${r.delta.toFixed(1)}`, 6.5, { color: FUS, bold: true }), { align: "left", valign: "top" });
  } else box(bx, base - hh, bw, hh, "", { fill: col, line: null });
});
label(ax, base + 0.03, pw, 0.13, math("layer *ℓ*", 6.5, MUT));
// legend (notation only), right-aligned on the y-label line
[[FUS, "*L*_{G}"], [PRT, "*S*^{*}"]].forEach(([c, t], k) => {
  const lx = A.x + A.w - 0.86 + k * 0.42, ly = base - ph - 0.19;
  sq(lx, ly + 0.03, c, 0.08);
  label(lx + 0.11, ly, 0.3, 0.14, math(t, 7), { align: "left" });
});

// =====================================================================
// (b) TGR §4.1 — before / after compute graph, both ending in the same z
const B = { x: A.x + A.w + 0.1, y: 0.06, w: 3.4, h: 1.84 };
panel(B.x, B.y, B.w, B.h, "Text-Gate Relocation (TGR)", "§4.1", `Eq. (${EQ.G}), (${EQ.Gq})`);
const bh = 0.22, d = 0.17;
const tag = (y, t, c) => label(B.x + 0.12, y - 0.08, 0.5, 0.16, txt(t, 7, { color: c, bold: true }), { align: "left" });
const ix = B.x + 0.62, iw = 0.3;
const inp = (y, src) => box(ix, y - bh / 2, iw, bh, math(src, 8));
const ahLabel = (cx, cy, col) => label(cx - 0.3, cy - d / 2 - 0.15, 0.27, 0.13, math("*a*_{*h*}", 7, col), { align: "right" });
// geometry of the after graph fixes the common output position
const qAx = ix + iw + 0.18, qaW = 0.42, wAx = qAx + qaW + 0.18, wAw = 0.82, gX = wAx + wAw + 0.24, zX = gX + d / 2 + 0.3;
// before
const b1 = B.y + 0.5, b2 = B.y + 0.82, bm = (b1 + b2) / 2;
tag(bm, "Before", ACC);
inp(b1, "*y*"); inp(b2, "*p*_{*h*}");
const gB = ix + iw + 0.3, cx = gB + 0.26, qBx = cx + 0.14;
circ(gB, b2, d, "×", ACC); ahLabel(gB, b2, ACC);
line([[ix + iw, b2], [gB - d / 2, b2]]);
line([[ix + iw, b1], [cx, b1], [cx, bm], [qBx, bm]]);
line([[gB + d / 2, b2], [cx, b2], [cx, bm]], { head: false });
box(qBx, bm - bh / 2, qaW, bh, math("*Q*_{*a*}", 8, ACC), { line: ACC, fill: "FBE9E7", lw: 1.2 });
const wBx = qBx + qaW + 0.18;
line([[qBx + qaW, bm], [wBx, bm]]);
box(wBx, bm - bh / 2, 0.62, bh, math("*Q*_{*w*}(*W*)", 8));
line([[wBx + 0.62, bm], [zX, bm]]);
label(zX + 0.03, bm - 0.09, 0.2, 0.18, math("*z*", 9), { align: "left" });
// separator
line([[B.x + 0.12, B.y + 1.04], [B.x + B.w - 0.12, B.y + 1.04]], { head: false, color: "D0D0D0", dash: "dash", lw: 0.6 });
// after: + directly above ×
const a1 = B.y + 1.3, a2 = B.y + 1.62;
tag((a1 + a2) / 2, "After", OK);
inp(a1, "*y*"); inp(a2, "*p*_{*h*}");
box(qAx, a1 - bh / 2, qaW, bh, math("*Q*_{*a*}", 8, OK), { line: OK, fill: OKL, lw: 1.2 });
box(qAx, a2 - bh / 2, qaW, bh, math("*Q*_{*a*}", 8, OK), { line: OK, fill: OKL, lw: 1.2 });
line([[ix + iw, a1], [qAx, a1]]); line([[ix + iw, a2], [qAx, a2]]);
box(wAx, a1 - bh / 2, wAw, bh, math("*Q*_{*w*}(*W*_{main})", 8));
box(wAx, a2 - bh / 2, wAw, bh, math("*Q*_{*w*}(*W*_{*h*})", 8));
line([[qAx + qaW, a1], [wAx, a1]]); line([[qAx + qaW, a2], [wAx, a2]]);
circ(gX, a2, d, "×", OK); ahLabel(gX, a2, OK);
circ(gX, a1, d, "+");
line([[wAx + wAw, a2], [gX - d / 2, a2]]);
line([[wAx + wAw, a1], [gX - d / 2, a1]]);
line([[gX, a2 - d / 2], [gX, a1 + d / 2]]);
line([[gX + d / 2, a1], [zX, a1]]);
label(zX + 0.03, a1 - 0.09, 0.2, 0.18, math("*z*", 9), { align: "left" });

// =====================================================================
// (c) SWP §4.2 — bit-widths on the backbone
const P = { x: B.x, y: B.y + B.h + 0.08, w: B.w, h: PH - B.h - 0.08 };
panel(P.x, P.y, P.w, P.h, "Selective Weight Protection (SWP)", "§4.2", "");
const blocks = {};
DIAG.forEach(r => { const b = r.name.split(".")[0]; (blocks[b] = blocks[b] || []).push(r); });
const show = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
const cs = 0.085, cg = 0.025, kw = 2 * cs + 3 * cg, kg = 0.03;
const nW = 2 * 0.11 + 3 * cg;
const rowW = 10 * kw + 9 * kg + 0.08 + nW;
const kX = P.x + (P.w - rowW) / 2, kY = P.y + 0.5, kh = 0.16 + 3 * (cs + cg) + 0.01;
label(kX, kY - 0.17, 2.5, 0.14, txt("YOLO-World-S backbone blocks", 6.5, { color: MUT }), { align: "left" });
show.forEach((b, k) => {
  const rs = blocks[b] || [], sel = rs.some(r => r.kind === "swp"), x0 = kX + k * (kw + kg);
  box(x0, kY, kw, kh, txt(b, 6.5, { color: MUT }), { line: sel ? PRT : FRM, lw: sel ? 1.2 : 0.6, round: true, r: 0.03, valign: "top" });
  const cl = b === "0" ? [{ kind: "proto" }] : rs;
  cl.slice(0, 6).forEach((r, j) => {
    const col = r.kind === "proto" ? "6E6E6E" : r.kind === "swp" ? PRT : OFF;
    sq(x0 + cg + (j % 2) * (cs + cg), kY + 0.16 + Math.floor(j / 2) * (cs + cg), col, cs);
  });
});
const nX = kX + 10 * kw + 9 * kg + 0.08;
box(nX, kY, nW, kh, math("*L*_{G}", 6.5, FUS), { fill: null, line: FUS, lw: 0.9, dash: "dash", round: true, r: 0.03, valign: "top" });
[0, 1, 2, 3].forEach(j => box(nX + cg + (j % 2) * (0.11 + cg), kY + 0.18 + Math.floor(j / 2) * (0.11 + cg), 0.11, 0.11, "", { fill: FUSL, line: FUS, lw: 0.5 }));
// legend, centred
const leg = [[PRT, math("*S*^{*} (W8)", 7), 0.62], [OFF, txt("W4", 7), 0.36], ["6E6E6E", txt("W8 (first layer)", 7), 0.95]];
const legW = leg.reduce((a, l) => a + l[2], 0) + 0.12 * (leg.length - 1);
let lx = P.x + (P.w - legW) / 2; const lY = kY + kh + 0.1;
leg.forEach(([c, t, w]) => { sq(lx, lY + 0.03, c, 0.08); label(lx + 0.12, lY, w, 0.14, t, { align: "left" }); lx += w + 0.12; });

// =====================================================================
// (d) TSM §4.3 — numeric example inserted by merge_v9.py (x = T.x + 0.12, y = T.y + 0.3, w = 3.5)
const T = { x: B.x + B.w + 0.1, y: 0.06, w: 3.74, h: PH };
panel(T.x, T.y, T.w, T.h, "Tied Scale Migration (TSM)", "§4.3", `Eq. (${EQ.mig}), (${EQ.tie})`);
const cT = T.y + 2.44;
line([[T.x + 0.12, cT], [T.x + T.w - 0.12, cT]], { head: false, color: "D0D0D0", dash: "dash", lw: 0.6 });
label(T.x + 0.12, cT + 0.03, 2.0, 0.18, txt("TSM check", 8, { bold: true }), { align: "left" });
label(T.x + T.w - 1.42, cT + 0.03, 1.3, 0.18, txt(`Eq. (${EQ.check})`, 7, { color: MUT }), { align: "right" });
// D_chk, then per candidate: FP-confident regions, flipped ones in red, and the resulting flip rate
const iX = T.x + 0.14, iY = cT + 0.27;
img(iX, iY, 0.42, 0.31, "chk0.jpg");
img(iX + 0.09, iY + 0.06, 0.42, 0.31, "chk2.jpg");
label(iX, iY + 0.39, 0.52, 0.13, math("*D*_{chk}", 7), { align: "center" });
const nC = 12, cz = 0.075, cgap = 0.012, pc = cz + cgap;
const flips = { on: [3, 9], off: [1, 4, 7, 10] };
const r1 = iY + 0.07, r2 = r1 + 0.24, rm = (r1 + r2) / 2, lX = iX + 0.6, gx0 = lX + 0.34;
const rowEnd = gx0 + nC * pc + 0.03, bX = rowEnd + 0.1, oX = iX + 2.8, oW = T.x + T.w - 0.14 - oX;
[["on", r1], ["off", r2]].forEach(([m, y]) => {
  label(lX, y - 0.09, 0.32, 0.18, math(`*θ*̂_{${m}}`, 7.5), { align: "right" });
  for (let j = 0; j < nC; j++) sq(gx0 + j * pc, y - cz / 2, flips[m].includes(j) ? FUS : "F3E7CB", cz);
  line([[rowEnd, y], [bX, y], [bX, rm]], { head: false });
});
// both candidates' flip rates are compared; the lower one becomes the output
line([[bX, rm], [oX, rm]]);
label(bX + 0.02, rm - 0.18, oX - bX - 0.04, 0.15, math("argmin_{ℳ}(Flip_{chk})", 6.5, MUT));
box(oX, rm - 0.12, oW, 0.24, math("*θ*̂_{ℳ}", 8), { line: ACC, lw: 1.2 });
label(gx0 + flips.on[0] * pc - 0.15, r1 - cz / 2 - 0.14, cz + 0.3, 0.13, txt("flip", 6.5, { color: FUS, bold: true }));

s.addNotes("v9 (11-17). Bars: YOLO-World-S Δℓ from configs/protect_cache; layers with small Δℓ removed. TSM check cells are schematic (16 FP-confident regions per candidate, flips in red).");
const OUT = process.argv[2] || path.join(HERE, "method_overview_v9_base.pptx");
pres.writeFile({ fileName: OUT }).then(() => console.log("written", OUT));
