// Method overview v8 (11-17). Changes from v7 (user feedback):
//  - every elbow line is ONE polyline shape (custom geometry), never separate segments
//  - text that belongs to a box is written in the box itself (shape text), not in a text box laid on top
//  - broken y axis uses wavy break marks; x axis keeps only informative layers, omitted runs shown as "⋯"
//  - no prose that the manuscript already defines: legends/labels use notation only
//  - SWP: no arrows between blocks, conv cells are squares, no S* sentence
//  - TSM check shows what decides on/off: the flip of an FP-confident decision on D_chk
// Build (from this folder): NODE_PATH=../../../tools/pptxgen/node_modules node make_overview_v8.js && python3 merge_v8.py
const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");
const HERE = __dirname;
const DIAG = JSON.parse(fs.readFileSync(path.join(HERE, "diag_yolov8s-world.json"), "utf8"));
const EQ = { diag: 3, G: 6, Gq: 7, mig: 9, tie: 11, check: 13 };   // manuscript numbers (11-17)

const W = 10.16, H = 3.66;
const F = "Arial", INK = "222222", MUT = "6B6A64", ARW = "8A8A8A", FRM = "777777", ACC = "B03A2E";
const FUS = "CC5C73", FUSL = "F6DCE2", PRT = "E3A33B", OK = "3B7D5B", OKL = "E3F0E8", OFF = "D9D9D9";

const pres = new pptxgen();
pres.defineLayout({ name: "OV8", width: W, height: H });
pres.layout = "OV8";
pres.title = "DeFiQ method overview v8";
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

// =====================================================================
// (a) layer-wise diagnosis §3.2
const A = { x: 0.06, y: 0.06, w: 2.5, h: 3.54 };
panel(A.x, A.y, A.w, A.h, "Layer-wise diagnosis", "§3.2", `Eq. (${EQ.diag})`);
img(A.x + 0.66, A.y + 0.34, 0.46, 0.34, "cal0.jpg");
img(A.x + 0.76, A.y + 0.41, 0.46, 0.34, "cal1.jpg");
label(A.x + 1.3, A.y + 0.46, 0.6, 0.2, math("*D*_{cal}", 8.5), { align: "left" });
const qY = A.y + 0.92, qH = 0.28, qW = 1.0;
box(A.x + 0.14, qY, qW, qH, math("*Q*^{8}", 9));
box(A.x + A.w - 0.14 - qW, qY, qW, qH, math("*Q*^{8}_{*ℓ*→4}", 9));
const fY = qY + 0.46;
line([[A.x + 0.14 + qW / 2, qY + qH], [A.x + 0.14 + qW / 2, fY]]);
line([[A.x + A.w - 0.14 - qW / 2, qY + qH], [A.x + A.w - 0.14 - qW / 2, fY]]);
box(A.x + 0.14, fY, A.w - 0.28, 0.28, math("Δ_{*ℓ*} = Flip_{cal}(*Q*^{8}_{*ℓ*→4}) − Flip_{cal}(*Q*^{8})", 8));
// bars: informative layers only; omitted runs -> "⋯"
const segs = [[0, 13], [25, 27], [31, 32], [41, 41], [47, 47]];
const ax = A.x + 0.4, base = A.y + 3.28, ph = 1.22, pitch = 0.074, gapW = 0.11;
const loMax = 6, loH = 0.72, brkY = 0.8, hiH0 = 0.88, hiMin = 29, hiMax = 32;
const yOf = v => v <= loMax ? v / loMax * loH : hiH0 + (Math.min(v, hiMax) - hiMin) / (hiMax - hiMin) * (ph - hiH0);
let x = ax + 0.04; const gaps = [];
const bars = [];
segs.forEach((sg, i) => {
  if (i) { gaps.push(x + gapW / 2); x += gapW; }
  for (let k = sg[0]; k <= sg[1]; k++) { bars.push({ r: DIAG[k], x }); x += pitch; }
});
const pw = x - ax + 0.02;
line([[ax, base - ph - 0.03], [ax, base], [ax + pw, base]], { head: false, color: FRM, lw: 0.6 });
[0, 2, 4, 6, 30].forEach(v => {
  line([[ax - 0.04, base - yOf(v)], [ax, base - yOf(v)]], { head: false, color: FRM, lw: 0.5 });
  label(ax - 0.3, base - yOf(v) - 0.07, 0.24, 0.14, txt(String(v), 6, { color: MUT }), { align: "right" });
});
label(ax - 0.34, base - ph - 0.2, 0.8, 0.14, math("Δ_{*ℓ*} (pp)", 6.5, MUT), { align: "left" });
bars.forEach(({ r, x: bx }) => {
  const v = Math.max(r.delta, 0), hh = yOf(v);
  if (hh < 0.004) return;
  const col = r.kind === "fusion" ? FUS : r.kind === "swp" ? PRT : "C4C4C4";
  box(bx + pitch * 0.12, base - hh, pitch * 0.76, hh, "", { fill: col, line: null });
  if (v > loMax) label(bx + pitch, base - hh - 0.01, 0.6, 0.26, txt(`${r.name.replace(".conv", "")}\n${r.delta.toFixed(1)}`, 6.5, { color: FUS, bold: true }), { align: "left", valign: "top" });
});
// wavy break across the axis and the tall bar
const bk = base - brkY;
box(ax - 0.05, bk - 0.035, pw + 0.1, 0.07, "", { fill: "FFFFFF", line: null });
wave(ax, bk - 0.02, 0.12); wave(ax, bk + 0.02, 0.12);
bars.filter(b => b.r.delta > loMax).forEach(b => { wave(b.x + pitch / 2, bk - 0.02, 0.14); wave(b.x + pitch / 2, bk + 0.02, 0.14); });
gaps.forEach(g => label(g - 0.08, base - 0.07, 0.16, 0.14, txt("⋯", 7, { color: MUT }), { fill: { color: "FFFFFF" } }));
label(ax, base + 0.04, pw, 0.13, math("layer *ℓ*", 6.5, MUT));
// legend: notation only
const lgY = base - ph - 0.2;
[[FUS, "*L*_{G}"], [PRT, "*S*^{*}"]].forEach(([c, t], k) => {
  const lx = ax + 0.85 + k * 0.5;
  sq(lx, lgY + 0.035, c);
  label(lx + 0.12, lgY, 0.4, 0.15, math(t, 7.5), { align: "left" });
});

// =====================================================================
// (b) TGR §4.1 — before / after compute graph
const B = { x: 2.66, y: 0.06, w: 3.6, h: 1.78 };
panel(B.x, B.y, B.w, B.h, "Text-Gate Relocation (TGR)", "§4.1", `Eq. (${EQ.G}), (${EQ.Gq})`);
const bh = 0.22, d = 0.17;
const tag = (y, t, c) => label(B.x + 0.12, y - 0.08, 0.5, 0.16, txt(t, 7, { color: c, bold: true }), { align: "left" });
const ix = B.x + 0.66, iw = 0.3;
const inp = (y, src) => box(ix, y - bh / 2, iw, bh, math(src, 8));
// before
const b1 = B.y + 0.48, b2 = B.y + 0.78, bm = (b1 + b2) / 2;
tag(bm, "Before", ACC);
inp(b1, "*y*"); inp(b2, "*p*_{*h*}");
const gB = ix + iw + 0.3;
circ(gB, b2, d, "×", ACC);
label(gB - 0.3, b2 + d / 2 + 0.01, 0.6, 0.13, math("*a*_{*h*}", 7, ACC));
line([[ix + iw, b2], [gB - d / 2, b2]]);
const cx = gB + 0.28, qBx = cx + 0.14;
line([[ix + iw, b1], [cx, b1], [cx, bm], [qBx, bm]]);
line([[gB + d / 2, b2], [cx, b2], [cx, bm]], { head: false });
box(qBx, bm - bh / 2, 0.42, bh, math("*Q*_{*a*}", 8, ACC), { line: ACC, fill: "FBE9E7", lw: 1.2 });
const wBx = qBx + 0.42 + 0.22;
line([[qBx + 0.42, bm], [wBx, bm]]);
box(wBx, bm - bh / 2, 0.62, bh, math("*Q*_{*w*}(*W*)", 8));
line([[wBx + 0.62, bm], [wBx + 0.62 + 0.28, bm]]);
// separator
line([[B.x + 0.12, B.y + 0.98], [B.x + B.w - 0.12, B.y + 0.98]], { head: false, color: "D0D0D0", dash: "dash", lw: 0.6 });
// after
const a1 = B.y + 1.22, a2 = B.y + 1.52;
tag((a1 + a2) / 2, "After", OK);
inp(a1, "*y*"); inp(a2, "*p*_{*h*}");
const qAx = ix + iw + 0.18;
box(qAx, a1 - bh / 2, 0.42, bh, math("*Q*_{*a*}", 8, OK), { line: OK, fill: OKL, lw: 1.2 });
box(qAx, a2 - bh / 2, 0.42, bh, math("*Q*_{*a*}", 8, OK), { line: OK, fill: OKL, lw: 1.2 });
line([[ix + iw, a1], [qAx, a1]]); line([[ix + iw, a2], [qAx, a2]]);
const wAx = qAx + 0.42 + 0.18, wAw = 0.82;
box(wAx, a1 - bh / 2, wAw, bh, math("*Q*_{*w*}(*W*_{main})", 8));
box(wAx, a2 - bh / 2, wAw, bh, math("*Q*_{*w*}(*W*_{*h*})", 8));
line([[qAx + 0.42, a1], [wAx, a1]]); line([[qAx + 0.42, a2], [wAx, a2]]);
const gA = wAx + wAw + 0.24, sA = gA + 0.32;
circ(gA, a2, d, "×", OK);
label(gA - 0.3, a2 + d / 2 + 0.01, 0.6, 0.13, math("*a*_{*h*}", 7, OK));
circ(sA, a1, d, "+");
line([[wAx + wAw, a2], [gA - d / 2, a2]]);
line([[wAx + wAw, a1], [sA - d / 2, a1]]);
line([[gA + d / 2, a2], [sA, a2], [sA, a1 + d / 2]]);
line([[sA + d / 2, a1], [sA + d / 2 + 0.28, a1]]);

// =====================================================================
// (c) SWP §4.2 — bit-widths on the architecture
const P = { x: 2.66, y: 1.92, w: 3.6, h: 1.68 };
panel(P.x, P.y, P.w, P.h, "Selective Weight Protection (SWP)", "§4.2", "");
const blocks = {};
DIAG.forEach(r => { const b = r.name.split(".")[0]; (blocks[b] = blocks[b] || []).push(r); });
const show = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
const cs = 0.085, cg = 0.03, kw = 0.26, kg = 0.03;
const kX = P.x + 0.16, kY = P.y + 0.38, kh = 0.17 + 3 * (cs + cg) + 0.03;
show.forEach((b, k) => {
  const rs = blocks[b] || [], sel = rs.some(r => r.kind === "swp"), x0 = kX + k * (kw + kg);
  box(x0, kY, kw, kh, txt(b, 6.5, { color: MUT }), { line: sel ? PRT : FRM, lw: sel ? 1.2 : 0.6, round: true, r: 0.03, valign: "top" });
  const cl = b === "0" ? [{ kind: "proto" }] : rs;
  cl.slice(0, 6).forEach((r, j) => {
    const col = r.kind === "proto" ? "6E6E6E" : r.kind === "swp" ? PRT : OFF;
    sq(x0 + 0.03 + (j % 2) * (cs + cg), kY + 0.17 + Math.floor(j / 2) * (cs + cg), col, cs);
  });
});
// block number at the top of each block: shape text is centred, so re-place it as a label on top of an empty box
// (kept as shape text above; the number reads at the box centre only if the box has no cells — fine for 3/5/7)
const nX = kX + 10 * (kw + kg) + 0.04, nW = 0.34;
rect(nX, kY, nW, kh, { line: FUS, lw: 0.9, dash: "dash", round: true, r: 0.03 });
label(nX, kY + 0.02, nW, 0.13, math("*L*_{G}", 6.5, FUS));
[0, 1, 2, 3].forEach(j => box(nX + 0.04 + (j % 2) * 0.15, kY + 0.2 + Math.floor(j / 2) * 0.15, 0.12, 0.12, "", { fill: FUSL, line: FUS, lw: 0.5 }));
label(kX, kY + kh + 0.03, 10 * (kw + kg), 0.13, txt("YOLO-World-S backbone blocks", 6.5, { color: MUT }));
const lY = kY + kh + 0.3;
[[PRT, math("*S*^{*} (W8)", 7)], [OFF, txt("W4", 7)], ["6E6E6E", txt("W8 (first layer)", 7)]].forEach(([c, t], k) => {
  const lx = P.x + 0.16 + [0, 0.8, 1.32][k];
  sq(lx, lY + 0.035, c);
  label(lx + 0.13, lY, 1.2, 0.15, t, { align: "left" });
});

// =====================================================================
// (d) TSM §4.3 — numeric example inserted by merge_v8.py (x = T.x + 0.12, y = T.y + 0.3, w = 3.5)
const T = { x: 6.36, y: 0.06, w: 3.74, h: 3.54 };
panel(T.x, T.y, T.w, T.h, "Tied Scale Migration (TSM)", "§4.3", `Eq. (${EQ.mig}), (${EQ.tie})`);
// TSM check: an FP-confident region of D_chk; does each candidate keep the FP decision?
const cT = T.y + 2.5;
line([[T.x + 0.12, cT], [T.x + T.w - 0.12, cT]], { head: false, color: "D0D0D0", dash: "dash", lw: 0.6 });
label(T.x + 0.12, cT + 0.04, 2.0, 0.18, txt("TSM check", 8, { bold: true }), { align: "left" });
label(T.x + T.w - 1.42, cT + 0.04, 1.3, 0.18, txt(`Eq. (${EQ.check})`, 7, { color: MUT }), { align: "right" });
const iX = T.x + 0.14, iY = cT + 0.3, iW = 0.66, iH = 0.495;
img(iX, iY, iW, iH, "chk2.jpg");
rect(iX + 0.25 * iW, iY + 0.06 * iH, 0.34 * iW, 0.9 * iH, { line: PRT, lw: 1.2 });
label(iX, iY + iH + 0.02, iW, 0.14, math("*D*_{chk}", 7, MUT));
const rx = iX + iW + 0.04, rw = 0.32, tw = 0.38, rH = 0.15, rG = 0.025;
const rowsC = [["FP", "cat", null], ["*θ*̂_{on}", "cat", null], ["*θ*̂_{off}", "dog", ACC]];
rowsC.forEach(([who, cls, col], k) => {
  const y = iY + 0.01 + k * (rH + rG);
  label(rx, y, rw, rH, math(who, 7.5), { align: "right" });
  box(rx + rw + 0.06, y, tw, rH, txt(cls, 7, { color: col || INK }), { line: col || FRM, lw: col ? 1.2 : 0.7, round: true, r: 0.03 });
  if (col) label(rx + rw + tw + 0.08, y, 0.3, rH, txt("flip", 7, { color: ACC, bold: true }), { align: "left" });
});
const eX = rx + rw + tw + 0.4, eW = T.x + T.w - 0.12 - eX;
box(eX, iY, eW, iH, mathLines(["Flip_{chk}(*θ*̂_{on}) ≤ Flip_{chk}(*θ*̂_{off})", "→ keep *θ*̂_{on}"], 7.2), { line: ACC, lw: 1.1 });

s.addNotes("v8 (11-17). Bars: YOLO-World-S Δℓ from configs/protect_cache, layers with small Δℓ omitted (⋯). TSM check: illustrative decision (cat→dog), not a measured case.");
const OUT = process.argv[2] || path.join(HERE, "method_overview_v8_base.pptx");
pres.writeFile({ fileName: OUT }).then(() => console.log("written", OUT));
