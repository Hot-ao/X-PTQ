// Method overview v7 (11-17). Each panel uses the figure convention that related work uses for that kind of module:
//   diagnosis  : per-layer sensitivity bar chart (HAWQ-style sensitivity plots)
//   TGR        : before/after compute graph with quantizer annotations (RepQ-ViT Fig. 1, HAWQ-V3 Fig. 1, white-paper Fig. 4)
//   SWP        : bit-width per layer drawn on the architecture (HAWQ-V2 Fig. 1)
//   TSM        : numeric matrices (SmoothQuant-style) = the user's tsm_numeric_v6.pptx, inserted unchanged by merge_v7.py
//   TSM check  : two candidates measured on held-out data, the lower one kept
// Build (from this folder): NODE_PATH=../../../tools/pptxgen/node_modules node make_overview_v7.js && python3 merge_v7.py
const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");
const HERE = __dirname;
const DIAG = JSON.parse(fs.readFileSync(path.join(HERE, "diag_yolov8s-world.json"), "utf8"));

const EQ = { fusion: 5, diag: 3, G: 6, Gq: 7, mig: 9, tie: 11, alpha: 12, check: 13 };   // manuscript numbers (11-17)

// ---- style (matches tsm_numeric_v6) ----
const W = 10.6, H = 3.66;
const F = "Arial", INK = "222222", MUT = "6B6A64", ARW = "8A8A8A", FRM = "777777", ACC = "B03A2E";
const FUS = "CC5C73", FUSL = "F6DCE2";      // fusion convs L_G (rose of the heat scale)
const PRT = "E3A33B", PRTL = "FBEBCB";      // protected layers S* (amber of the heat scale)
const OK = "3B7D5B", OKL = "E3F0E8";        // after TGR (RepQ-ViT convention: before red, after green)
const OFF = "D9D9D9";

const pres = new pptxgen();
pres.defineLayout({ name: "OV7", width: W, height: H });
pres.layout = "OV7";
pres.title = "DeFiQ method overview v7";
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
const txt = (t, sz, o = {}) => [{ text: t, options: Object.assign({ fontFace: F, fontSize: sz, color: INK }, o) }];
function label(x, y, w, h, runs, o = {}) {
  s.addText(runs, Object.assign({ x, y, w, h, margin: 0, align: "center", valign: "middle", isTextBox: true, fit: "none" }, o));
}
function rect(x, y, w, h, o = {}) {
  s.addShape(o.round ? pres.shapes.ROUNDED_RECTANGLE : pres.shapes.RECTANGLE, Object.assign({
    x, y, w, h, fill: o.fill ? { color: o.fill } : { type: "none" },
    line: o.line === null ? { type: "none" } : { color: o.line || FRM, width: o.lw || 0.9, dashType: o.dash || "solid" },
  }, o.round ? { rectRadius: o.r || 0.04 } : {}));
}
function boxT(x, y, w, h, runs, o = {}) { rect(x, y, w, h, Object.assign({ fill: "FFFFFF" }, o)); label(x, y, w, h, runs); }
function seg(x1, y1, x2, y2, o = {}) {
  s.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.max(Math.abs(x2 - x1), 0.0001), h: Math.max(Math.abs(y2 - y1), 0.0001),
    flipH: x2 < x1, flipV: y2 < y1,
    line: { color: o.color || ARW, width: o.lw || 1.0, dashType: o.dash || "solid", endArrowType: o.head ? "triangle" : undefined },
  });
}
const arrow = (x1, y1, x2, y2, o = {}) => seg(x1, y1, x2, y2, Object.assign({ head: true }, o));
function path_(pts, o = {}) { for (let k = 0; k < pts.length - 1; k++) seg(...pts[k], ...pts[k + 1], Object.assign({}, o, { head: o.head !== false && k === pts.length - 2 })); }
function circ(x, y, d, t, col = FRM, fill = "FFFFFF") {   // (x, y) = centre
  s.addShape(pres.shapes.OVAL, { x: x - d / 2, y: y - d / 2, w: d, h: d, fill: { color: fill }, line: { color: col, width: 0.9 } });
  label(x - d / 2, y - d / 2 - 0.005, d, d, txt(t, 7.5, { bold: true, color: col === FRM ? INK : col }));
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
function sq(x, y, c, d = 0.085) { s.addShape(pres.shapes.RECTANGLE, { x, y, w: d, h: d, fill: { color: c }, line: { type: "none" } }); }

// =====================================================================
// (a) layer-wise diagnosis §3.2 — sensitivity bar chart
const A = { x: 0.06, y: 0.06, w: 2.94, h: 3.54 };
panel(A.x, A.y, A.w, A.h, "Layer-wise diagnosis", "§3.2", `Eq. (${EQ.diag})`);
img(A.x + 0.14, A.y + 0.36, 0.46, 0.34, "cal0.jpg");
img(A.x + 0.24, A.y + 0.43, 0.46, 0.34, "cal1.jpg");
label(A.x + 0.82, A.y + 0.38, 2.0, 0.36, math("*D*_{cal}: 256 COCO images\nCOCO vocabulary (80 classes)", 7, MUT), { align: "left" });
const qY = A.y + 0.94, qH = 0.3;
boxT(A.x + 0.14, qY, 1.22, qH, math("*Q*^{8}", 9).concat(txt("  W8A8", 6.5, { color: MUT })));
boxT(A.x + 1.58, qY, 1.22, qH, math("*Q*^{8}_{*ℓ*→4}", 9).concat(math("  layer *ℓ* at W4", 6.5, MUT)));
const fY = qY + 0.5;
arrow(A.x + 0.75, qY + qH, A.x + 0.75, fY); arrow(A.x + 2.19, qY + qH, A.x + 2.19, fY);
boxT(A.x + 0.14, fY, 2.66, 0.3, math("Δ_{*ℓ*} = Flip_{cal}(*Q*^{8}_{*ℓ*→4}) − Flip_{cal}(*Q*^{8})", 8.5));
// bars, broken axis 6 -> 29 pp so that 12.cv2 = 30.8 stays visible
const ax = A.x + 0.42, base = A.y + 3.26, pw = 2.36, ph = 0.86;
const loMax = 6, loH = 0.52, brk = [0.56, 0.63], hiMin = 29, hiMax = 32, hiH0 = 0.66;
const yOf = v => v <= loMax ? v / loMax * loH : hiH0 + (Math.min(v, hiMax) - hiMin) / (hiMax - hiMin) * (ph - hiH0);
seg(ax, base, ax, base - ph - 0.02, { color: FRM, lw: 0.6 });
seg(ax - 0.02, base, ax + pw + 0.02, base, { color: FRM, lw: 0.6 });
[0, 2, 4, 6, 30].forEach(v => { seg(ax - 0.04, base - yOf(v), ax, base - yOf(v), { color: FRM, lw: 0.5 }); label(ax - 0.3, base - yOf(v) - 0.07, 0.24, 0.14, txt(String(v), 6, { color: MUT }), { align: "right" }); });
rect(ax - 0.03, base - brk[1], 0.06, brk[1] - brk[0], { fill: "FFFFFF", line: null });
brk.forEach(b => seg(ax - 0.05, base - b + 0.03, ax + 0.05, base - b - 0.03, { color: FRM, lw: 0.6 }));
label(ax - 0.3, base - ph - 0.2, 1.2, 0.14, math("Δ_{*ℓ*} (pp), YOLO-World-S", 6.5, MUT), { align: "left" });
const n = DIAG.length, bw = pw / n;
DIAG.forEach((r, k) => {
  const v = Math.max(r.delta, 0), hh = yOf(v);
  if (hh < 0.004) return;
  const col = r.kind === "fusion" ? FUS : r.kind === "swp" ? PRT : "C4C4C4";
  const x0 = ax + k * bw + bw * 0.1, wd = bw * 0.8;
  s.addShape(pres.shapes.RECTANGLE, { x: x0, y: base - hh, w: wd, h: hh, fill: { color: col }, line: { type: "none" } });
  if (v > loMax) {
    rect(x0 - 0.015, base - brk[1], wd + 0.03, brk[1] - brk[0], { fill: "FFFFFF", line: null });
    brk.forEach(b => seg(x0 - 0.03, base - b + 0.03, x0 + wd + 0.03, base - b - 0.03, { color: FRM, lw: 0.6 }));
    label(x0 + wd + 0.03, base - hh - 0.02, 0.6, 0.26, txt(`${r.name.replace(".conv", "")}\n${r.delta.toFixed(1)}`, 6.5, { color: FUS, bold: true }), { align: "left", valign: "top" });
  }
});
label(ax, base + 0.03, pw, 0.13, txt("layers in network order", 6.5, { color: MUT }));
const lgX = A.x + 0.2, lgY = fY + 0.42;
[[FUS, math("*L*_{G}: text-fusion convs → TGR", 6.8)], [PRT, math("*S*^{*}: top Δ_{*ℓ*} within budget *B* → SWP", 6.8)]].forEach(([c, t], k) => {
  sq(lgX, lgY + k * 0.17 + 0.035, c);
  label(lgX + 0.13, lgY + k * 0.17, 1.6, 0.15, t, { align: "left" });
});

// =====================================================================
// (b) TGR §4.1 — before / after compute graph of a text-fusion conv
const B = { x: 3.1, y: 0.06, w: 3.6, h: 1.78 };
panel(B.x, B.y, B.w, B.h, "Text-Gate Relocation (TGR)", "§4.1", `Eq. (${EQ.G}), (${EQ.Gq})`);
const bh = 0.22, d = 0.17;
const tag = (y, t, c) => label(B.x + 0.1, y - 0.08, 0.5, 0.16, txt(t, 7, { color: c, bold: true }), { align: "left" });
const inp = (x, y, src) => boxT(x, y - bh / 2, 0.3, bh, math(src, 8));
// before: y and a⊙p are concatenated, then ONE activation quantizer
const b1 = B.y + 0.46, b2 = B.y + 0.74, bm = (b1 + b2) / 2;
tag(bm, "Before", ACC);
const ix = B.x + 0.62;
inp(ix, b1, "*y*"); inp(ix, b2, "*p*_{*h*}");
const gB = ix + 0.58;
circ(gB, b2, d, "×", ACC);
label(gB - 0.3, b2 + 0.11, 0.6, 0.14, math("*a*_{*h*}", 7, ACC));
arrow(ix + 0.3, b2, gB - d / 2, b2);
const cx = gB + 0.3;                                                // concat point
path_([[ix + 0.3, b1], [cx, b1], [cx, bm]], { head: false });
path_([[gB + d / 2, b2], [cx, b2], [cx, bm]], { head: false });
const qBx = cx + 0.12;
boxT(qBx, bm - bh / 2, 0.5, bh, math("*Q*_{*a*}", 8, ACC), { line: ACC, fill: "FBE9E7", lw: 1.2 });
arrow(cx, bm, qBx, bm);
label(qBx - 0.25, bm - bh / 2 - 0.16, 1.0, 0.14, txt("one scale", 6.5, { color: ACC }));
const wBx = qBx + 0.68;
boxT(wBx, bm - bh / 2, 0.62, bh, math("*Q*_{*w*}(*W*)", 8));
arrow(qBx + 0.5, bm, wBx, bm);
arrow(wBx + 0.62, bm, wBx + 0.9, bm);
label(wBx + 0.92, bm - 0.08, 0.5, 0.16, txt("out", 7, { color: MUT }), { align: "left" });
// separator
seg(B.x + 0.12, B.y + 0.93, B.x + B.w - 0.12, B.y + 0.93, { color: "D0D0D0", dash: "dash", lw: 0.6 });
// after: each branch has its own quantizer; the gate multiplies the conv output
const a1 = B.y + 1.2, a2 = B.y + 1.52;
tag((a1 + a2) / 2, "After", OK);
inp(ix, a1, "*y*"); inp(ix, a2, "*p*_{*h*}");
const qAx = ix + 0.48;
boxT(qAx, a1 - bh / 2, 0.42, bh, math("*Q*_{*a*}", 8, OK), { line: OK, fill: OKL, lw: 1.2 });
boxT(qAx, a2 - bh / 2, 0.42, bh, math("*Q*_{*a*}", 8, OK), { line: OK, fill: OKL, lw: 1.2 });
arrow(ix + 0.3, a1, qAx, a1); arrow(ix + 0.3, a2, qAx, a2);
const wAx = qAx + 0.58;
boxT(wAx, a1 - bh / 2, 0.82, bh, math("*Q*_{*w*}(*W*_{main})", 8));
boxT(wAx, a2 - bh / 2, 0.82, bh, math("*Q*_{*w*}(*W*_{*h*})", 8));
arrow(qAx + 0.42, a1, wAx, a1); arrow(qAx + 0.42, a2, wAx, a2);
const gA = wAx + 0.82 + 0.22, sA = gA + 0.32;
circ(gA, a2, d, "×", OK);
label(gA - 0.2, a2 + 0.1, 0.4, 0.14, math("*a*_{*h*}", 7, OK));
circ(sA, a1, d, "+");
arrow(wAx + 0.82, a2, gA - d / 2, a2);
arrow(wAx + 0.82, a1, sA - d / 2, a1);
path_([[gA + d / 2, a2], [sA, a2], [sA, a1 + d / 2]]);
arrow(sA + d / 2, a1, sA + 0.3, a1);
label(sA + 0.32, a1 - 0.08, 0.3, 0.16, txt("out", 7, { color: MUT }), { align: "left" });
label(qAx - 0.25, a1 - bh / 2 - 0.15, 0.92, 0.13, txt("scale per branch", 6.5, { color: OK }));
label(wAx, a2 + bh / 2 + 0.02, 0.82, 0.13, txt("gate after the conv", 6.5, { color: OK }), { align: "right" });

// =====================================================================
// (c) SWP §4.2 — bit-widths on the architecture (HAWQ-style)
const P = { x: 3.1, y: 1.92, w: 3.6, h: 1.68 };
panel(P.x, P.y, P.w, P.h, "Selective Weight Protection (SWP)", "§4.2", "");
const blocks = {};
DIAG.forEach(r => { const b = r.name.split(".")[0]; (blocks[b] = blocks[b] || []).push(r); });
const show = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
const kX = P.x + 0.14, kY = P.y + 0.36, kw = 0.25, kh = 0.6, kg = 0.04;
show.forEach((b, k) => {
  const rs = blocks[b] || [], sel = rs.some(r => r.kind === "swp"), x = kX + k * (kw + kg);
  rect(x, kY, kw, kh, { fill: "FFFFFF", line: sel ? PRT : FRM, lw: sel ? 1.2 : 0.6, round: true, r: 0.03 });
  label(x, kY + 0.02, kw, 0.12, txt(b, 6.5, { color: MUT }));
  const cl = b === "0" ? [{ kind: "proto" }] : rs;
  cl.slice(0, 6).forEach((r, j) => {
    const cx2 = x + 0.035 + (j % 2) * 0.095, cy = kY + 0.17 + Math.floor(j / 2) * 0.135;
    const col = r.kind === "proto" ? "6E6E6E" : r.kind === "swp" ? PRT : OFF;
    s.addShape(pres.shapes.RECTANGLE, { x: cx2, y: cy, w: 0.08, h: 0.1, fill: { color: col }, line: { type: "none" } });
  });
  if (k) arrow(x - kg, kY + kh / 2, x, kY + kh / 2, { lw: 0.6 });
});
const nX = kX + 10 * (kw + kg) + 0.04;
rect(nX, kY, 0.46, kh, { fill: "FFFFFF", line: FUS, lw: 0.9, dash: "dash", round: true, r: 0.03 });
label(nX, kY + 0.02, 0.46, 0.12, math("*L*_{G}", 6.5, FUS));
[0, 1, 2, 3].forEach(j => s.addShape(pres.shapes.RECTANGLE, { x: nX + 0.07 + (j % 2) * 0.18, y: kY + 0.2 + Math.floor(j / 2) * 0.17, w: 0.13, h: 0.11, fill: { color: FUSL }, line: { color: FUS, width: 0.5 } }));
label(kX, kY + kh + 0.02, 10 * (kw + kg), 0.13, txt("backbone blocks (YOLO-World-S)", 6.5, { color: MUT }));
label(nX - 0.12, kY + kh + 0.02, 0.7, 0.13, txt("→ TGR", 6.5, { color: FUS }));
const lY = P.y + 1.16;
[[PRT, math("W8 (*S*^{*})", 7)], [OFF, txt("W4", 7)], ["6E6E6E", txt("W8 (first layer, protocol)", 7)]].forEach(([c, t], k) => {
  const lx = P.x + 0.14 + [0, 0.74, 1.26][k];
  sq(lx, lY + 0.035, c);
  label(lx + 0.13, lY, 1.4, 0.15, t, { align: "left" });
});
label(P.x + 0.14, P.y + 1.36, 3.4, 0.16, math("*S*^{*}: top Δ_{*ℓ*} outside *L*_{G} until *w*(*S*^{*}) ≈ *B* = 1.5%; only weights kept at 8 bits", 6.8), { align: "left" });

// =====================================================================
// (d) TSM §4.3 — numeric example (tsm_numeric_v6) goes into the top area via merge_v7.py
const T = { x: 6.8, y: 0.06, w: 3.74, h: 3.54 };
panel(T.x, T.y, T.w, T.h, "Tied Scale Migration (TSM)", "§4.3", `Eq. (${EQ.mig}), (${EQ.tie})`);
// TSM check: two candidates, measured on D_chk, keep the lower flip rate
const cT = T.y + 2.5;
seg(T.x + 0.12, cT, T.x + T.w - 0.12, cT, { color: "D0D0D0", dash: "dash", lw: 0.6 });
label(T.x + 0.12, cT + 0.04, 2.0, 0.18, txt("TSM check", 8, { bold: true }), { align: "left" });
label(T.x + T.w - 1.42, cT + 0.04, 1.3, 0.18, txt(`Eq. (${EQ.check})`, 7, { color: MUT }), { align: "right" });
const r1 = cT + 0.4, r2 = r1 + 0.3, ch = 0.22, c1 = T.x + 0.12, c2 = c1 + 0.58, c3 = c2 + 0.68;
boxT(c1, r1 - ch / 2, 0.44, ch, math("*θ*̂_{on}", 8));
boxT(c1, r2 - ch / 2, 0.44, ch, math("*θ*̂_{off}", 8));
boxT(c2, r1 - ch / 2, 0.52, ch, txt("Recon", 7.5)); boxT(c2, r2 - ch / 2, 0.52, ch, txt("Recon", 7.5));
arrow(c1 + 0.44, r1, c2, r1); arrow(c1 + 0.44, r2, c2, r2);
// measured flip rates on D_chk (schematic bars)
const bx0 = c3 + 0.1, bL = 0.9;
rect(c3, r1 - ch / 2 - 0.2, bL + 0.42, r2 - r1 + ch + 0.27, { line: FRM, lw: 0.7 });
label(c3, r1 - ch / 2 - 0.19, bL + 0.42, 0.15, math("Flip_{chk} on *D*_{chk}", 6.8, INK));
arrow(c2 + 0.52, r1, c3, r1); arrow(c2 + 0.52, r2, c3, r2);
s.addShape(pres.shapes.RECTANGLE, { x: bx0, y: r1 - 0.06, w: bL * 0.55, h: 0.12, fill: { color: "F5CC73" }, line: { type: "none" } });
s.addShape(pres.shapes.RECTANGLE, { x: bx0, y: r2 - 0.06, w: bL * 0.8, h: 0.12, fill: { color: "E09473" }, line: { type: "none" } });
label(bx0 + bL * 0.55 + 0.04, r1 - 0.08, 0.3, 0.16, txt("on", 6.5, { color: MUT }), { align: "left" });
label(bx0 + bL * 0.8 + 0.04, r2 - 0.08, 0.3, 0.16, txt("off", 6.5, { color: MUT }), { align: "left" });
const oX = c3 + bL + 0.42 + 0.14, oW = T.x + T.w - 0.12 - oX;
arrow(c3 + bL + 0.42, (r1 + r2) / 2, oX, (r1 + r2) / 2);
boxT(oX, (r1 + r2) / 2 - 0.2, oW, 0.4, [], { line: ACC, lw: 1.2 });
label(oX, (r1 + r2) / 2 - 0.17, oW, 0.16, txt("keep the lower", 7));
label(oX, (r1 + r2) / 2 + 0.0, oW, 0.16, math("(tie → *θ*̂_{on})", 6.5, MUT));
label(c1, r2 + ch / 2 + 0.07, T.w - 0.24, 0.14, math("*D*_{chk}: 200 train2017 images not in *D*_{cal};  Recon: BRECQ / QDrop / PD-Quant", 6, MUT), { align: "left" });

// =====================================================================
// flow between panels (gray)
const gAB = A.x + A.w + 0.05, gBT = B.x + B.w + 0.05;
const lg1 = lgY + 0.075, lg2 = lgY + 0.245;
path_([[A.x + A.w - 0.04, lg1], [gAB, lg1], [gAB, B.y + 0.96], [B.x, B.y + 0.96]], { dash: "dash", color: FUS });
path_([[A.x + A.w - 0.04, lg2], [gAB + 0.0, lg2], [gAB, P.y + 0.66], [P.x, P.y + 0.66]], { dash: "dash", color: PRT });
path_([[B.x + B.w, B.y + 0.96], [gBT, B.y + 0.96], [gBT, (r1 + r2) / 2]], { head: false });
path_([[P.x + P.w, P.y + 0.66], [gBT, P.y + 0.66], [gBT, (r1 + r2) / 2]], { head: false });
path_([[gBT, (r1 + r2) / 2], [T.x + 0.06, (r1 + r2) / 2]], { head: false });
seg(T.x + 0.06, r1, T.x + 0.06, r2);
arrow(T.x + 0.06, r1, c1, r1); arrow(T.x + 0.06, r2, c1, r2);

s.addNotes("v7 (11-17). Bars: YOLO-World-S Δℓ (configs/protect_cache). TSM example = tsm_numeric_v6.pptx merged by merge_v7.py. Check bars are schematic (no values).");
const OUT = process.argv[2] || path.join(HERE, "method_overview_v7_base.pptx");
pres.writeFile({ fileName: OUT }).then(() => console.log("written", OUT));
