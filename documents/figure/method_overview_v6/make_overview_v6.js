// Method overview v6 (11-17). Every panel follows the tsm_numeric_v6 pattern: a small worked example (heat cells,
// one red highlight for what matters, gray arrows) next to its equation, so the flow reads at a glance.
// The TSM example itself is the user's tsm_numeric_v6.pptx, inserted unchanged by merge_v6.py into the empty area left here.
// Build (from this folder): NODE_PATH=../../../tools/pptxgen/node_modules node make_overview_v6.js && python3 merge_v6.py
const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");
const HERE = __dirname;
const DIAG = JSON.parse(fs.readFileSync(path.join(HERE, "diag_yolov8s-world.json"), "utf8"));   // name, delta (pp), wpct (% of weights)
const FUSION = ["12.cv2", "15.cv2", "19.cv2", "22.cv2"];

// equation numbers in the manuscript (11-17); change here only
const EQ = { fusion: 5, diag: 3, G: 6, Gq: 7, mig: 9, tie: 11, alpha: 12, check: 13 };

// ---- v6 style ----
const W = 10.6, H = 3.72;
const F = "Arial", INK = "222222", MUT = "6B6A64", ARW = "8A8A8A", FRM = "777777", GRP = "BFBFBF", ACC = "B03A2E";
const CS = 0.27;                      // cell size (in)

const pres = new pptxgen();
pres.defineLayout({ name: "OV6", width: W, height: H });
pres.layout = "OV6";
pres.title = "DeFiQ method overview v6";
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
  s.addText(runs, Object.assign({ x, y, w, h, margin: 0, align: "left", valign: "middle", isTextBox: true, fit: "none" }, o));
}
function rect(x, y, w, h, o = {}) {
  s.addShape(o.round ? pres.shapes.ROUNDED_RECTANGLE : pres.shapes.RECTANGLE, Object.assign({
    x, y, w, h, fill: o.fill ? { color: o.fill } : { type: "none" },
    line: o.line === null ? { type: "none" } : { color: o.line || FRM, width: o.lw || 0.9, dashType: o.dash || "solid" },
  }, o.round ? { rectRadius: o.r || 0.04 } : {}, o.name ? { objectName: o.name } : {}));
}
function seg(x1, y1, x2, y2, o = {}) {
  s.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.max(Math.abs(x2 - x1), 0.0001), h: Math.max(Math.abs(y2 - y1), 0.0001),
    flipH: x2 < x1, flipV: y2 < y1,
    line: { color: o.color || ARW, width: o.lw || 1.0, dashType: o.dash || "solid", endArrowType: o.head ? "triangle" : undefined },
  });
}
function path_(pts, o = {}) { for (let k = 0; k < pts.length - 1; k++) seg(...pts[k], ...pts[k + 1], Object.assign({}, o, { head: o.head !== false && k === pts.length - 2 })); }
// heat colours of tsm_numeric_v6
const hex = c => c.map(v => Math.round(v * 255).toString(16).padStart(2, "0")).join("").toUpperCase();
function heat(v, vmax) {
  const t = Math.min(Math.abs(v) / vmax, 1), c0 = [0.95, 0.94, 0.91], c1 = [0.96, 0.80, 0.45], c2 = [0.80, 0.36, 0.45];
  const lerp = (a, b, u) => a.map((x, i) => x + (b[i] - x) * u);
  return hex(t < 0.5 ? lerp(c0, c1, t / 0.5) : lerp(c1, c2, (t - 0.5) / 0.5));
}
// one row of cells; o.vmax for heat, o.text for display strings, o.fill to override, o.dashed frame
function cells(vals, x, y, o = {}) {
  const cw = o.cw || CS, ch = o.ch || CS;
  vals.forEach((v, j) => s.addText(o.text ? o.text[j] : String(v), {
    x: x + j * cw, y, w: cw, h: ch, margin: 0, align: "center", valign: "middle", isTextBox: true,
    fontFace: F, fontSize: o.sz || 8, color: o.color ? o.color[j] || INK : INK,
    fill: { color: o.fill ? o.fill[j] : heat(v, o.vmax) }, line: { color: "FFFFFF", width: 1.2 },
    objectName: `${o.name || "c"}_${j}`,
  }));
  rect(x, y, vals.length * cw, ch, { line: FRM, lw: 0.9, dash: o.dashed ? "dash" : "solid", name: `${o.name || "c"}_frame` });
}
const hl = (x, y, w, h, name = "highlight") => rect(x, y, w, h, { line: ACC, lw: 1.8, name });
const group = (x, y, w, h) => rect(x, y, w, h, { line: GRP, lw: 1.0, dash: "lgDash" });
function panel(x, y, w, h, title, sec, eq) {
  rect(x, y, w, h, { line: "A8A8A8", lw: 1.2, dash: "lgDash", round: true, r: 0.05 });
  label(x + 0.12, y + 0.05, w - 0.24, 0.2, [
    { text: title, options: { fontFace: F, fontSize: 9, bold: true, color: INK } },
    { text: "  " + sec, options: { fontFace: F, fontSize: 7, color: MUT } },
  ]);
  if (eq) label(x + w - 1.42, y + 0.05, 1.3, 0.2, txt(eq, 7, { color: MUT }), { align: "right" });
}
function eqbox(x, y, w, h, src, sz = 8.5) { rect(x, y, w, h, { line: FRM, lw: 0.9, fill: "FFFFFF" }); label(x, y, w, h, math(src, sz), { align: "center" }); }
function img(x, y, w, h, f) { s.addImage({ path: path.join(HERE, "img", f), x, y, w, h }); rect(x, y, w, h, { line: "FFFFFF", lw: 0.75 }); }

// =====================================================================
// (a) layer-wise diagnosis §3.2: inputs -> Eq. (3) -> Δℓ row (real YOLO-World-S values)
const A = { x: 0.06, y: 0.06, w: 2.94, h: 3.6 };
panel(A.x, A.y, A.w, A.h, "Layer-wise diagnosis", "§3.2", `Eq. (${EQ.diag})`);
img(A.x + 0.14, A.y + 0.36, 0.46, 0.34, "cal0.jpg");
img(A.x + 0.24, A.y + 0.43, 0.46, 0.34, "cal1.jpg");
label(A.x + 0.82, A.y + 0.38, 2.0, 0.36, math("*D*_{cal}: 256 COCO images\nCOCO vocabulary (80 classes)", 7, MUT));
const qY = A.y + 0.92, qH = 0.3;
eqbox(A.x + 0.14, qY, 1.22, qH, "*Q*^{8}", 9); label(A.x + 0.14, qY + qH + 0.02, 1.22, 0.14, txt("RTN W8A8", 6.5, { color: MUT }), { align: "center" });
eqbox(A.x + 1.58, qY, 1.22, qH, "*Q*^{8}_{*ℓ*→4}", 9); label(A.x + 1.58, qY + qH + 0.02, 1.22, 0.14, math("only layer *ℓ* at W4", 6.5, MUT), { align: "center" });
const eY = qY + 0.66;
seg(A.x + 0.75, qY + qH + 0.17, A.x + 0.75, eY, { head: true });
seg(A.x + 2.19, qY + qH + 0.17, A.x + 2.19, eY, { head: true });
eqbox(A.x + 0.14, eY, 2.66, 0.3, "Δ_{*ℓ*} = Flip_{cal}(*Q*^{8}_{*ℓ*→4}) − Flip_{cal}(*Q*^{8})", 8.5);
// example: Δℓ for eight layers in network order
const pick = ["1", "2.cv2", "4.cv1", "6.cv1", "8.cv2", "9.cv2", "12.cv2", "15.cv2"];
const rows = pick.map(n => DIAG.find(r => r.name === n));
const dX = A.x + 0.31, dY = eY + 0.62, dW = 0.31;
seg(A.x + 1.47, eY + 0.3, A.x + 1.47, dY - 0.19, { head: true });
label(A.x + 0.1, dY - 0.19, 2.5, 0.17, math("Δ_{*ℓ*} (pp), YOLO-World-S", 7, MUT));
pick.forEach((n, j) => label(dX + j * dW - 0.05, dY + CS + 0.02, dW + 0.1, 0.14, txt(n, 6, { color: MUT }), { align: "center" }));
cells(rows.map(r => r.delta), dX, dY, { cw: dW, vmax: 5, sz: 7.5, name: "delta", text: rows.map(r => r.delta.toFixed(1)) });
const jF = pick.indexOf("12.cv2");
hl(dX + jF * dW, dY, dW, CS, "delta_fusion_highlight");
// outcomes: fusion conv (by architecture) -> TGR ; ranking of the rest -> SWP
const oY = dY + CS + 0.3;
group(A.x + 0.14, oY, 2.66, 0.56);
label(A.x + 0.22, oY + 0.07, 2.55, 0.18, math("12.cv2: text-fusion conv (*L*_{G}) → TGR", 7));
label(A.x + 0.22, oY + 0.31, 2.55, 0.18, math("other layers: rank by Δ_{*ℓ*} → SWP", 7));

// =====================================================================
// (b) TGR §4.1: toy example. Before, one scale covers [y; a⊙p] and the gated text sets it; after, each branch has its own.
const B = { x: 3.1, y: 0.06, w: 3.6, h: 1.84 };
panel(B.x, B.y, B.w, B.h, "Text-Gate Relocation (TGR)", "§4.1", `Eq. (${EQ.G}), (${EQ.Gq})`);
const yv = [1, -2, 3, 2], pv = [2, -1], a = 4, ap = pv.map(v => a * v);
const bX = B.x + 0.3, bY = B.y + 0.52, gap = 0.16;
// before: one activation scale for the concatenated input; the gated text sets the range
label(bX, bY - 0.2, 2.0, 0.18, math("*Q*_{*a*}([*y*; *a*_{*h*}⊙*p*_{*h*}]),  *a*_{*h*} = 4", 8));
cells(yv.concat(ap), bX, bY, { vmax: 8, name: "concat" });
hl(bX + 4 * CS, bY, 2 * CS, CS, "concat_text_highlight");
label(bX + 6 * CS + gap + 0.12, bY, 1.4, CS, math("one scale, range 8", 7, ACC));
// TGR arrow (down)
const tY = bY + CS + 0.03, aY = bY + CS + 0.38;
seg(bX + 3 * CS, tY, bX + 3 * CS, aY - 0.2, { head: true });
label(bX + 3 * CS + 0.07, tY + 0.01, 0.5, 0.15, txt("TGR", 7, { color: MUT, bold: true }));
// after: each branch has its own scale; the gate multiplies the conv output
label(bX, aY - 0.19, 1.0, 0.17, math("*Q*_{*a*}(*y*)", 8));
cells(yv, bX, aY, { vmax: 3, name: "y" });
label(bX + 4 * CS + gap, aY - 0.19, 1.0, 0.17, math("*Q*_{*a*}(*p*_{*h*})", 8));
cells(pv, bX + 4 * CS + gap, aY, { vmax: 2, name: "p" });
hl(bX + 4 * CS + gap, aY, 2 * CS, CS, "p_highlight");
label(bX + 6 * CS + gap + 0.12, aY - 0.06, 1.4, 0.16, txt("own scales: range 3, 2", 7, { color: ACC }));
label(bX + 6 * CS + gap + 0.12, aY + 0.13, 1.4, 0.16, math("×*a*_{*h*} after the conv", 7, ACC));
eqbox(B.x + 0.14, B.y + 1.48, B.w - 0.28, 0.28, "*W* [*y*; *a*⊙*p*]  =  *W*_{main} *y* + Σ_{*h*} *a*_{*h*} ⊙ (*W*_{*h*} *p*_{*h*})", 8.5);

// =====================================================================
// (c) SWP §4.2: the actual greedy walk on YOLO-World-S (B = 1.5%, a layer that overshoots 1.1B is skipped)
const P = { x: 3.1, y: 1.98, w: 3.6, h: 1.68 };
panel(P.x, P.y, P.w, P.h, "Selective Weight Protection (SWP)", "§4.2", "");
const Bud = 1.5;
const order = DIAG.filter(r => !FUSION.includes(r.name)).sort((p, q) => q.delta - p.delta);
const walk = []; let cum = 0;
for (const r of order) {
  if (cum >= Bud * 0.9) break;
  if (cum + r.wpct > Bud * 1.1) { walk.push({ r, skip: true }); continue; }
  cum += r.wpct; walk.push({ r, skip: false, cum });
}
const sX = P.x + 0.42, sY = P.y + 0.46, sW = 0.312, sH = 0.24;
const SK = "EDEDED";
const rowL = (y, src) => label(P.x + 0.12, y, 0.36, sH, math(src, 7.5), { align: "left" });
rowL(sY, "Δ_{*ℓ*}"); rowL(sY + sH, "*w* (%)"); rowL(sY + 2 * sH, "Σ*w*");
cells(walk.map(t => t.r.delta), sX, sY, { cw: sW, ch: sH, vmax: 5, sz: 7, name: "swp_delta", text: walk.map(t => t.r.delta.toFixed(1)),
  fill: walk.map(t => (t.skip ? SK : heat(t.r.delta, 5))) });
cells(walk.map(t => t.r.wpct), sX, sY + sH, { cw: sW, ch: sH, vmax: 1, sz: 7, name: "swp_w", text: walk.map(t => t.r.wpct.toFixed(2)),
  fill: walk.map(t => (t.skip ? SK : "F3E7CB")) });
cells(walk.map(t => t.cum || 0), sX, sY + 2 * sH, { cw: sW, ch: sH, sz: 7, name: "swp_cum", text: walk.map(t => (t.skip ? "skip" : t.cum.toFixed(2))),
  fill: walk.map(t => (t.skip ? SK : heat(t.cum, Bud * 1.6))), color: walk.map(t => (t.skip ? MUT : INK)) });
const jS = walk.findIndex(t => t.skip), jL = walk.length - 1;
hl(sX + jL * sW, sY + 2 * sH, sW, sH, "swp_final_highlight");
label(sX + jS * sW - 0.3, sY + 3 * sH + 0.02, sW + 0.6, 0.14, math("> 1.1*B*", 6.5, MUT), { align: "center" });
label(sX + jL * sW - 0.45, sY + 3 * sH + 0.02, sW + 0.5, 0.14, math("≈ *B* = 1.5%", 6.5, ACC), { align: "right" });
// layer names above
walk.forEach((t, j) => label(sX + j * sW - 0.1, sY - 0.17, sW + 0.2, 0.15, txt(t.r.name.replace(".m.", ".m"), 5, { color: MUT }), { align: "center" }));
rowL(sY - 0.17, "*ℓ*");
label(P.x + 0.12, sY + 3 * sH + 0.19, 3.4, 0.15, math("add layers in descending Δ_{*ℓ*} (outside *L*_{G}) until *w*(*S*^{*}) ≈ *B*;", 6.8));
label(P.x + 0.12, sY + 3 * sH + 0.34, 3.4, 0.15, math("*S*^{*} keeps 8-bit weights; activations stay low-bit", 6.8));

// =====================================================================
// (d) TSM §4.3: top area is filled by merge_v6.py with tsm_numeric_v6 (x = T.x + 0.12, y = T.y + 0.3, w = 3.5 in)
const T = { x: 6.8, y: 0.06, w: 3.74, h: 3.6 };
panel(T.x, T.y, T.w, T.h, "Tied Scale Migration (TSM)", "§4.3", `Eq. (${EQ.mig}), (${EQ.tie})`);
// TSM check
const cT = T.y + 2.52;
seg(T.x + 0.12, cT, T.x + T.w - 0.12, cT, { color: GRP, dash: "lgDash" });
label(T.x + 0.12, cT + 0.04, 2.0, 0.18, [{ text: "TSM check", options: { fontFace: F, fontSize: 8, bold: true, color: INK } }]);
label(T.x + T.w - 1.42, cT + 0.04, 1.3, 0.18, txt(`Eq. (${EQ.check})`, 7, { color: MUT }), { align: "right" });
const r1 = cT + 0.27, r2 = r1 + 0.3, bh = 0.25, c1 = T.x + 0.12, c2 = c1 + 0.62, c3 = c2 + 0.72;
eqbox(c1, r1, 0.48, bh, "*θ*̂_{on}", 8); hl(c1, r1, 0.48, bh, "theta_on_highlight");
eqbox(c1, r2, 0.48, bh, "*θ*̂_{off}", 8);
eqbox(c2, r1, 0.58, bh, "Recon", 7.5); eqbox(c2, r2, 0.58, bh, "Recon", 7.5);
seg(c1 + 0.48, r1 + bh / 2, c2, r1 + bh / 2, { head: true }); seg(c1 + 0.48, r2 + bh / 2, c2, r2 + bh / 2, { head: true });
seg(c2 + 0.58, r1 + bh / 2, c3, r1 + bh / 2, { head: true }); seg(c2 + 0.58, r2 + bh / 2, c3, r2 + bh / 2, { head: true });
rect(c3, r1, T.x + T.w - 0.12 - c3, r2 + bh - r1, { line: FRM, lw: 0.9, fill: "FFFFFF" });
label(c3, r1 + 0.04, T.x + T.w - 0.12 - c3, 0.22, math("keep *θ*̂_{on} if Flip_{chk}(*θ*̂_{on}) ≤", 7.8), { align: "center" });
label(c3, r1 + 0.27, T.x + T.w - 0.12 - c3, 0.22, math("Flip_{chk}(*θ*̂_{off}), else *θ*̂_{off}", 7.8), { align: "center" });
label(c1, r2 + bh + 0.03, T.w - 0.24, 0.15, math("Recon: BRECQ / QDrop / PD-Quant;  *D*_{chk}: 200 train2017 images not in *D*_{cal}", 6, MUT));

// =====================================================================
// flow between panels (gray)
const gAB = A.x + A.w + 0.05, gBT = B.x + B.w + 0.05;
path_([[A.x + 2.8, oY + 0.16], [gAB, oY + 0.16], [gAB, B.y + 0.9], [B.x, B.y + 0.9]], { dash: "dash" });
path_([[A.x + 2.8, oY + 0.4], [gAB, oY + 0.4], [gAB, P.y + 0.86], [P.x, P.y + 0.86]], { dash: "dash" });
path_([[B.x + B.w, B.y + 0.9], [gBT, B.y + 0.9], [gBT, r1 + 0.28]], { head: false });
path_([[P.x + P.w, P.y + 0.86], [gBT, P.y + 0.86], [gBT, r1 + 0.28]], { head: false });
path_([[gBT, r1 + 0.28], [T.x + 0.06, r1 + 0.28]], { head: false });
seg(T.x + 0.06, r1 + bh / 2, T.x + 0.06, r2 + bh / 2);
seg(T.x + 0.06, r1 + bh / 2, c1, r1 + bh / 2, { head: true }); seg(T.x + 0.06, r2 + bh / 2, c1, r2 + bh / 2, { head: true });

s.addNotes("v6 (11-17). Diagnosis/SWP values: configs/protect_cache yolov8s-world (coco Δ, mparam). TGR numbers are a toy example. TSM example = tsm_numeric_v6.pptx merged by merge_v6.py.");
const OUT = process.argv[2] || path.join(HERE, "method_overview_v6_base.pptx");
pres.writeFile({ fileName: OUT }).then(() => console.log("written", OUT));
