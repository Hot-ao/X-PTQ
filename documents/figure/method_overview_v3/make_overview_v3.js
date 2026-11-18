// 방법 개요 그림 v3 (11-17). v2에서 바꾼 것: 전체 Arial, 여백 축소(10×3.45in), 원고 기호와 식 번호 통일,
// 식은 패널당 하나만(나머지는 Eq. 번호로 가리킨다), 진단 막대는 끊긴 축(broken axis)으로 12.cv2를 그대로 보인다.
// 사용(저장소 루트): NODE_PATH=tools/pptxgen/node_modules node documents/figure/method_overview_v3/make_overview_v3.js
const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");
const HERE = __dirname;
const DIAG = JSON.parse(fs.readFileSync(path.join(HERE, "diag_yolov8s-world.json"), "utf8"));

// 원고 식 번호(11-17 기준). 식 순서가 바뀌면 여기만 고친다.
const EQ = { fusion: 5, diag: 3, G: 6, Gq: 7, mig: 9, tie: 11, alpha: 12, check: 13 };

const W = 10, H = 3.45;
const INK = "1F1F1F", MUTED = "6B6B6B", LINE = "4D4D4D";
const C = {
  diag: { ln: "A07A12", fill: "FBF4E2" },
  tgr: { ln: "2E7D4F", fill: "E6F2EA" },
  swp: { ln: "6A3FA0", fill: "EFE7F7" },
  tsm: { ln: "2F6DB5", fill: "E7EFF9" },
  q: { ln: "8A8A8A", fill: "EDEDED" },
  txt: { ln: "C77A1E", fill: "FFF4E5" },
};
const F = "Arial";

const pres = new pptxgen();
pres.defineLayout({ name: "OV", width: W, height: H });
pres.layout = "OV";
pres.title = "DeFiQ method overview v3";
const s = pres.addSlide();
s.background = { color: "FFFFFF" };

// ---- 수식 마크업: _{..} 아래첨자, ^{..} 위첨자, *..* 이탤릭(변수). ----
function math(src, sz, color = INK, base = {}) {
  const runs = [];
  let i = 0, ital = false, buf = "";
  const push = (t, o = {}) => { if (t) runs.push({ text: t, options: Object.assign({ fontFace: F, fontSize: sz, color, italic: ital }, base, o) }); };
  while (i < src.length) {
    const ch = src[i];
    if ((ch === "_" || ch === "^") && src[i + 1] === "{") {
      push(buf); buf = "";
      let d = 1, j = i + 2;
      while (j < src.length && d) { if (src[j] === "{") d++; if (src[j] === "}") d--; j++; }
      const inner = src.slice(i + 2, j - 1);
      inner.split(/(\*[^*]+\*)/).forEach(part => {
        if (!part) return;
        const it = part.length > 2 && part.startsWith("*") && part.endsWith("*");
        push(it ? part.slice(1, -1) : part, { [ch === "_" ? "subscript" : "superscript"]: true, italic: it, fontSize: sz });
      });
      i = j; continue;
    }
    if (ch === "*") { push(buf); buf = ""; ital = !ital; i++; continue; }
    buf += ch; i++;
  }
  push(buf);
  return runs;
}
function txt(t, sz, o = {}) { return [{ text: t, options: Object.assign({ fontFace: F, fontSize: sz, color: INK }, o) }]; }

// ---- 도형 ----
function rect(x, y, w, h, o = {}) {
  s.addShape(o.round ? pres.shapes.ROUNDED_RECTANGLE : pres.shapes.RECTANGLE, Object.assign({
    x, y, w, h,
    fill: o.fill ? { color: o.fill } : { type: "none" },
    line: o.line === null ? { type: "none" } : { color: o.line || LINE, width: o.lw || 0.75, dashType: o.dash || "solid" },
  }, o.round ? { rectRadius: o.r || 0.05 } : {}));
}
function label(x, y, w, h, runs, o = {}) {
  s.addText(runs, Object.assign({ x, y, w, h, margin: 0, align: "center", valign: "middle", isTextBox: true, fit: "none" }, o));
}
function boxT(x, y, w, h, runs, o = {}) { rect(x, y, w, h, o); label(x, y, w, h, runs, { align: o.align || "center" }); }
function arrow(x1, y1, x2, y2, o = {}) {
  const x = Math.min(x1, x2), y = Math.min(y1, y2), w = Math.abs(x2 - x1), h = Math.abs(y2 - y1);
  s.addShape(pres.shapes.LINE, {
    x, y, w: Math.max(w, 0.0001), h: Math.max(h, 0.0001),
    flipH: x2 < x1, flipV: y2 < y1,
    line: { color: o.color || LINE, width: o.lw || 0.9, dashType: o.dash || "solid", endArrowType: o.head === false ? undefined : "triangle" },
  });
}
function path_(pts, o = {}) {
  for (let k = 0; k < pts.length - 1; k++)
    arrow(pts[k][0], pts[k][1], pts[k + 1][0], pts[k + 1][1], Object.assign({}, o, { head: k === pts.length - 2 && o.head !== false }));
}
function panel(x, y, w, h, col, title, sec) {
  rect(x, y, w, h, { line: col.ln, lw: 1.0, dash: "dash", round: true, r: 0.06 });
  label(x, y + 0.04, w, 0.22, [
    { text: title, options: { fontFace: F, fontSize: 8.5, bold: true, color: INK } },
    { text: "  " + sec, options: { fontFace: F, fontSize: 7, color: MUTED } },
  ]);
}
function eqtag(x, y, t, w = 0.7) { label(x, y, w, 0.14, txt(t, 6, { color: MUTED }), { align: "right" }); }
function img(x, y, w, h, f) { s.addImage({ path: path.join(HERE, "img", f), x, y, w, h }); rect(x, y, w, h, { line: "FFFFFF", lw: 0.75 }); }
function circ(x, y, d, t) { s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: "FFFFFF" }, line: { color: LINE, width: 0.75 } }); label(x, y - 0.005, d, d, txt(t, 7.5, { bold: true })); }
function sq(x, y, c, d = 0.085) { s.addShape(pres.shapes.RECTANGLE, { x, y, w: d, h: d, fill: { color: c }, line: { type: "none" } }); }

// =====================================================================
// (a) 진단 §3.2
const A = { x: 0.06, y: 0.06, w: 2.9, h: 3.33 };
panel(A.x, A.y, A.w, A.h, C.diag, "Layer-wise diagnosis", "§3.2");
// D_cal
img(A.x + 0.12, A.y + 0.34, 0.5, 0.37, "cal0.jpg");
img(A.x + 0.24, A.y + 0.42, 0.5, 0.37, "cal1.jpg");
label(A.x + 0.86, A.y + 0.36, 1.95, 0.4, math("*D*_{cal}: 256 COCO images,\nCOCO vocabulary (80 classes)", 6.8), { align: "left" });
// Q^8, Q^8_{ℓ→4}
const mY = A.y + 0.96;
boxT(A.x + 0.14, mY, 1.2, 0.34, math("*Q*^{8}", 9).concat(txt("  RTN W8A8", 6.2, { color: MUTED })), { fill: C.q.fill, line: C.q.ln, round: true });
boxT(A.x + 1.52, mY, 1.26, 0.34, math("*Q*^{8}_{*ℓ*→4}", 9).concat(txt("  layer ℓ → W4", 6.2, { color: MUTED })), { fill: C.q.fill, line: C.q.ln, round: true });
arrow(A.x + 0.74, A.y + 0.82, A.x + 0.74, mY);
arrow(A.x + 2.15, A.y + 0.82, A.x + 2.15, mY);
// Eq. (3)
const fY = mY + 0.5;
arrow(A.x + 0.74, mY + 0.34, A.x + 0.74, fY);
arrow(A.x + 2.15, mY + 0.34, A.x + 2.15, fY);
boxT(A.x + 0.14, fY, 2.64, 0.3, math("Δ_{*ℓ*} = Flip_{cal}(*Q*^{8}_{*ℓ*→4}) − Flip_{cal}(*Q*^{8})", 8.5), { fill: C.diag.fill, line: C.diag.ln });
eqtag(A.x + 2.1, fY + 0.31, `Eq. (${EQ.diag})`);
// 막대: 층별 Δℓ, 끊긴 축. 아래 구간 0–6 pp, 위 구간 29–32 pp.
const ax = A.x + 0.46, base = A.y + 3.08, pw = 2.3, ph = 0.8;
const loMax = 6, loH = 0.5, brk = [0.54, 0.61], hiMin = 29, hiMax = 32, hiH0 = 0.63;
const yOf = v => v <= loMax ? v / loMax * loH : hiH0 + (Math.min(v, hiMax) - hiMin) / (hiMax - hiMin) * (ph - hiH0);
// 축
arrow(ax, base, ax, base - ph - 0.02, { lw: 0.6, head: false });
rect(ax - 0.02, base, pw + 0.04, 0.0001, { line: LINE, lw: 0.6 });
[0, 2, 4, 6].forEach(v => { rect(ax - 0.04, base - yOf(v), 0.04, 0.0001, { line: LINE, lw: 0.5 }); label(ax - 0.3, base - yOf(v) - 0.07, 0.24, 0.14, txt(String(v), 5.5, { color: MUTED }), { align: "right" }); });
[30].forEach(v => { rect(ax - 0.04, base - yOf(v), 0.04, 0.0001, { line: LINE, lw: 0.5 }); label(ax - 0.3, base - yOf(v) - 0.07, 0.24, 0.14, txt(String(v), 5.5, { color: MUTED }), { align: "right" }); });
// 축의 끊김 표시
rect(ax - 0.03, base - brk[1], 0.06, brk[1] - brk[0], { fill: "FFFFFF", line: null });
[brk[0], brk[1]].forEach(b => arrow(ax - 0.05, base - b + 0.03, ax + 0.05, base - b - 0.03, { lw: 0.6, head: false }));
label(ax - 0.42, base - ph - 0.17, 0.7, 0.14, math("Δ_{*ℓ*} (pp)", 6.2, MUTED), { align: "left" });
const n = DIAG.length, bw = pw / n;
DIAG.forEach((r, k) => {
  const v = Math.max(r.delta, 0), hh = yOf(v);
  if (hh < 0.004) return;
  const col = r.kind === "fusion" ? C.tgr.ln : r.kind === "swp" ? C.swp.ln : "BDBDBD";
  const x0 = ax + k * bw + bw * 0.1, wdt = bw * 0.8;
  s.addShape(pres.shapes.RECTANGLE, { x: x0, y: base - hh, w: wdt, h: hh, fill: { color: col }, line: { type: "none" } });
  if (v > loMax) {   // 막대의 끊김
    rect(x0 - 0.015, base - brk[1], wdt + 0.03, brk[1] - brk[0], { fill: "FFFFFF", line: null });
    [brk[0], brk[1]].forEach(b => arrow(x0 - 0.03, base - b + 0.03, x0 + wdt + 0.03, base - b - 0.03, { lw: 0.6, head: false }));
    label(x0 + wdt + 0.03, base - hh - 0.02, 0.6, 0.26, txt(`${r.name.replace(".conv", "")}\n${r.delta.toFixed(1)}`, 6, { color: C.tgr.ln, bold: true }), { align: "left", valign: "top" });
  }
});
label(ax, base + 0.02, pw, 0.13, txt("layers in network order (YOLO-World-S)", 5.8, { color: MUTED }));
// 범례
const lgX = ax + 0.5, lgY = base - ph - 0.36;
[[C.tgr.ln, math("*L*_{G}: fusion convs → TGR", 6.4)], [C.swp.ln, math("*S*^{*}: largest Δ_{*ℓ*}, *w*(*S*^{*}) ≈ *B* → SWP", 6.4)]].forEach(([c, t], k) => {
  sq(lgX, lgY + k * 0.17 + 0.03, c);
  label(lgX + 0.13, lgY + k * 0.17, 1.6, 0.15, t, { align: "left" });
});

// =====================================================================
// (b) TGR §4.1
const B = { x: 3.1, y: 0.06, w: 3.7, h: 1.72 };
panel(B.x, B.y, B.w, B.h, C.tgr, "Text-Gate Relocation (TGR)", "§4.1");
const r1 = B.y + 0.38, r2 = B.y + 0.72, bh = 0.26;
const lx0 = B.x + 0.1, lw0 = 0.5;
boxT(lx0, r1, lw0, bh, math("*y*", 9), { fill: "FFFFFF", line: LINE });
boxT(lx0, r2, lw0, bh, math("*p*_{*h*}", 9), { fill: "FFFFFF", line: LINE });
label(lx0 - 0.02, r1 - 0.15, lw0 + 0.04, 0.13, txt("visual", 5.8, { color: MUTED }));
label(lx0 - 0.02, r2 + bh + 0.01, lw0 + 0.04, 0.13, txt("text", 5.8, { color: MUTED }));
const qx = lx0 + lw0 + 0.2, qw = 0.72;
boxT(qx, r1, qw, bh, math("*Q*_{*a*}(*y*)", 8.5), { fill: C.q.fill, line: C.q.ln });
boxT(qx, r2, qw, bh, math("*Q*_{*a*}(*p*_{*h*})", 8.5), { fill: C.q.fill, line: C.q.ln });
const wx = qx + qw + 0.16, ww = 0.9;
boxT(wx, r1, ww, bh, math("*Q*_{*w*}(*W*_{main})", 8.5), { fill: C.tgr.fill, line: C.tgr.ln, lw: 1.0 });
boxT(wx, r2, ww, bh, math("*Q*_{*w*}(*W*_{*h*})", 8.5), { fill: C.tgr.fill, line: C.tgr.ln, lw: 1.0 });
arrow(lx0 + lw0, r1 + bh / 2, qx, r1 + bh / 2); arrow(lx0 + lw0, r2 + bh / 2, qx, r2 + bh / 2);
arrow(qx + qw, r1 + bh / 2, wx, r1 + bh / 2); arrow(qx + qw, r2 + bh / 2, wx, r2 + bh / 2);
const d = 0.2, gx = wx + ww + 0.16, sx = gx + 0.34, slx = sx + d + 0.12;
circ(gx, r2 + bh / 2 - d / 2, d, "⊙");
circ(sx, r1 + bh / 2 - d / 2, d, "+");
arrow(wx + ww, r2 + bh / 2, gx, r2 + bh / 2);
arrow(wx + ww, r1 + bh / 2, sx, r1 + bh / 2);
path_([[gx + d, r2 + bh / 2], [sx + d / 2, r2 + bh / 2], [sx + d / 2, r1 + bh / 2 + d / 2]]);
boxT(slx, r1, 0.42, bh, txt("SiLU", 7), { fill: "FFFFFF", line: LINE });
arrow(sx + d, r1 + bh / 2, slx, r1 + bh / 2);
label(wx - 0.05, r1 - 0.16, ww + 0.1, 0.13, txt("one scale per branch", 5.8, { color: C.tgr.ln }));
// 게이트 a_h: 텍스트 임베딩에서, conv 뒤에서 곱한다
const aY = r2 + bh + 0.08;
boxT(qx + 0.1, aY, 1.3, 0.22, math("gate *a*_{*h*}  (Eq. ", 7).concat(txt(`${EQ.fusion})`, 7)), { fill: C.txt.fill, line: C.txt.ln });
path_([[qx + 1.4, aY + 0.11], [gx + d / 2, aY + 0.11], [gx + d / 2, r2 + bh / 2 + d / 2]], { color: C.txt.ln });
label(qx + 1.5, aY - 0.01, 1.0, 0.13, txt("moved after the conv", 5.8, { color: C.txt.ln }), { align: "left" });
// Eq. (6)
const eY = B.y + 1.4;
rect(B.x + 0.1, eY, 3.5, 0.26, { fill: C.tgr.fill, line: C.tgr.ln });
label(B.x + 0.22, eY, 2.4, 0.26, math("*W* [*y*; *a*⊙*p*]  =  *W*_{main} *y* + Σ_{*h*} *a*_{*h*} ⊙ (*W*_{*h*} *p*_{*h*})", 8.5), { align: "left" });
eqtag(B.x + 2.5, eY + 0.06, `Eq. (${EQ.G}); quantized: Eq. (${EQ.Gq})`, 1.0);

// =====================================================================
// (c) SWP §4.2
const P = { x: 3.1, y: 1.86, w: 3.7, h: 1.53 };
panel(P.x, P.y, P.w, P.h, C.swp, "Selective Weight Protection (SWP)", "§4.2");
const blocks = {};
DIAG.forEach(r => { const b = r.name.split(".")[0]; (blocks[b] = blocks[b] || []).push(r); });
const show = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
const kX = P.x + 0.12, kY = P.y + 0.34, kw = 0.26, kh = 0.56, kg = 0.035;
show.forEach((b, k) => {
  const rs = blocks[b] || [];
  const sel = rs.some(r => r.kind === "swp");
  rect(kX + k * (kw + kg), kY, kw, kh, { fill: "FFFFFF", line: sel ? C.swp.ln : C.q.ln, lw: sel ? 1.0 : 0.6, round: true, r: 0.03 });
  label(kX + k * (kw + kg), kY + 0.02, kw, 0.12, txt(b, 6, { color: MUTED }));
  const cells = b === "0" ? [{ kind: "proto" }] : rs;
  cells.slice(0, 6).forEach((r, j) => {
    const cx = kX + k * (kw + kg) + 0.04 + (j % 2) * 0.1, cy = kY + 0.16 + Math.floor(j / 2) * 0.125;
    const col = r.kind === "proto" ? "6E6E6E" : r.kind === "swp" ? C.swp.ln : "D9D9D9";
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: cy, w: 0.08, h: 0.095, fill: { color: col }, line: { type: "none" } });
  });
});
const nX = kX + 10 * (kw + kg) + 0.05;
rect(nX, kY, 0.5, kh, { fill: "FFFFFF", line: C.tgr.ln, lw: 0.75, dash: "dash", round: true, r: 0.03 });
label(nX, kY + 0.02, 0.5, 0.12, math("*L*_{G}", 6.5, C.tgr.ln));
[0, 1, 2, 3].forEach(j => s.addShape(pres.shapes.RECTANGLE, { x: nX + 0.08 + (j % 2) * 0.2, y: kY + 0.19 + Math.floor(j / 2) * 0.15, w: 0.13, h: 0.1, fill: { color: C.tgr.fill }, line: { color: C.tgr.ln, width: 0.5 } }));
label(kX, kY + kh + 0.01, 10 * (kw + kg), 0.13, txt("backbone blocks (YOLO-World-S)", 5.8, { color: MUTED }));
label(nX - 0.1, kY + kh + 0.01, 0.7, 0.13, txt("→ TGR", 5.8, { color: C.tgr.ln }));
// 범례 + 규칙 한 줄
const lY = P.y + 1.12;
[[C.swp.ln, math("W8 (*S*^{*})", 6.4)], ["D9D9D9", txt("W4", 6.4)], ["6E6E6E", txt("W8 (first layer, protocol)", 6.4)]].forEach(([c, t], k) => {
  const lx = P.x + 0.12 + [0, 0.72, 1.24][k];
  sq(lx, lY + 0.03, c);
  label(lx + 0.12, lY, 1.3, 0.14, t, { align: "left" });
});
label(P.x + 0.12, P.y + 1.3, 3.5, 0.16, math("*S*^{*}: largest Δ_{*ℓ*} outside *L*_{G} until *w*(*S*^{*}) ≈ *B* = 1.5%; weights kept at 8 bits", 6.4), { align: "left" });

// =====================================================================
// (d) TSM §4.3 + TSM check
const T = { x: 6.94, y: 0.06, w: 3.0, h: 3.33 };
panel(T.x, T.y, T.w, T.h, C.tsm, "Tied Scale Migration (TSM)", "§4.3");
const pY = T.y + 0.42;
rect(T.x + 0.12, pY, 0.95, 0.4, { fill: C.q.fill, line: C.q.ln, round: true });
label(T.x + 0.12, pY + 0.04, 0.95, 0.15, txt("producer", 6.8));
label(T.x + 0.12, pY + 0.2, 0.95, 0.16, math("requantize: *x* / *s*", 7));
boxT(T.x + 1.66, pY - 0.12, 1.2, 0.27, math("*W*_{1} diag(*s*)", 8), { fill: C.tsm.fill, line: C.tsm.ln });
boxT(T.x + 1.66, pY + 0.25, 1.2, 0.27, math("*W*_{2} diag(*s*)", 8), { fill: C.tsm.fill, line: C.tsm.ln });
arrow(T.x + 1.07, pY + 0.2, T.x + 1.66, pY + 0.015);
arrow(T.x + 1.07, pY + 0.2, T.x + 1.66, pY + 0.385);
label(T.x + 1.66, pY + 0.54, 1.2, 0.13, math("consumers *C*(*g*)", 6.2, MUTED));
label(T.x + 0.1, pY + 0.44, 1.2, 0.13, txt("folded in; no run-time op", 5.8, { color: MUTED }));
boxT(T.x + 0.12, T.y + 1.16, 2.76, 0.27, math("*s*_{*c*} = *s*_{*g*(*c*)},   *s*_{*c*}(*α*) = max|*x*_{*c*}|^{*α*} / max|*W*_{:,*c*}|^{1−*α*}", 7.5), { fill: C.tsm.fill, line: C.tsm.ln });
eqtag(T.x + 0.9, T.y + 1.44, `Eq. (${EQ.mig}), (${EQ.tie}); α per unit: Eq. (${EQ.alpha})`, 2.0);
// TSM check
rect(T.x + 0.1, T.y + 1.66, 2.8, 0.0001, { line: C.tsm.ln, lw: 0.5, dash: "dash" });
label(T.x + 0.1, T.y + 1.7, 2.8, 0.18, txt("TSM check", 7.8, { bold: true }));
const cY = T.y + 1.96;
boxT(T.x + 0.12, cY, 0.82, 0.26, math("*θ*̂_{on}", 8, C.tsm.ln), { fill: C.tsm.fill, line: C.tsm.ln, round: true });
boxT(T.x + 0.12, cY + 0.36, 0.82, 0.26, math("*θ*̂_{off}", 8, MUTED), { fill: "FFFFFF", line: C.q.ln, round: true });
boxT(T.x + 1.08, cY, 0.72, 0.26, math("Recon", 7.5), { fill: C.q.fill, line: C.q.ln });
boxT(T.x + 1.08, cY + 0.36, 0.72, 0.26, math("Recon", 7.5), { fill: C.q.fill, line: C.q.ln });
arrow(T.x + 0.94, cY + 0.13, T.x + 1.08, cY + 0.13); arrow(T.x + 0.94, cY + 0.49, T.x + 1.08, cY + 0.49);
label(T.x + 0.9, cY + 0.63, 1.1, 0.13, txt("BRECQ / QDrop / PD-Quant", 5.4, { color: MUTED }));
img(T.x + 2.02, cY - 0.02, 0.56, 0.42, "chk0.jpg");
img(T.x + 2.16, cY + 0.08, 0.56, 0.42, "chk1.jpg");
label(T.x + 1.76, cY + 0.55, 1.18, 0.26, math("*D*_{chk}: 200 train images\n(not in *D*_{cal})", 5.4, MUTED));
const kY2 = T.y + 2.86;
path_([[T.x + 1.8, cY + 0.13], [T.x + 1.9, cY + 0.13], [T.x + 1.9, kY2]], { head: true });
path_([[T.x + 1.8, cY + 0.49], [T.x + 1.9, cY + 0.49]], { head: false });
boxT(T.x + 0.12, kY2, 2.76, 0.27, math("keep *θ*̂_{on} if Flip_{chk}(*θ*̂_{on}) ≤ Flip_{chk}(*θ*̂_{off}), else *θ*̂_{off}", 7.5), { fill: C.tsm.fill, line: C.tsm.ln });
eqtag(T.x + 2.2, kY2 + 0.28, `Eq. (${EQ.check})`);

// =====================================================================
// 패널 사이 흐름
const gapAB = A.x + A.w + 0.08, gapBT = B.x + B.w + 0.07;
path_([[A.x + A.w, lgY + 0.075], [gapAB, lgY + 0.075], [gapAB, B.y + 1.0], [B.x, B.y + 1.0]], { color: C.tgr.ln, dash: "dash", lw: 1.0 });
path_([[A.x + A.w, lgY + 0.245], [gapAB + 0.04, lgY + 0.245], [gapAB + 0.04, P.y + 0.62], [P.x, P.y + 0.62]], { color: C.swp.ln, dash: "dash", lw: 1.0 });
path_([[B.x + B.w, B.y + 1.0], [gapBT, B.y + 1.0], [gapBT, cY + 0.31]], { lw: 1.0, head: false });
path_([[P.x + P.w, P.y + 0.62], [gapBT, P.y + 0.62], [gapBT, cY + 0.31]], { lw: 1.0, head: false });
path_([[gapBT, cY + 0.31], [T.x + 0.04, cY + 0.31]], { lw: 1.0, head: false });
path_([[T.x + 0.04, cY + 0.13], [T.x + 0.04, cY + 0.49]], { lw: 1.0, head: false });
arrow(T.x + 0.04, cY + 0.13, T.x + 0.12, cY + 0.13, { lw: 1.0 });
arrow(T.x + 0.04, cY + 0.49, T.x + 0.12, cY + 0.49, { lw: 1.0 });

s.addNotes("v3 (11-17). 식 번호는 make_overview_v3.js의 EQ에서 바꾼다. 막대 = YOLO-World-S 층별 Δℓ(configs/protect_cache), 축은 6 pp와 29 pp 사이를 끊었다.");
const OUT = process.argv[2] || path.join(HERE, "method_overview_v3.pptx");
pres.writeFile({ fileName: OUT }).then(() => console.log("written", OUT));
