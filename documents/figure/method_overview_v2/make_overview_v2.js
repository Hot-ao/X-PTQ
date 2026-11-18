// 방법 개요 그림 v2 (11-16). RegPTQ Fig.5 스타일: 점선 패널 + 패널 제목, 평면 블록, 수식 상자에 식 번호.
// 패널: (a) 진단 §3.2 Eq.(3)  (b) TGR §4.1 Eq.(6)(7)  (c) SWP §4.2 Eq.(9)  (d) TSM §4.3 Eq.(11)(12) + 사전 검사 Eq.(15)
// 데이터: diag_yolov8s-world.json(YOLO-World-S 층별 Δℓ, configs/protect_cache), img/(COCO train2017 calibration·검사 예시)
// 사용(저장소 루트): NODE_PATH=tools/pptxgen/node_modules node documents/figure/method_overview_v2/make_overview_v2.js
// 출력: documents/figure/method_overview_v2/method_overview_v2.pptx (PowerPoint에서 PDF로 내보내 논문에 쓴다)
const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");
const HERE = __dirname;
const DIAG = JSON.parse(fs.readFileSync(path.join(HERE, "diag_yolov8s-world.json"), "utf8"));

// 식 번호(11-16 원고 기준). 원고의 식 순서가 바뀌면 여기만 고친다.
const EQ = { diag: 3, fusionq: 6, G: 7, P: 9, mig: 11, tie: 12, check: 15 };

const W = 10, H = 4.12;
const INK = "222222", MUTED = "6B6B6B", LINE = "555555", PALE = "F4F4F4";
const C = {
  diag: { ln: "A07A12", fill: "FBF4E2" },   // 진단(황토)
  tgr: { ln: "2E7D4F", fill: "E6F2EA" },    // TGR(초록)
  swp: { ln: "6A3FA0", fill: "EFE7F7" },    // SWP(보라)
  tsm: { ln: "2F6DB5", fill: "E7EFF9" },    // TSM(파랑)
  q: { ln: "8A8A8A", fill: "EDEDED" },      // 양자화된 블록(회색)
};
const MATH = "Times New Roman", SANS = "Arial";

const pres = new pptxgen();
pres.defineLayout({ name: "OV", width: W, height: H });
pres.layout = "OV";
pres.title = "DeFiQ method overview v2";
const s = pres.addSlide();
s.background = { color: "FFFFFF" };

// ---- 수식 마크업: _{..} 아래첨자, ^{..} 위첨자, *..* 이탤릭(변수). 나머지는 똑바로. ----
function math(src, sz, color = INK, base = {}) {
  const runs = [];
  let i = 0, ital = false;
  const push = (t, o = {}) => { if (t) runs.push({ text: t, options: Object.assign({ fontFace: MATH, fontSize: sz, color, italic: ital }, base, o) }); };
  let buf = "";
  while (i < src.length) {
    const ch = src[i];
    if ((ch === "_" || ch === "^") && src[i + 1] === "{") {
      push(buf); buf = "";
      let d = 1, j = i + 2;
      while (j < src.length && d) { if (src[j] === "{") d++; if (src[j] === "}") d--; j++; }
      const inner = src.slice(i + 2, j - 1);
      // 안쪽의 *..* 처리
      inner.split(/(\*[^*]+\*)/).forEach(part => {
        if (!part) return;
        const it = part.startsWith("*") && part.endsWith("*");
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
function txt(t, sz, o = {}) { return [{ text: t, options: Object.assign({ fontFace: SANS, fontSize: sz, color: INK }, o) }]; }

// ---- 도형 ----
function rect(x, y, w, h, o = {}) {
  s.addShape(o.round ? pres.shapes.ROUNDED_RECTANGLE : pres.shapes.RECTANGLE, Object.assign({
    x, y, w, h,
    fill: o.fill ? { color: o.fill } : { type: "none" },
    line: o.line === null ? { type: "none" } : { color: o.line || LINE, width: o.lw || 0.75, dashType: o.dash || "solid" },
  }, o.round ? { rectRadius: o.r || 0.05 } : {}, o.name ? { objectName: o.name } : {}));
}
function label(x, y, w, h, runs, o = {}) {
  s.addText(runs, Object.assign({ x, y, w, h, margin: 0, align: "center", valign: "middle", isTextBox: true, fit: "none" }, o));
}
function boxT(x, y, w, h, runs, o = {}) { rect(x, y, w, h, o); label(x, y, w, h, runs, { align: o.align || "center" }); }
// 직선 화살표(가로/세로/대각선 모두)
function arrow(x1, y1, x2, y2, o = {}) {
  const x = Math.min(x1, x2), y = Math.min(y1, y2), w = Math.abs(x2 - x1), h = Math.abs(y2 - y1);
  s.addShape(pres.shapes.LINE, {
    x, y, w: Math.max(w, 0.0001), h: Math.max(h, 0.0001),
    flipH: x2 < x1, flipV: y2 < y1,
    line: { color: o.color || LINE, width: o.lw || 0.9, dashType: o.dash || "solid", endArrowType: o.head === false ? undefined : "triangle" },
  });
}
// 꺾인 화살표: 점 목록을 차례로 잇고 마지막에만 화살촉
function path_(pts, o = {}) {
  for (let k = 0; k < pts.length - 1; k++)
    arrow(pts[k][0], pts[k][1], pts[k + 1][0], pts[k + 1][1], Object.assign({}, o, { head: k === pts.length - 2 && o.head !== false }));
}
function panel(x, y, w, h, col, title, sec) {
  rect(x, y, w, h, { line: col.ln, lw: 1.0, dash: "dash", round: true, r: 0.06 });
  label(x, y + 0.05, w, 0.24, [
    { text: title, options: { fontFace: SANS, fontSize: 9, bold: true, color: INK } },
    { text: "  " + sec, options: { fontFace: SANS, fontSize: 7.5, color: MUTED } },
  ]);
}
function eqtag(x, y, n, w = 0.42) { label(x, y, w, 0.14, txt(`Eq. (${n})`, 6, { color: MUTED, italic: false }), { align: "right" }); }
function chip(x, y, w, t, col = C.q) { boxT(x, y, w, 0.17, txt(t, 6.2, { color: INK }), { fill: col.fill, line: col.ln, lw: 0.6, round: true, r: 0.04 }); }
function img(x, y, w, h, f) { s.addImage({ path: path.join(HERE, "img", f), x, y, w, h }); rect(x, y, w, h, { line: "FFFFFF", lw: 0.75 }); }
function circ(x, y, d, t) { s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: "FFFFFF" }, line: { color: LINE, width: 0.75 } }); label(x, y - 0.005, d, d, txt(t, 7, { bold: true })); }

// =====================================================================
// (a) 진단 §3.2
const A = { x: 0.06, y: 0.06, w: 2.72, h: 3.72 };
panel(A.x, A.y, A.w, A.h, C.diag, "Leakage-free Diagnosis", "§3.2");
// 입력: calibration 이미지 + COCO 어휘
["cal0.jpg", "cal1.jpg", "cal2.jpg"].forEach((f, k) => img(A.x + 0.14 + k * 0.1, A.y + 0.38 + k * 0.06, 0.56, 0.42, f));
label(A.x + 0.08, A.y + 0.92, 0.95, 0.14, txt("calibration images", 6.2, { color: MUTED }));
const chipsX = A.x + 1.15;
[["person", 0.44], ["giraffe", 0.44], ["zebra", 0.38], ["train", 0.36]].reduce((xx, [t, w], k) => {
  chip(xx, A.y + 0.42 + Math.floor(k / 2) * 0.23, w, t, { fill: "FFF4E5", ln: "C77A1E" });
  return k % 2 === 1 ? chipsX : xx + w + 0.06;
}, chipsX);
label(chipsX, A.y + 0.89, 1.5, 0.14, txt("COCO vocab. (80 classes)", 6.2, { color: MUTED }), { align: "left" });
label(chipsX + 0.92, A.y + 0.65, 0.5, 0.14, txt("…", 7, { color: MUTED }), { align: "left" });
// 두 모델
const mY = A.y + 1.22;
boxT(A.x + 0.14, mY, 1.12, 0.36, math("*Q*^{8}  (W8A8)", 8), { fill: C.q.fill, line: C.q.ln, round: true });
boxT(A.x + 1.46, mY, 1.18, 0.36, math("*Q*^{8}_{*ℓ*→4}", 8).concat(txt("  (layer ℓ → W4)", 6, { color: MUTED })), { fill: C.q.fill, line: C.q.ln, round: true });
arrow(A.x + 0.6, A.y + 1.07, A.x + 0.6, mY);
arrow(A.x + 2.0, A.y + 1.07, A.x + 2.0, mY);
// 식 (3)
const fY = mY + 0.5;
arrow(A.x + 0.7, mY + 0.36, A.x + 0.7, fY);
arrow(A.x + 2.05, mY + 0.36, A.x + 2.05, fY);
boxT(A.x + 0.14, fY, 2.5, 0.32, math("Δ_{*ℓ*} = Flip_{cal}(*Q*^{8}_{*ℓ*→4}) − Flip_{cal}(*Q*^{8})", 8.5), { fill: C.diag.fill, line: C.diag.ln });
eqtag(A.x + 2.22, fY + 0.33, EQ.diag);
// 막대: 층별 Δℓ (네트워크 순서), 12.cv2는 잘라서 값 표기
const bX = A.x + 0.3, bY = A.y + 3.42, bW = 2.32, bHmax = 0.95, cap = 5.5;
const n = DIAG.length, bw = bW / n;
rect(bX - 0.02, bY, bW + 0.04, 0.0001, { line: LINE, lw: 0.6 });
DIAG.forEach((r, k) => {
  const v = Math.max(r.delta, 0), hh = Math.min(v, cap) / cap * bHmax;
  if (hh < 0.004) return;
  const col = r.kind === "fusion" ? C.tgr.ln : r.kind === "swp" ? C.swp.ln : "B5B5B5";
  s.addShape(pres.shapes.RECTANGLE, { x: bX + k * bw + bw * 0.12, y: bY - hh, w: bw * 0.76, h: hh, fill: { color: col }, line: { type: "none" } });
  if (r.delta > cap) {   // 잘린 막대 표시
    rect(bX + k * bw - 0.01, bY - hh + 0.06, bw + 0.02, 0.025, { fill: "FFFFFF", line: null });
    label(bX + k * bw - 0.25, bY - hh - 0.16, 0.6, 0.14, txt(`${r.name.replace(".conv", "")} (${r.delta.toFixed(1)})`, 6, { color: C.tgr.ln, bold: true }));
  }
});
label(A.x + 0.06, bY - bHmax - 0.05, 0.22, bHmax, math("Δ_{*ℓ*}", 7.5, MUTED), { valign: "top", align: "left" });
label(bX, bY + 0.03, bW, 0.14, txt("layers in network order (YOLO-World-S)", 6, { color: MUTED }));
// 막대 범례
const lgX = A.x + 1.38, lgY = bY - bHmax + 0.04;
[[C.tgr.ln, "fusion conv → TGR"], [C.swp.ln, "top Δℓ (1.5%) → SWP"]].forEach(([c, t], k) => {
  s.addShape(pres.shapes.RECTANGLE, { x: lgX, y: lgY + k * 0.17 + 0.03, w: 0.09, h: 0.09, fill: { color: c }, line: { type: "none" } });
  label(lgX + 0.13, lgY + k * 0.17, 1.2, 0.15, txt(t, 6.2), { align: "left" });
});

// =====================================================================
// (b) TGR §4.1
const B = { x: 2.96, y: 0.06, w: 3.86, h: 2.0 };
panel(B.x, B.y, B.w, B.h, C.tgr, "Text-Gate Relocation (TGR)", "§4.1");
const r1 = B.y + 0.48, r2 = B.y + 0.88;            // 시각 분기 / 텍스트 분기 줄
const bh = 0.28, lx0 = B.x + 0.1, lw0 = 0.62;
boxT(lx0, r1, lw0, bh, math("*y*", 9).concat(txt("  visual", 5.8, { color: MUTED })), { fill: "FFFFFF", line: LINE });
boxT(lx0, r2, lw0, bh, math("*p*_{*h*}", 9).concat(txt("  text", 5.8, { color: MUTED })), { fill: "FFFFFF", line: LINE });
const qx = B.x + 0.84, qw = 0.78;
boxT(qx, r1, qw, bh, math("*Q_{a}*(*y*)", 8.5), { fill: C.q.fill, line: C.q.ln });
boxT(qx, r2, qw, bh, math("*Q_{a}*(*p*_{*h*})", 8.5), { fill: C.q.fill, line: C.q.ln });
const wx = qx + qw + 0.14, ww = 0.86;
boxT(wx, r1, ww, bh, math("*Q_{w}*(*W*_{main})", 8.5), { fill: C.tgr.fill, line: C.tgr.ln, lw: 1.0 });
boxT(wx, r2, ww, bh, math("*Q_{w}*(*W*_{*h*})", 8.5), { fill: C.tgr.fill, line: C.tgr.ln, lw: 1.0 });
arrow(lx0 + lw0, r1 + bh / 2, qx, r1 + bh / 2); arrow(lx0 + lw0, r2 + bh / 2, qx, r2 + bh / 2);
arrow(qx + qw, r1 + bh / 2, wx, r1 + bh / 2); arrow(qx + qw, r2 + bh / 2, wx, r2 + bh / 2);
// 게이트(⊙)는 conv 뒤로, 합(+), SiLU
const d = 0.2, gx = wx + ww + 0.14, sx = gx + 0.32, slx = sx + d + 0.12;
circ(gx, r2 + bh / 2 - d / 2, d, "⊙");
circ(sx, r1 + bh / 2 - d / 2, d, "+");
arrow(wx + ww, r2 + bh / 2, gx, r2 + bh / 2);
arrow(wx + ww, r1 + bh / 2, sx, r1 + bh / 2);
path_([[gx + d, r2 + bh / 2], [sx + d / 2, r2 + bh / 2], [sx + d / 2, r1 + bh / 2 + d / 2]]);
boxT(slx, r1, 0.4, bh, txt("SiLU", 7), { fill: "FFFFFF", line: LINE });
arrow(sx + d, r1 + bh / 2, slx, r1 + bh / 2);
label(wx - 0.1, r1 - 0.16, 1.6, 0.14, txt("separate scale per branch", 6.2, { color: C.tgr.ln, italic: true }));
// 텍스트 → 게이트 a_h (위치별 값, conv 뒤에서 곱한다)
const aY = r2 + 0.4;
chip(lx0, aY + 0.03, 0.62, "text emb.", { fill: "FFF4E5", ln: "C77A1E" });
boxT(qx, aY, 1.3, 0.23, math("gate *a*_{*h*} (max sim.)", 7.5), { fill: "FFF4E5", line: "C77A1E" });
arrow(lx0 + 0.62, aY + 0.115, qx, aY + 0.115, { color: "C77A1E" });
path_([[qx + 1.3, aY + 0.115], [gx + d / 2, aY + 0.115], [gx + d / 2, r2 + bh / 2 + d / 2]], { color: "C77A1E" });
// 식 (6) → (7)
const eY = B.y + 1.66;
label(B.x + 0.1, eY - 0.16, 3.0, 0.14, math("before: *Q_{a}*([*y*; *a*⊙*p*]) shares one scale across branches, Eq. (" + EQ.fusionq + ")", 6.2, MUTED), { align: "left" });
boxT(B.x + 0.1, eY, 3.66, 0.27,
  math("*W* [*y*; *a*⊙*p*]  =  *W*_{main} *y* + Σ_{*h*} *a*_{*h*} ⊙ (*W*_{*h*} *p*_{*h*})", 8.5),
  { fill: C.tgr.fill, line: C.tgr.ln });
eqtag(B.x + 3.34, eY - 0.15, EQ.G);

// =====================================================================
// (c) SWP §4.2
const P = { x: 2.96, y: 2.14, w: 3.86, h: 1.64 };
panel(P.x, P.y, P.w, P.h, C.swp, "Selective Weight Protection (SWP)", "§4.2");
// backbone 블록 0–9: 각 블록의 conv를 작은 칸으로. 보라 = W8(SWP), 회색 = W4, 진회색 = W8(프로토콜, 첫 층)
const blocks = {};
DIAG.forEach(r => { const b = r.name.split(".")[0]; (blocks[b] = blocks[b] || []).push(r); });
const show = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
const kX = P.x + 0.12, kY = P.y + 0.38, kw = 0.27, kh = 0.6, kg = 0.04;
show.forEach((b, k) => {
  const rs = blocks[b] || [];
  const sel = rs.some(r => r.kind === "swp");
  rect(kX + k * (kw + kg), kY, kw, kh, { fill: "FFFFFF", line: sel ? C.swp.ln : C.q.ln, lw: sel ? 1.0 : 0.6, round: true, r: 0.03 });
  label(kX + k * (kw + kg), kY + 0.02, kw, 0.13, txt(b, 6.2, { color: MUTED }));
  const cells = b === "0" ? [{ kind: "proto" }] : rs;
  cells.slice(0, 6).forEach((r, j) => {
    const cx = kX + k * (kw + kg) + 0.045 + (j % 2) * 0.1, cy = kY + 0.18 + Math.floor(j / 2) * 0.13;
    const col = r.kind === "proto" ? "6E6E6E" : r.kind === "swp" ? C.swp.ln : "D9D9D9";
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: cy, w: 0.08, h: 0.1, fill: { color: col }, line: { type: "none" } });
  });
});
// neck: 텍스트 융합 conv는 TGR이 맡아 SWP 후보에서 뺀다
const nX = kX + 10 * (kw + kg) + 0.06;
rect(nX, kY, 0.5, kh, { fill: "FFFFFF", line: C.tgr.ln, lw: 0.75, dash: "dash", round: true, r: 0.03 });
label(nX, kY + 0.02, 0.5, 0.13, txt("fusion", 6, { color: C.tgr.ln }));
[0, 1, 2, 3].forEach(j => s.addShape(pres.shapes.RECTANGLE, { x: nX + 0.08 + (j % 2) * 0.2, y: kY + 0.2 + Math.floor(j / 2) * 0.15, w: 0.13, h: 0.1, fill: { color: C.tgr.fill }, line: { color: C.tgr.ln, width: 0.5 } }));
label(nX - 0.05, kY + kh + 0.01, 0.6, 0.12, txt("→ TGR", 5.8, { color: C.tgr.ln }));
label(kX, kY + kh + 0.01, 10 * (kw + kg), 0.13, txt("backbone blocks (YOLO-World-S)", 5.8, { color: MUTED }));
// 칸 범례
const cl = [["W8 (SWP)", C.swp.ln], ["W4", "D9D9D9"], ["W8 (protocol)", "6E6E6E"]];
cl.forEach(([t, c], k) => {
  const lx = P.x + 0.12 + k * 0.82;
  s.addShape(pres.shapes.RECTANGLE, { x: lx, y: P.y + 1.18 + 0.03, w: 0.08, h: 0.08, fill: { color: c }, line: { type: "none" } });
  label(lx + 0.11, P.y + 1.18, 0.75, 0.14, txt(t, 6), { align: "left" });
});
// 식 (9)
boxT(P.x + 0.12, P.y + 1.34, 3.72, 0.26,
  math("*S*^{*} = argmax Σ_{*ℓ*∈*S*} Δ_{*ℓ*}  over  *S* ⊆ *L* ∖ *L*_{G},   s.t.  *w*(*S*) ≤ 1.1*B*,  *B* = 1.5%", 8),
  { fill: C.swp.fill, line: C.swp.ln });
eqtag(P.x + 3.42, P.y + 1.19, EQ.P);

// =====================================================================
// (d) TSM §4.3 + 사전 검사
const T = { x: 7.0, y: 0.06, w: 2.94, h: 3.72 };
panel(T.x, T.y, T.w, T.h, C.tsm, "Tied Scale Migration (TSM)", "§4.3");
// 생산 층 → x/s → 소비 층들 (s 하나를 공유)
const pY = T.y + 0.42;
rect(T.x + 0.12, pY, 0.9, 0.4, { fill: C.q.fill, line: C.q.ln, round: true });
label(T.x + 0.12, pY + 0.04, 0.9, 0.15, txt("producer", 6.8));
label(T.x + 0.12, pY + 0.2, 0.9, 0.16, math("requant ÷ *s*", 7.5));
boxT(T.x + 1.62, pY - 0.12, 1.2, 0.27, math("*W*_{1} diag(*s*)", 8), { fill: C.tsm.fill, line: C.tsm.ln });
boxT(T.x + 1.62, pY + 0.25, 1.2, 0.27, math("*W*_{2} diag(*s*)", 8), { fill: C.tsm.fill, line: C.tsm.ln });
arrow(T.x + 1.02, pY + 0.2, T.x + 1.62, pY + 0.015);
arrow(T.x + 1.02, pY + 0.2, T.x + 1.62, pY + 0.385);
label(T.x + 1.02, pY + 0.27, 0.6, 0.14, math("*x*/*s*", 7.5, C.tsm.ln));
label(T.x + 1.62, pY + 0.54, 1.2, 0.13, txt("consumers j ∈ C(g) share one s", 5.8, { color: MUTED }));
label(T.x + 0.1, pY + 0.42, 0.95, 0.24, txt("folded into requant:\nno extra op at runtime", 5.6, { color: MUTED }), { valign: "top" });
boxT(T.x + 0.12, T.y + 1.18, 2.74, 0.27, math("*Wx* = (*W* diag(*s*)) (diag(*s*)^{−1} *x*),   *s*_{*c*} = *s*_{*g*(*c*)}", 8), { fill: C.tsm.fill, line: C.tsm.ln });
eqtag(T.x + 2.2, T.y + 1.46, `${EQ.mig}), (${EQ.tie}`, 0.66);
// 사전 검사: TSM on / off 두 후보 → 재구성 → 검사 flip이 낮은 쪽
rect(T.x + 0.1, T.y + 1.66, 2.78, 0.0001, { line: C.tsm.ln, lw: 0.5, dash: "dash" });
label(T.x + 0.1, T.y + 1.7, 2.78, 0.18, [{ text: "TSM check", options: { fontFace: SANS, fontSize: 7.8, bold: true, color: INK } },
  { text: "  (on or off, decided before evaluation)", options: { fontFace: SANS, fontSize: 6, color: MUTED } }]);
const cY = T.y + 2.0;
boxT(T.x + 0.12, cY, 0.92, 0.26, txt("TSM on", 7, { bold: true, color: C.tsm.ln }), { fill: C.tsm.fill, line: C.tsm.ln, round: true });
boxT(T.x + 0.12, cY + 0.36, 0.92, 0.26, txt("TSM off", 7, { bold: true, color: MUTED }), { fill: "FFFFFF", line: C.q.ln, round: true });
boxT(T.x + 1.2, cY, 0.72, 0.26, math("Recon(·)", 7.5), { fill: C.q.fill, line: C.q.ln });
boxT(T.x + 1.2, cY + 0.36, 0.72, 0.26, math("Recon(·)", 7.5), { fill: C.q.fill, line: C.q.ln });
arrow(T.x + 1.04, cY + 0.13, T.x + 1.2, cY + 0.13); arrow(T.x + 1.04, cY + 0.49, T.x + 1.2, cY + 0.49);
label(T.x + 0.9, cY + 0.63, 1.2, 0.13, txt("BRECQ / QDrop / PD-Quant", 5.4, { color: MUTED }));
// 검사 이미지
["chk0.jpg", "chk1.jpg", "chk2.jpg"].forEach((f, k) => img(T.x + 2.08 + k * 0.08, cY - 0.02 + k * 0.05, 0.6, 0.45, f));
label(T.x + 1.98, cY + 0.55, 0.95, 0.24, txt("200 train images\nnot in calibration", 5.6, { color: MUTED }));
// 비교 식 (15)
const kY2 = T.y + 2.9;
path_([[T.x + 1.92, cY + 0.13], [T.x + 1.98, cY + 0.13], [T.x + 1.98, kY2]], { head: true });
path_([[T.x + 1.92, cY + 0.49], [T.x + 1.98, cY + 0.49]], { head: false });
boxT(T.x + 0.12, kY2, 2.74, 0.27, math("Flip_{chk}(on) ≤ Flip_{chk}(off)  ⇒  keep TSM", 8), { fill: C.tsm.fill, line: C.tsm.ln });
eqtag(T.x + 2.44, kY2 + 0.28, EQ.check);
boxT(T.x + 0.92, T.y + 3.42, 1.14, 0.26, txt("DeFiQ model", 7.5, { bold: true }), { fill: "FFFFFF", line: INK, lw: 1.0, round: true });
arrow(T.x + 1.49, kY2 + 0.27, T.x + 1.49, T.y + 3.42);

// =====================================================================
// 패널 사이 흐름
// 진단 → TGR(융합 conv 위치), 진단 → SWP(보호 층 S*)
const gapAB = A.x + A.w + 0.09, gapBT = B.x + B.w + 0.09;
path_([[A.x + A.w, lgY + 0.075], [gapAB, lgY + 0.075], [gapAB, B.y + 1.1], [B.x, B.y + 1.1]], { color: C.tgr.ln, dash: "dash", lw: 1.0 });
path_([[A.x + A.w, lgY + 0.245], [gapAB + 0.04, lgY + 0.245], [gapAB + 0.04, P.y + 0.68], [P.x, P.y + 0.68]], { color: C.swp.ln, dash: "dash", lw: 1.0 });
// TGR·SWP를 적용한 모델 → TSM 두 후보(갈래)
path_([[B.x + B.w, B.y + 1.0], [gapBT, B.y + 1.0], [gapBT, T.y + 2.31]], { lw: 1.0, head: false });
path_([[P.x + P.w, P.y + 0.68], [gapBT, P.y + 0.68], [gapBT, T.y + 2.31]], { lw: 1.0, head: false });
path_([[gapBT, T.y + 2.31], [T.x + 0.05, T.y + 2.31]], { lw: 1.0, head: false });
path_([[T.x + 0.05, T.y + 2.13], [T.x + 0.05, T.y + 2.49]], { lw: 1.0, head: false });
arrow(T.x + 0.05, T.y + 2.13, T.x + 0.12, T.y + 2.13, { lw: 1.0 });
arrow(T.x + 0.05, T.y + 2.49, T.x + 0.12, T.y + 2.49, { lw: 1.0 });

// 아래 범례(화살표 종류)
const LY = 3.88;
[[LINE, "solid", "model / data flow"], [C.tgr.ln, "dash", "diagnosis selects where TGR / SWP apply"], ["C77A1E", "solid", "text-embedding path"]].reduce((lx, [c, dsh, t]) => {
  arrow(lx, LY + 0.08, lx + 0.32, LY + 0.08, { color: c, dash: dsh, lw: 1.0 });
  label(lx + 0.38, LY, 2.6, 0.16, txt(t, 6.4), { align: "left" });
  return lx + 0.48 + t.length * 0.05;
}, 0.2);
label(6.2, LY, 3.7, 0.16, txt("All choices use calibration / train data only; no evaluation data or LVIS vocab.", 6.2, { color: MUTED, italic: true }), { align: "right" });

s.addNotes("v2 (11-16). 식 번호는 make_overview_v2.js의 EQ에서 바꾼다. 막대 = YOLO-World-S 층별 Δℓ(configs/protect_cache), 12.cv2는 잘라서 값 표기. 이미지 = COCO train2017 calibration(정렬 앞 256장) / 검사(256–455번) 예시.");
const OUT = process.argv[2] || path.join(HERE, "method_overview_v2.pptx");
pres.writeFile({ fileName: OUT }).then(() => console.log("written", OUT));
