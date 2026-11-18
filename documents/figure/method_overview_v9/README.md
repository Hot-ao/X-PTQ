# Method overview v9 (11-17)
Feedback on v8 applied:
- Whitespace removed: diagnosis panel 2.5 → 2.3 in, smaller Q boxes, canvas 10.16×3.66 → 9.76×3.44 in, SWP panel matched to its content.
- Calibration input: image stack (D_cal) + stacked vocabulary cards (𝒱, the COCO class-name list). Follows the vocabulary/prompt-list style of GLIP Fig. 2 and YOLO-World Fig. 2–3.
- Bars: layers with small Δℓ are dropped from the plot (no ⋯, no white masking). Bars are evenly spaced, and the x axis is one continuous line. Only the y axis and the 12.cv2 bar carry a wavy break mark.
- TGR: a_h placed above ×; + directly above ×, joined by a straight vertical arrow. Before/After both end at the same output z (identical in FP, Eq. 6).
- SWP: "YOLO-World-S backbone blocks" starts above block 0; legend centred below.
- TSM check: for each candidate, 16 FP-confident regions of D_chk are shown as cells (flips, k̂_i ≠ k*_i, in red) → Flip_chk; the selected θ̂_on row has a red frame. The cells are schematic.
Build: `NODE_PATH=../../../tools/pptxgen/node_modules node make_overview_v9.js && python3 merge_v9.py`
