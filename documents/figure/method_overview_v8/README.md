# Method overview v8 (11-17)
Reflects the feedback on v7.
- Lines: each polyline with corners is drawn as a single shape (custom geometry), so the segments cannot come apart.
- Box text is written into the shape itself (shape text). Free text boxes are used only for axis labels, tags and legends.
- Diagnosis: the y axis is broken with a wavy mark (≈). On the x axis, only layers with large Δℓ or that are L_G/S* are kept; omitted runs are shown as "⋯" (segments: 0–13, 25–27, 31–32, 41, 47).
- Descriptions of notation already defined in the manuscript are removed: legends show notation only (L_G, S*), and TGR has no descriptive text.
- SWP: no arrows between blocks, square conv cells, no S* explanation sentence.
- TSM check: one FP-confident region of D_chk. FP = cat, θ̂_on = cat, θ̂_off = dog (flip) → Flip_chk comparison → keep θ̂_on. This is an illustrative example, not a measured case.
- TSM: tsm_numeric_v6.pptx inserted unchanged (merge_v8.py).
Build: `NODE_PATH=../../../tools/pptxgen/node_modules node make_overview_v8.js && python3 merge_v8.py`
