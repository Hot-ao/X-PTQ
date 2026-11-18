# Method overview v4 (11-17)

v3 plus the following change: the top of the TSM panel now uses the numeric example from TSM v5 (`../method_overview_v3/tsm/tsm_numeric_v5.pptx`).
- Removed: the producer/consumer schematic and the s-formula box.
- Added in their place: [max|x_c| over x] and [C(g): tied max over W^(1), W^(2)] → s → a bus from s → x/s and W^(j)diag(s) (÷s and diag(s) labels).
- The highlight color uses the TSM panel color (blue), so the panel matches the other panels' color scheme. The heatmap colors are the same as v5.
- The TSM check part moved down by about 0.2 inch to make room. Its content is unchanged.

Build (repo root): `NODE_PATH=tools/pptxgen/node_modules node documents/figure/method_overview_v4/make_overview_v4.js`
Equation numbers are set in the `EQ` object in the script. Rendered check: `method_overview_v4.pdf`, `preview.png`.
