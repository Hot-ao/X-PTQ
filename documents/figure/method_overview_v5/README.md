# Method overview v5 (11-17)
Restyles the whole overview to match the user's tsm_numeric_v5 look, and inserts the v5 TSM figure into the TSM panel unchanged.
- Style: Arial; ink 222222 / muted 6B6A64; gray lgDash panel borders with no fill; gray flow arrows; warm palette (TGR CC5C73, SWP amber, TSM accent B03A2E).
- Build (2 steps, run from this folder):
  1. `NODE_PATH=../../../tools/pptxgen/node_modules node make_overview_v5.js` → method_overview_v5_base.pptx (the top of the TSM panel is left empty)
  2. `python3 merge_v5.py` → method_overview_v5.pptx. It inserts all shapes from ../method_overview_v3/tsm/tsm_numeric_v5.pptx as one group, applying only a uniform scale (k≈0.68, including font sizes and line widths).
- After you edit tsm_numeric_v5.pptx, rerun step 2 and the change is picked up.
- Check renders: method_overview_v5.pdf, preview.png
