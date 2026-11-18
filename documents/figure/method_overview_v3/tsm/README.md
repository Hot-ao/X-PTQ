# TSM figure (shared-scale constraint), 11-17

This folder holds a figure that shows TSM's shared-scale constraint (§4.3, "Shared-scale constraint") as a SmoothQuant-style numeric example.

## Latest version (v4, based on the user's sketch)
| File | Contents |
|---|---|
| `tsm_numeric_v4.pptx` | Editable PowerPoint (native shapes) |
| `make_tsm_numeric_v4.js` | Generator: `NODE_PATH=tools/pptxgen/node_modules node documents/figure/method_overview_v3/tsm/make_tsm_numeric_v4.js` |
| `tsm_numeric_v4.pdf`, `tsm_numeric_v4_preview.png` | Rendered checks (LibreOffice) |

Layout: two groups sit side by side, [max|x_c| over x] and [C(g): tied max over W^(1), W^(2)]. An outer box around both makes them the input to s.
The two statistics rows, s, and x/s lie on one line. From s, "÷s" goes to x/s and "diag(s)" goes to the stacked W^(j)diag(s).

## Previous version (v3)
| File | Contents |
|---|---|
| `tsm_numeric_v3.pptx` | Editable PowerPoint, built from native shapes. Cell values and colors can be edited directly. |
| `make_tsm_numeric_v3.js` | Generator for the pptx. Run from the repo root: `NODE_PATH=tools/pptxgen/node_modules node documents/figure/method_overview_v3/tsm/make_tsm_numeric_v3.js` |
| `tsm_numeric_v3.png` / `_transparent.png` / `.pdf` | matplotlib version |
| `tsm_numeric_v3.py` | Generator for the matplotlib version: `.venv/bin/python documents/figure/method_overview_v3/tsm/tsm_numeric_v3.py` |

The numbers use α = 0.5. The x column maxima are [2, 16, 2, 9]. Taking the max over both consumers W⁽¹⁾ and W⁽²⁾ gives [2, 4, 2, 1], so s = [1, 2, 1, 3].
The red column shows the effect of the constraint: looking at W⁽¹⁾ alone would give s₂ = √(16/1) = 4, but tying with W⁽²⁾ gives s₂ = √(16/4) = 2.
The figure omits residual-add grouping (producers merged into one group g) and the floor ε. Those are described in the text.

## Notation used in the figure (matches 4_method.tex)
x, W^(j), C(g), max|x_c|, max_j max|W^(j)_{:,c}| (eq:tie), s_c(α) (eq:mig), x/s, W^(j)diag(s).
The figure contains no equation numbers, so it does not need redrawing when equations are renumbered.

## drafts/
Discarded drafts from the design discussion.
- `producer_consumer_*.png`: 3D tensor style (draft → combined fan-out/residual version → annotated → notation only)
- `tsm_numeric.png`, `tsm_numeric_v2.png`: earlier layouts of the numeric example (v2 had curved arrows)
- `draw*.py`, `draw_sq*.py`: their scripts. Output paths still point at the scratchpad.
