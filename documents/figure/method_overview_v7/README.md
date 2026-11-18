# Method overview v7 (11-17)
Each panel follows the figure convention related work uses for that kind of module.
| Panel | Convention | Reference |
|---|---|---|
| Layer-wise diagnosis | per-layer sensitivity bar chart (YOLO-World-S Δℓ, broken axis) | HAWQ-family sensitivity plots |
| TGR | before/after compute graph; quantizers drawn as nodes, before in red, after in green | RepQ-ViT Fig. 1, HAWQ-V3 Fig. 1, quantization white paper Fig. 4 |
| SWP | bit-width shown per layer on the architecture | HAWQ-V2 Fig. 1 |
| TSM | numeric matrices; tsm_numeric_v6.pptx inserted unchanged | SmoothQuant |
| TSM check | two candidates → Flip_chk measured on D_chk → keep the lower (the bars are schematic, with no values) | — |

Build (from this folder): `NODE_PATH=../../../tools/pptxgen/node_modules node make_overview_v7.js && python3 merge_v7.py`
