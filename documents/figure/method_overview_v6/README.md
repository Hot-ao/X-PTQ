# Method overview v6 (11-17)
Applies the tsm_numeric_v6 approach (a small worked example plus its equation, heat cells, one red highlight, gray arrows) to every panel.
- TSM: inserts ../method_overview_v3/tsm/tsm_numeric_v6.pptx unchanged as one group, scaled uniformly by 0.91.
- Diagnosis: actual YOLO-World-S Δℓ values (configs/protect_cache yolov8s-world_..._cls.json, the coco column) for 8 layers in network order. 12.cv2 = 30.8 is highlighted.
- SWP: the actual greedy selection on the same model, using Δℓ and weight fraction w (from mparam). 9.cv2 is skipped because it exceeds 1.1B (4.21%). Stops at Σw = 1.59% ≈ B.
- TGR: a toy example (y=[1,-2,3,2], p_h=[2,-1], a_h=4). Before, a single range of 8; after, each branch has its own range (3 and 2).
- Build (from this folder): `NODE_PATH=../../../tools/pptxgen/node_modules node make_overview_v6.js && python3 merge_v6.py`
- If you edit tsm_numeric_v6.pptx, running only merge_v6.py will pick up the change.
