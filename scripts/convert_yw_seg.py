"""YOLO-World-Seg(mmyolo, AILab-CVC) 체크포인트에서 mask 분기만 꺼내 ultralytics 이름으로 저장한다 (11-05).

seghead_finetune 체크포인트는 검출 부분을 얼려 학습해서, M의 검출 가중치는 ultralytics yolov8m-world.pt와 같다
(445/445 텐서 일치 확인). 그래서 mask 분기(seg_preds, proto_pred)만 옮겨 core/model_zoo.attach_world_seg로 붙인다.
내려받은 파일이라 weights_only=True로만 읽는다.

사용: .venv/bin/python -I scripts/convert_yw_seg.py <mmyolo .pth> <출력 .pt>
"""
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
sd = torch.load(src, map_location="cpu", weights_only=True)["state_dict"]
P = "bbox_head.head_module."
out = {}
for k, v in sd.items():
    if not k.startswith(P):
        continue
    k = k[len(P):]
    if k.startswith("seg_preds."):           # seg_preds.{l}.{0,1}.{conv,bn}.*, seg_preds.{l}.2.{weight,bias}
        out["cv5." + k[len("seg_preds."):]] = v
    elif k.startswith("proto_pred."):        # proto_pred.conv{1,2,3}.{conv,bn}.* -> proto.cv{1,2,3}.*
        out["proto.cv" + k[len("proto_pred.conv"):]] = v
torch.save(out, dst)
print(f"{len(out)}개 텐서 -> {dst}")
