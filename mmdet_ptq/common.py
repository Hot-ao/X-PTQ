"""11-15 InlierQ 재현용 mmdet 공통 도우미(.venv_mmdet 전용; torch 2.1 + mmcv 2.1 + mmdet 3.3).
체크포인트는 내려받은 파일이라 torch.load(weights_only=True)로만 연다."""
import os, copy, torch
from mmengine.config import Config
from mmengine.runner import Runner

MMDET_CFG = os.path.join(os.path.dirname(__import__("mmdet").__file__), ".mim", "configs")
CKPT = "/data/taeho/mmdet_ckpt"
COCO = "/data/taeho/coco_datasets/"
MODELS = {
    "retina_rx32": ("regnet/retinanet_regnetx-3.2GF_fpn_1x_coco.py",
                    "retinanet_regnetx-3.2GF_fpn_1x_coco_20200520_163141-cb1509e8.pth"),
    "frcnn_r50": ("faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py", "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"),
    "frcnn_r50_iou": ("faster_rcnn/faster-rcnn_r50_fpn_iou_1x_coco.py",
                      "faster_rcnn_r50_fpn_iou_1x_coco_20200506_095954-938e81f0.pth"),
    "frcnn_r50_caffe": ("faster_rcnn/faster-rcnn_r50-caffe_fpn_1x_coco.py",
                        "faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth"),
}


def make_cfg(name, work_dir, batch=8, workers=4):
    cfg_path, _ = MODELS[name]
    cfg = Config.fromfile(os.path.join(MMDET_CFG, cfg_path))
    cfg.work_dir = work_dir
    cfg.load_from = None
    for k in ("test_dataloader", "val_dataloader", "train_dataloader"):
        dl = cfg.get(k)
        if dl is not None:
            dl.dataset.data_root = COCO
    for k in ("test_evaluator", "val_evaluator"):
        ev = cfg.get(k)
        if ev is not None:
            ev.ann_file = COCO + "annotations/instances_val2017.json"
    cfg.test_dataloader.batch_size = batch
    cfg.test_dataloader.num_workers = workers
    cfg.default_hooks.logger.interval = 200
    cfg.default_hooks.visualization = dict(type="DetVisualizationHook", draw=False)
    cfg.vis_backends = [dict(type="LocalVisBackend")]
    cfg.visualizer.vis_backends = cfg.vis_backends
    return cfg


def load_weights(model, name):
    sd = torch.load(os.path.join(CKPT, MODELS[name][1]), map_location="cpu", weights_only=True)["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])


def build_runner(name, work_dir, batch=8):
    runner = Runner.from_cfg(make_cfg(name, work_dir, batch))
    load_weights(runner.model, name)
    return runner
