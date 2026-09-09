"""Evaluation and mask-free inference with the final STCray DERA model."""

_base_ = [
    "../_base_/model/dera_swin_t.py",
    "../_base_/runtime.py",
]

from dera.data import build_coco_data_config  # noqa: E402

data_root = "data/stcray/"
load_from = None
model = dict(
    type="DERAGroundingDINO",
    training_phase="residual",
    bbox_head=dict(num_classes=20),
    backbone=dict(type="DERASwinTransformer", edge_head_channels=32, init_cfg=None),
)
_data = build_coco_data_config("stcray", data_root, stage="inference", batch_size=1)
train_cfg = _data["train_cfg"]
train_dataloader = _data["train_dataloader"]
val_dataloader = _data["val_dataloader"]
test_dataloader = _data["test_dataloader"]
val_evaluator = _data["val_evaluator"]
test_evaluator = _data["test_evaluator"]
del _data
