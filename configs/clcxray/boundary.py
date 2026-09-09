"""Stage 2: learn DERA's mask-supervised boundary prior on CLCXray."""

_base_ = [
    "../_base_/model/dera_swin_t.py",
    "../_base_/runtime.py",
    "../_base_/schedules/boundary.py",
]

from dera.data import build_coco_data_config  # noqa: E402

data_root = "data/clcxray/"
load_from = None  # Supplied explicitly by ``dera train --init-checkpoint``.
model = dict(
    type="DERAGroundingDINO",
    training_phase="boundary",
    edge_loss_weight=1.0,
    edge_kernel_size=3,
    edge_focal_alpha=0.75,
    edge_focal_gamma=2.0,
    bbox_head=dict(num_classes=12),
    backbone=dict(type="DERASwinTransformer", edge_head_channels=32, init_cfg=None),
)
_data = build_coco_data_config("clcxray", data_root, stage="boundary", batch_size=1)
train_dataloader = _data["train_dataloader"]
val_cfg = _data["val_cfg"]
test_cfg = _data["test_cfg"]
val_dataloader = _data["val_dataloader"]
val_evaluator = _data["val_evaluator"]
test_dataloader = _data["test_dataloader"]
test_evaluator = _data["test_evaluator"]
del _data
