"""Stage 3: tune DERA's zero-initialized residuals on CLCXray."""

_base_ = [
    "../_base_/model/dera_swin_t.py",
    "../_base_/runtime.py",
    "../_base_/schedules/residual.py",
]

from dera.data import build_coco_data_config  # noqa: E402

data_root = "data/clcxray/"
load_from = None  # Supplied explicitly by ``dera train --init-checkpoint``.
model = dict(
    type="DERAGroundingDINO",
    training_phase="residual",
    bbox_head=dict(num_classes=12),
    backbone=dict(type="DERASwinTransformer", edge_head_channels=32, init_cfg=None),
)
_data = build_coco_data_config("clcxray", data_root, stage="residual", batch_size=1)
train_dataloader = _data["train_dataloader"]
val_dataloader = _data["val_dataloader"]
test_dataloader = _data["test_dataloader"]
val_evaluator = _data["val_evaluator"]
test_evaluator = _data["test_evaluator"]
del _data
