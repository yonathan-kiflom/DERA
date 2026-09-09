"""Stage 1: adapt the DERA foundation on STCray."""

_base_ = [
    "../_base_/model/dera_swin_t.py",
    "../_base_/runtime.py",
    "../_base_/schedules/foundation.py",
]

from dera.data import build_coco_data_config  # noqa: E402

data_root = "data/stcray/"
model = dict(bbox_head=dict(num_classes=20))
_data = build_coco_data_config("stcray", data_root, stage="foundation", batch_size=4)
train_dataloader = _data["train_dataloader"]
val_dataloader = _data["val_dataloader"]
test_dataloader = _data["test_dataloader"]
val_evaluator = _data["val_evaluator"]
test_evaluator = _data["test_evaluator"]
del _data
