"""Stage 1: adapt the DERA foundation on PIDray."""

_base_ = [
    "../_base_/model/dera_swin_t.py",
    "../_base_/runtime.py",
    "../_base_/schedules/foundation.py",
]

from dera.data import build_coco_data_config  # noqa: E402

data_root = "data/pidray/"
model = dict(bbox_head=dict(num_classes=12))
# The checkpoint-producing PIDray run completed all 15 epochs before decay.
param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=15,
        by_epoch=True,
        milestones=[15],
        gamma=0.1,
    )
]
_data = build_coco_data_config("pidray", data_root, stage="foundation", batch_size=1)
train_dataloader = _data["train_dataloader"]
val_dataloader = _data["val_dataloader"]
test_dataloader = _data["test_dataloader"]
val_evaluator = _data["val_evaluator"]
test_evaluator = _data["test_evaluator"]
del _data
