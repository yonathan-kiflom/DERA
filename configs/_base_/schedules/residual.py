"""Residual adaptation: only two zero-initialized projections train."""

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
)
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=3, val_interval=1)
param_scheduler = []
