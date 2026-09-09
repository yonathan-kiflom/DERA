"""Boundary learning: only two side heads and their fusion are trainable."""

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
)
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=2, val_interval=100)
param_scheduler = []
