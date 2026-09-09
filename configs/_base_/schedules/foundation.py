"""Foundation adaptation: detector + PiDiNet/Swin fusion, BERT frozen."""

base_lr = 1e-4
max_epochs = 15

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "backbone.pidinet": dict(lr_mult=1.0),
            "backbone.pid_projs": dict(lr_mult=1.0),
            "backbone": dict(lr_mult=0.1),
            "language_model": dict(lr_mult=0.0),
        }
    ),
)
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[12],
        gamma=0.1,
    )
]
