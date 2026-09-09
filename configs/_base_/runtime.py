"""Small reproducible runtime policy shared by all DERA stages."""

default_scope = "mmdet"
launcher = "none"
resume = False
randomness = dict(seed=2026, deterministic=False)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
log_level = "INFO"
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=20),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1, save_best="auto"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="GroundingVisualizationHook"),
)
