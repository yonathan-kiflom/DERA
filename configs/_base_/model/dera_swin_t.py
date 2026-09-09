"""Canonical DERA model.

The Grounding DINO settings are adapted from MMDetection's
``grounding_dino_swin-t_pretrain_obj365.py`` (Apache-2.0). Keeping this
small model definition local makes external-extension configs independent
of MMDetection's optional ``.mim`` config installation.
"""

custom_imports = dict(imports=["dera.models"], allow_failed_imports=False)

grounding_dino_pretrained = (
    "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/"
    "grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/"
    "grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_"
    "20231204_095047-b448804b.pth"
)
grounding_dino_pretrained_sha256 = (
    "b448804bb1af6fa688887f0f2454625edbeeae4e868bc95620e3e6413581051a"
)
load_from = grounding_dino_pretrained

model = dict(
    type="DERAGroundingDINO",
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    training_phase="foundation",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
    ),
    language_model=dict(
        type="BertModel",
        name="bert-base-uncased",
        max_tokens=256,
        pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=["[CLS]", "[SEP]", ".", "?"],
        add_pooling_layer=False,
    ),
    backbone=dict(
        type="DERAFoundationSwinTransformer",
        embed_dims=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=True,
        frozen_stages=-1,
        pidinet_channels=30,
        pdc_arch="carv4",
        cnn_pyramid_channels=(30, 60, 120, 120),
        # The full Grounding DINO checkpoint below already contains Swin.
        init_cfg=None,
    ),
    neck=dict(
        type="ChannelMapper",
        in_channels=[192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type="GN", num_groups=32),
        num_outs=4,
    ),
    encoder=dict(
        num_layers=6,
        num_cp=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
        ),
        text_layer_cfg=dict(
            self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_drop=0.0),
        ),
        fusion_layer_cfg=dict(
            v_dim=256, l_dim=256, embed_dim=1024, num_heads=4, init_values=1e-4
        ),
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
        ),
        post_norm_cfg=None,
    ),
    positional_encoding=dict(num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type="GroundingDINOHead",
        num_classes=256,
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(max_text_len=256, log_scale="auto", bias=True),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
    ),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100),
    ),
    train_cfg=dict(
        assigner=dict(
            type="HungarianAssigner",
            match_costs=[
                dict(type="BinaryFocalLossCost", weight=2.0),
                dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                dict(type="IoUCost", iou_mode="giou", weight=2.0),
            ],
        )
    ),
    test_cfg=dict(max_per_img=300),
)
