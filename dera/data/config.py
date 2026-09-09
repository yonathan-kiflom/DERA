"""Build MMDetection COCO dataloaders from a DERA dataset spec."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .catalog import load_dataset_spec

_SCALES = [(height, 1333) for height in range(480, 801, 32)]


def _train_pipeline(with_mask: bool) -> list[dict[str, Any]]:
    return [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations", with_bbox=True, with_mask=with_mask),
        dict(type="RandomFlip", prob=0.5),
        dict(
            type="RandomChoice",
            transforms=[
                [dict(type="RandomChoiceResize", scales=_SCALES, keep_ratio=True)],
                [
                    dict(
                        type="RandomChoiceResize",
                        scales=[(400, 4200), (500, 4200), (600, 4200)],
                        keep_ratio=True,
                    ),
                    dict(
                        type="RandomCrop",
                        crop_type="absolute_range",
                        crop_size=(384, 600),
                        allow_negative_crop=True,
                    ),
                    dict(type="RandomChoiceResize", scales=_SCALES, keep_ratio=True),
                ],
            ],
        ),
        dict(
            type="PackDetInputs",
            meta_keys=(
                "img_id",
                "img_path",
                "ori_shape",
                "img_shape",
                "scale_factor",
                "flip",
                "flip_direction",
                "text",
                "custom_entities",
            ),
        ),
    ]


def _test_pipeline() -> list[dict[str, Any]]:
    return [
        dict(type="LoadImageFromFile", imdecode_backend="pillow"),
        dict(
            type="FixScaleResize",
            scale=(800, 1333),
            keep_ratio=True,
            backend="pillow",
        ),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(
            type="PackDetInputs",
            meta_keys=(
                "img_id",
                "img_path",
                "ori_shape",
                "img_shape",
                "scale_factor",
                "text",
                "custom_entities",
            ),
        ),
    ]


def _join(root: str, relative: str) -> str:
    """Join config paths without resolving them against the current host."""
    return str(Path(root) / relative)


def build_coco_data_config(
    dataset: str,
    data_root: str,
    *,
    stage: str,
    batch_size: int = 4,
    num_workers: int = 4,
) -> dict[str, Any]:
    """Return dataloaders and evaluators for one DERA stage.

    Boundary learning reads the mask-supervised training annotation. All
    other stages read the box annotation. Test annotations never need masks.
    """
    if stage not in {"foundation", "boundary", "residual", "inference"}:
        raise ValueError(f"Unsupported DERA stage: {stage!r}")

    spec = load_dataset_spec(dataset)
    classes = tuple(spec["classes"])
    metainfo = dict(classes=classes)
    train_spec = spec["splits"]["train"]
    val_spec = spec["splits"].get("val", spec["splits"]["test"])
    test_spec = spec["splits"]["test"]
    train_annotation = (
        train_spec.get("mask_annotation", train_spec["annotation"])
        if stage == "boundary"
        else train_spec["annotation"]
    )

    train_dataset = dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file=train_annotation,
        data_prefix=dict(img=train_spec["images"]),
        metainfo=metainfo,
        return_classes=True,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=_train_pipeline(with_mask=stage == "boundary"),
    )

    def evaluation_dataset(split_spec: dict[str, Any]) -> dict[str, Any]:
        split_classes = tuple(split_spec.get("classes", classes))
        return dict(
            type="CocoDataset",
            data_root=data_root,
            ann_file=split_spec["annotation"],
            data_prefix=dict(img=split_spec["images"]),
            metainfo=dict(classes=split_classes),
            return_classes=True,
            test_mode=True,
            pipeline=_test_pipeline(),
        )

    val_dataset = evaluation_dataset(val_spec)
    test_dataset = evaluation_dataset(test_spec)

    config: dict[str, Any] = dict(
        train_dataloader=dict(
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            sampler=dict(type="DefaultSampler", shuffle=True),
            batch_sampler=dict(type="AspectRatioBatchSampler"),
            dataset=train_dataset,
        ),
        val_dataloader=dict(
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            drop_last=False,
            sampler=dict(type="DefaultSampler", shuffle=False),
            dataset=val_dataset,
        ),
        test_dataloader=dict(
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            drop_last=False,
            sampler=dict(type="DefaultSampler", shuffle=False),
            dataset=test_dataset,
        ),
        val_evaluator=dict(
            type="CocoMetric",
            ann_file=_join(data_root, val_spec["annotation"]),
            metric="bbox",
            format_only=False,
        ),
        test_evaluator=dict(
            type="CocoMetric",
            ann_file=_join(data_root, test_spec["annotation"]),
            metric="bbox",
            format_only=False,
        ),
    )

    if stage == "boundary":
        # The boundary-only phase has no meaningful box validation.
        config.update(
            val_cfg=None,
            test_cfg=None,
            val_dataloader=None,
            val_evaluator=None,
            test_dataloader=None,
            test_evaluator=None,
        )
    elif stage == "inference":
        config.update(train_cfg=None, train_dataloader=None)
    return config
