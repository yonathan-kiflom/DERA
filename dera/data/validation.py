"""Strict, dependency-light validation of DERA COCO annotations."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .catalog import load_dataset_spec


@dataclass
class ValidationReport:
    dataset: str
    split: str
    images: int = 0
    annotations: int = 0
    masks: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    annotation_sha256: str = ""

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        status = "OK" if self.ok else "FAILED"
        return (
            f"{status}: {self.dataset}/{self.split}: {self.images} images, "
            f"{self.annotations} annotations, {self.masks} masks, "
            f"{len(self.errors)} errors, {len(self.warnings)} warnings"
        )


def _duplicates(values: Iterable[Any]) -> set[Any]:
    seen: set[Any] = set()
    duplicates: set[Any] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _valid_segmentation(segmentation: Any) -> bool:
    if isinstance(segmentation, dict):
        size = segmentation.get("size")
        counts = segmentation.get("counts")
        return (
            isinstance(size, list)
            and len(size) == 2
            and all(isinstance(value, int) and value > 0 for value in size)
            and isinstance(counts, (str, list))
            and bool(counts)
        )
    if isinstance(segmentation, list) and segmentation:
        return all(
            isinstance(polygon, list)
            and len(polygon) >= 6
            and len(polygon) % 2 == 0
            and all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in polygon
            )
            for polygon in segmentation
        )
    return False


def _check_mask_decode(segmentation: Any, height: int, width: int) -> str | None:
    """Decode masks when pycocotools is present; schema checks always run."""
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        return None
    try:
        rle = segmentation
        if isinstance(segmentation, list):
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
        decoded = mask_utils.decode(rle)
        if decoded.shape[:2] != (height, width):
            return f"decoded shape {decoded.shape[:2]} != {(height, width)}"
        if not decoded.any():
            return "decoded mask is empty"
    except Exception as error:  # pycocotools raises several exception types
        return f"cannot decode mask ({error})"
    return None


def validate_dataset(
    dataset: str,
    data_root: str | Path,
    *,
    split: str = "train",
    require_masks: bool = False,
    check_images: bool = True,
    max_errors: int = 50,
) -> ValidationReport:
    """Validate IDs, categories, boxes, image files, and optional masks."""
    spec = load_dataset_spec(dataset)
    if split not in spec["splits"]:
        raise ValueError(f"{dataset!r} does not define split {split!r}")
    split_spec = spec["splits"][split]
    annotation_key = (
        "mask_annotation"
        if require_masks and split_spec.get("mask_annotation")
        else "annotation"
    )
    root = Path(data_root).expanduser().resolve()
    annotation_path = root / split_spec[annotation_key]
    report = ValidationReport(dataset=dataset, split=split)

    if not annotation_path.is_file():
        report.errors.append(f"missing annotation: {annotation_path}")
        return report
    raw = annotation_path.read_bytes()
    report.annotation_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        coco = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        report.errors.append(f"invalid JSON: {error}")
        return report
    if not isinstance(coco, dict):
        report.errors.append("COCO root must be an object")
        return report

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    if not all(isinstance(items, list) for items in (images, annotations, categories)):
        report.errors.append("images, annotations, and categories must be lists")
        return report
    report.images = len(images)
    report.annotations = len(annotations)

    image_ids = [item.get("id") for item in images if isinstance(item, dict)]
    annotation_ids = [item.get("id") for item in annotations if isinstance(item, dict)]
    category_ids = [item.get("id") for item in categories if isinstance(item, dict)]
    for label, values in (
        ("image", image_ids),
        ("annotation", annotation_ids),
        ("category", category_ids),
    ):
        duplicates = _duplicates(values)
        if duplicates:
            report.errors.append(
                f"duplicate {label} IDs: {sorted(duplicates, key=str)[:5]}"
            )

    ordered_categories = sorted(
        (item for item in categories if isinstance(item, dict)),
        key=lambda item: item.get("id", -1),
    )
    actual_classes = [item.get("name") for item in ordered_categories]
    expected_classes = split_spec.get("classes", spec["classes"])
    if actual_classes != expected_classes:
        report.errors.append(
            "category names/order do not match the dataset specification: "
            f"expected {expected_classes!r}, got {actual_classes!r}"
        )

    images_by_id: dict[Any, dict[str, Any]] = {}
    image_dir = root / split_spec["images"]
    for index, image in enumerate(images):
        if not isinstance(image, dict):
            report.errors.append(f"images[{index}] is not an object")
            continue
        image_id = image.get("id")
        width, height = image.get("width"), image.get("height")
        file_name = image.get("file_name")
        if (
            not isinstance(width, int)
            or width <= 0
            or not isinstance(height, int)
            or height <= 0
        ):
            report.errors.append(f"image {image_id!r} has invalid dimensions")
            continue
        if not isinstance(file_name, str) or not file_name:
            report.errors.append(f"image {image_id!r} has no file_name")
        elif check_images and not (image_dir / file_name).is_file():
            report.errors.append(f"missing image: {image_dir / file_name}")
        images_by_id[image_id] = image
        if len(report.errors) >= max_errors:
            break

    valid_category_ids = set(category_ids)
    for index, annotation in enumerate(annotations):
        if len(report.errors) >= max_errors:
            break
        if not isinstance(annotation, dict):
            report.errors.append(f"annotations[{index}] is not an object")
            continue
        annotation_id = annotation.get("id", index)
        image = images_by_id.get(annotation.get("image_id"))
        if image is None:
            report.errors.append(
                f"annotation {annotation_id!r} references an unknown image"
            )
            continue
        if annotation.get("category_id") not in valid_category_ids:
            report.errors.append(
                f"annotation {annotation_id!r} references an unknown category"
            )

        bbox = annotation.get("bbox")
        if not (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in bbox
            )
        ):
            report.errors.append(f"annotation {annotation_id!r} has invalid bbox")
        else:
            x, y, width, height = bbox
            image_width, image_height = image["width"], image["height"]
            if width <= 0 or height <= 0:
                report.errors.append(
                    f"annotation {annotation_id!r} has a non-positive bbox"
                )
            fully_outside = (
                x >= image_width
                or y >= image_height
                or x + width <= 0
                or y + height <= 0
            )
            partly_outside = (
                x < -1e-3
                or y < -1e-3
                or x + width > image_width + 1e-3
                or y + height > image_height + 1e-3
            )
            if fully_outside:
                report.errors.append(
                    f"annotation {annotation_id!r} bbox is fully outside its image"
                )
            elif partly_outside:
                report.warnings.append(
                    f"annotation {annotation_id!r} bbox crosses its image "
                    "boundary and will be clipped by MMDetection"
                )

        segmentation = annotation.get("segmentation")
        if segmentation is not None and _valid_segmentation(segmentation):
            report.masks += 1
        elif require_masks:
            report.errors.append(
                f"annotation {annotation_id!r} has no valid segmentation"
            )
            continue
        if require_masks:
            decode_error = _check_mask_decode(
                segmentation, image["height"], image["width"]
            )
            if decode_error:
                report.errors.append(f"annotation {annotation_id!r}: {decode_error}")

    if len(report.errors) >= max_errors:
        report.warnings.append(f"stopped after reaching the {max_errors}-error limit")
    expected_hash = split_spec.get(f"{annotation_key}_sha256")
    if expected_hash and expected_hash != report.annotation_sha256:
        report.errors.append(
            f"annotation SHA-256 mismatch: expected {expected_hash}, "
            f"got {report.annotation_sha256}"
        )
    if not images:
        report.errors.append("annotation contains no images")
    return report
