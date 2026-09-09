"""Synthetic checks for the public dataset workflow."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from dera.data import build_coco_data_config, validate_dataset


def _write_fixture(root: Path, *, bbox: list[float]) -> None:
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "test").mkdir(parents=True)
    (root / "annotations").mkdir()
    (root / "images" / "train" / "sample.jpg").write_bytes(b"fixture")
    coco = {
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 10, "height": 10}],
        "categories": [{"id": 1, "name": "item"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "segmentation": [[1, 1, 8, 1, 8, 8, 1, 8]],
            }
        ],
    }
    (root / "annotations" / "train.json").write_text(json.dumps(coco))


def _write_spec(root: Path) -> Path:
    specs = root / "specs"
    specs.mkdir()
    spec = {
        "name": "fixture",
        "classes": ["item"],
        "splits": {
            "train": {
                "annotation": "annotations/train.json",
                "mask_annotation": "annotations/train.json",
                "images": "images/train/",
            },
            "test": {
                "annotation": "annotations/test.json",
                "images": "images/test/",
            },
        },
    }
    # Reuse a supported catalog name while overriding only the spec location.
    (specs / "pidray.yaml").write_text(yaml.safe_dump(spec))
    return specs


def test_boundary_config_requests_masks(monkeypatch, tmp_path):
    monkeypatch.setenv("DERA_DATASET_SPECS", str(_write_spec(tmp_path)))
    config = build_coco_data_config(
        "pidray", str(tmp_path), stage="boundary", batch_size=1
    )
    annotation = config["train_dataloader"]["dataset"]["pipeline"][1]
    assert annotation["with_mask"] is True
    assert config["val_dataloader"] is None


def test_validator_accepts_valid_mask_data(monkeypatch, tmp_path):
    monkeypatch.setenv("DERA_DATASET_SPECS", str(_write_spec(tmp_path)))
    _write_fixture(tmp_path, bbox=[1, 1, 7, 7])
    report = validate_dataset("pidray", tmp_path, require_masks=True)
    assert report.ok, report.errors
    assert (report.images, report.annotations, report.masks) == (1, 1, 1)


def test_validator_rejects_fully_outside_box(monkeypatch, tmp_path):
    monkeypatch.setenv("DERA_DATASET_SPECS", str(_write_spec(tmp_path)))
    _write_fixture(tmp_path, bbox=[11, 1, 2, 2])
    report = validate_dataset("pidray", tmp_path, require_masks=True)
    assert not report.ok
    assert any("fully outside" in error for error in report.errors)
