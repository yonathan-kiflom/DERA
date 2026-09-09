"""Regression checks for the public MMEngine configuration surface."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from dera.cli import _run_training

mmengine = pytest.importorskip("mmengine")
from mmengine.config import Config  # noqa: E402

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def _delete_markers(value: Any, path: str = "") -> list[str]:
    markers: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if key == "_delete_":
                markers.append(child_path)
            markers.extend(_delete_markers(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            markers.extend(_delete_markers(child, f"{path}[{index}]"))
    return markers


@pytest.mark.parametrize(
    "config_path",
    sorted(
        path
        for path in CONFIG_ROOT.glob("*/*.py")
        if not path.parent.name.startswith("_")
    ),
    ids=lambda path: str(path.relative_to(CONFIG_ROOT)),
)
def test_public_config_is_fully_resolved_and_runner_consistent(config_path):
    cfg = Config.fromfile(str(config_path))
    assert not _delete_markers(cfg.to_dict())

    for prefix in ("val", "test"):
        values = (
            cfg.get(f"{prefix}_dataloader"),
            cfg.get(f"{prefix}_cfg"),
            cfg.get(f"{prefix}_evaluator"),
        )
        assert all(value is None for value in values) or all(
            value is not None for value in values
        )


def test_pidray_fulltest_contract_preserves_all_categories():
    from dera.data import build_coco_data_config

    cfg = build_coco_data_config(
        "pidray", "/datasets/PIDray", stage="inference", batch_size=1
    )
    test = cfg["test_dataloader"]["dataset"]
    assert test["ann_file"] == "annotations/fulltest.json"
    assert test["data_prefix"]["img"] == "fulltest/"
    assert len(test["metainfo"]["classes"]) == 12
    assert test["metainfo"]["classes"][4:7] == (
        "Scissors",
        "Wrench",
        "Gun",
    )


def test_stage_batch_sizes_and_checkpoint_initialization():
    expected_foundation = {"pidray": 1, "clcxray": 4, "stcray": 4}
    for dataset, foundation_batch in expected_foundation.items():
        foundation = Config.fromfile(str(CONFIG_ROOT / dataset / "foundation.py"))
        assert foundation.train_dataloader.batch_size == foundation_batch
        assert foundation.model.backbone.init_cfg is None

        for stage in ("boundary", "residual", "inference"):
            cfg = Config.fromfile(str(CONFIG_ROOT / dataset / f"{stage}.py"))
            dataloader = (
                cfg.test_dataloader if stage == "inference" else cfg.train_dataloader
            )
            assert dataloader.batch_size == 1
            assert cfg.model.backbone.init_cfg is None


def test_canonical_stage_hyperparameters():
    for dataset in ("pidray", "clcxray", "stcray"):
        foundation = Config.fromfile(str(CONFIG_ROOT / dataset / "foundation.py"))
        assert foundation.train_cfg.max_epochs == 15
        assert foundation.optim_wrapper.optimizer.lr == 1e-4
        expected_milestone = 15 if dataset == "pidray" else 12
        assert foundation.param_scheduler[0].milestones == [expected_milestone]

        boundary = Config.fromfile(str(CONFIG_ROOT / dataset / "boundary.py"))
        assert boundary.train_cfg.max_epochs == 2
        assert boundary.optim_wrapper.optimizer.lr == 1e-4
        assert boundary.model.edge_focal_alpha == 0.75
        assert boundary.model.edge_focal_gamma == 2.0

        residual = Config.fromfile(str(CONFIG_ROOT / dataset / "residual.py"))
        assert residual.train_cfg.max_epochs == 3
        assert residual.optim_wrapper.optimizer.lr == 1e-4


@pytest.mark.parametrize(
    "key",
    (
        "load_from",
        "model.training_phase",
        "model.backbone.type",
        "train_dataloader.dataset.ann_file",
    ),
)
def test_runtime_overrides_cannot_change_stage_identity(tmp_path, key):
    from dera.engine import prepare_config

    with pytest.raises(ValueError, match="protected stage setting"):
        prepare_config(
            "pidray",
            "boundary",
            tmp_path / "data",
            tmp_path / "run",
            init_checkpoint="parent.pth",
            cfg_options={key: "unsafe"},
        )


def test_fresh_training_rejects_a_nonempty_work_directory(tmp_path):
    work_dir = tmp_path / "existing-run"
    work_dir.mkdir()
    (work_dir / "old-checkpoint.pth").write_bytes(b"stale")
    args = Namespace(
        dataset="pidray",
        stage="foundation",
        data_root=str(tmp_path / "data"),
        work_dir=str(work_dir),
        init_checkpoint=None,
        resume=None,
        batch_size=None,
        num_workers=None,
        cfg_option=[],
    )

    with pytest.raises(ValueError, match="fresh run in nonempty"):
        _run_training(args)
