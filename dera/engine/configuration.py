"""Resolve and safely customize public DERA configurations."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

from dera.data import available_datasets, load_dataset_spec

_STAGES = ("foundation", "boundary", "residual", "inference")

# These values define a stage's identity, checkpoint lineage, and validated
# dataset. They have dedicated CLI arguments or dataset specifications and may
# not be replaced through the generic MMEngine override escape hatch.
_PROTECTED_CFG_PATHS = (
    "custom_imports",
    "default_scope",
    "work_dir",
    "load_from",
    "resume",
    "model.type",
    "model.training_phase",
    "model.backbone.type",
    "train_cfg.type",
    "val_cfg",
    "test_cfg",
    "train_dataloader.dataset",
    "val_dataloader.dataset",
    "test_dataloader.dataset",
    "val_evaluator.ann_file",
    "test_evaluator.ann_file",
)


def _config_roots() -> list[Path]:
    roots: list[Path] = []
    if override := os.environ.get("DERA_CONFIGS"):
        roots.append(Path(override).expanduser())
    roots.append(Path(__file__).resolve().parents[2] / "configs")
    roots.append(Path(sys.prefix) / "share" / "dera" / "configs")
    return roots


def config_path(dataset: str, stage: str) -> Path:
    dataset = dataset.lower()
    if dataset not in available_datasets():
        raise ValueError(f"Unsupported dataset: {dataset!r}")
    if stage not in _STAGES:
        raise ValueError(f"Unsupported stage: {stage!r}")
    searched = []
    for root in _config_roots():
        path = root / dataset / f"{stage}.py"
        searched.append(str(path))
        if path.is_file():
            return path
    raise FileNotFoundError("Config not found; searched: " + ", ".join(searched))


def _patch_dataset(dataset_cfg: Any, data_root: str) -> None:
    if not isinstance(dataset_cfg, Mapping):
        return
    if dataset_cfg.get("type") in {"ConcatDataset", "RepeatDataset"}:
        for child in dataset_cfg.get("datasets", []):
            _patch_dataset(child, data_root)
        _patch_dataset(dataset_cfg.get("dataset"), data_root)
        return
    if "data_root" in dataset_cfg:
        dataset_cfg["data_root"] = data_root


def _patch_dataloaders(cfg: Any, data_root: str) -> None:
    for key in ("train_dataloader", "val_dataloader", "test_dataloader"):
        dataloader = cfg.get(key)
        if isinstance(dataloader, Mapping):
            _patch_dataset(dataloader.get("dataset"), data_root)


def _patch_evaluators(cfg: Any, dataset: str, data_root: str) -> None:
    splits = load_dataset_spec(dataset)["splits"]
    pairs = (
        ("val_evaluator", splits.get("val", splits["test"])),
        ("test_evaluator", splits["test"]),
    )
    for key, split in pairs:
        evaluator = cfg.get(key)
        if isinstance(evaluator, Mapping) and "ann_file" in evaluator:
            evaluator["ann_file"] = str(Path(data_root) / split["annotation"])


def _set_batch_size(cfg: Any, batch_size: int | None) -> None:
    if batch_size is None:
        return
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for key in ("train_dataloader", "val_dataloader", "test_dataloader"):
        dataloader = cfg.get(key)
        if isinstance(dataloader, Mapping):
            dataloader["batch_size"] = batch_size


def _set_workers(cfg: Any, num_workers: int | None) -> None:
    if num_workers is None:
        return
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    for key in ("train_dataloader", "val_dataloader", "test_dataloader"):
        dataloader = cfg.get(key)
        if isinstance(dataloader, Mapping):
            dataloader["num_workers"] = num_workers
            dataloader["persistent_workers"] = num_workers > 0


def _reject_protected_options(options: Mapping[str, Any]) -> None:
    for key in options:
        if any(
            key == protected
            or key.startswith(f"{protected}.")
            or protected.startswith(f"{key}.")
            for protected in _PROTECTED_CFG_PATHS
        ):
            raise ValueError(
                f"--cfg-option {key!r} changes a protected stage setting. "
                "Use the dedicated CLI argument or edit a copied config/spec."
            )


def prepare_config(
    dataset: str,
    stage: str,
    data_root: str | Path,
    work_dir: str | Path,
    *,
    init_checkpoint: str | None = None,
    resume: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    cfg_options: Mapping[str, Any] | None = None,
) -> Any:
    """Load a DERA config and apply explicit, auditable runtime values."""
    if init_checkpoint and resume:
        raise ValueError("--init-checkpoint and --resume are mutually exclusive")

    try:
        from mmengine.config import Config
    except ImportError as error:
        raise RuntimeError(
            "MMEngine is not installed. Follow the DERA installation guide."
        ) from error

    cfg = Config.fromfile(str(config_path(dataset, stage)))
    root = str(Path(data_root).expanduser().resolve())
    cfg.work_dir = str(Path(work_dir).expanduser().resolve())
    _patch_dataloaders(cfg, root)
    _patch_evaluators(cfg, dataset, root)
    _set_batch_size(cfg, batch_size)
    _set_workers(cfg, num_workers)

    if init_checkpoint:
        cfg.load_from = init_checkpoint
        cfg.resume = False
        if stage == "foundation":
            # A full Grounding DINO initialization already contains Swin.
            # Avoid a redundant network fetch when a local cache is supplied.
            cfg.model.backbone.init_cfg = None
    elif resume:
        cfg.load_from = resume
        cfg.resume = True
    else:
        cfg.resume = False
    if cfg_options:
        _reject_protected_options(cfg_options)
        cfg.merge_from_dict(dict(cfg_options))
    return cfg
