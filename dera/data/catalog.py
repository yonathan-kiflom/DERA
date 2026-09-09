"""Load the small, human-readable dataset specifications."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

_DATASETS = ("pidray", "clcxray", "stcray")


def available_datasets() -> tuple[str, ...]:
    """Return datasets for which DERA ships a reproducible recipe."""
    return _DATASETS


def _spec_roots() -> list[Path]:
    roots: list[Path] = []
    if override := os.environ.get("DERA_DATASET_SPECS"):
        roots.append(Path(override).expanduser())
    roots.append(Path(__file__).resolve().parents[2] / "dataset_specs")
    roots.append(Path(sys.prefix) / "share" / "dera" / "dataset_specs")
    return roots


def load_dataset_spec(name: str) -> dict[str, Any]:
    """Load and minimally validate one dataset specification.

    Set ``DERA_DATASET_SPECS`` to use a custom specification directory.
    Source checkouts and installed wheels are discovered automatically.
    """
    normalized = name.lower()
    if normalized not in _DATASETS:
        choices = ", ".join(_DATASETS)
        raise ValueError(f"Unknown dataset {name!r}; choose one of: {choices}")

    searched = []
    for root in _spec_roots():
        path = root / f"{normalized}.yaml"
        searched.append(str(path))
        if path.is_file():
            with path.open(encoding="utf-8") as stream:
                spec = yaml.safe_load(stream)
            if not isinstance(spec, dict):
                raise ValueError(f"{path} must contain a YAML mapping")
            required = {"name", "classes", "splits"}
            missing = required.difference(spec)
            if missing:
                raise ValueError(
                    f"{path} is missing keys: {', '.join(sorted(missing))}"
                )
            if not isinstance(spec["classes"], list) or not spec["classes"]:
                raise ValueError(f"{path}: classes must be a non-empty list")
            for split in ("train", "test"):
                if split not in spec["splits"]:
                    raise ValueError(f"{path}: missing {split!r} split")
                split_spec = spec["splits"][split]
                if not {"annotation", "images"} <= split_spec.keys():
                    raise ValueError(
                        f"{path}: split {split!r} needs annotation and images"
                    )
            return spec

    locations = "\n  - ".join(searched)
    raise FileNotFoundError(
        f"Could not find {normalized}.yaml. Searched:\n  - {locations}"
    )
