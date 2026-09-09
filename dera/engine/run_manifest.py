"""Checkpoint selection and compact reproducibility manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_checkpoint(work_dir: str | Path, *, use_marker: bool = True) -> Path:
    """Prefer the validation-selected checkpoint, then MMEngine's last one."""
    root = Path(work_dir)
    selected_marker = root / "selected_checkpoint"
    if use_marker and selected_marker.is_file():
        selected = Path(selected_marker.read_text(encoding="utf-8").strip())
        if not selected.is_absolute():
            selected = root / selected
        if selected.is_file():
            return selected.resolve()

    best = sorted(root.glob("best_*.pth"), key=lambda path: path.stat().st_mtime)
    if best:
        return best[-1].resolve()

    marker = root / "last_checkpoint"
    if marker.is_file():
        candidate = Path(marker.read_text(encoding="utf-8").strip())
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file():
            return candidate.resolve()

    checkpoints = sorted(root.glob("*.pth"), key=lambda path: path.stat().st_mtime)
    if checkpoints:
        return checkpoints[-1].resolve()
    raise FileNotFoundError(f"No checkpoint found in {root}")


def write_manifest(work_dir: str | Path, values: dict[str, Any]) -> Path:
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "run_manifest.json"
    path.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def write_selected_checkpoint(work_dir: str | Path, checkpoint: str | Path) -> Path:
    """Write the stable handoff marker consumed by the next training stage."""
    marker = Path(work_dir) / "selected_checkpoint"
    marker.write_text(str(Path(checkpoint).resolve()) + "\n", encoding="utf-8")
    return marker
