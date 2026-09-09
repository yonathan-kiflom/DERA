"""Opt-in provenance test for the canonical three-stage PIDray run.

Public CI skips this test because checkpoints are intentionally not committed.
Maintainers can set the three ``DERA_PIDRAY_*_CHECKPOINT`` environment
variables to trusted artifacts.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

_CHECKPOINTS = {
    "foundation": "DERA_PIDRAY_FOUNDATION_CHECKPOINT",
    "boundary": "DERA_PIDRAY_BOUNDARY_CHECKPOINT",
    "final": "DERA_PIDRAY_FINAL_CHECKPOINT",
}

_SIDE_HEAD_KEYS = {
    f"backbone.edge_side_heads.{side}.{suffix}"
    for side in (0, 1)
    for suffix in (
        "0.weight",
        "1.weight",
        "1.bias",
        "3.weight",
        "3.bias",
    )
}
_FUSION_KEYS = {
    "backbone.edge_fuse.weight",
    "backbone.edge_fuse.bias",
}
_RESIDUAL_KEYS = {
    f"backbone.edge_residual_projs.{stage}.{parameter}"
    for stage in (0, 1)
    for parameter in ("weight", "bias")
}
_ADDED_EDGE_KEYS = _SIDE_HEAD_KEYS | _FUSION_KEYS | _RESIDUAL_KEYS


def _resolve_checkpoints() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    unavailable: list[str] = []
    for stage, environment_name in _CHECKPOINTS.items():
        override = os.environ.get(environment_name)
        if not override:
            unavailable.append(stage)
            continue
        path = Path(override).expanduser()
        if not path.is_file():
            pytest.fail(f"{environment_name} does not name a file: {path}")
        paths[stage] = path

    if unavailable:
        variables = ", ".join(_CHECKPOINTS.values())
        pytest.skip(
            "private PIDray checkpoints are unavailable; set "
            f"{variables} to run this provenance test"
        )
    return paths


def _load_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    # These artifacts are maintainer-controlled and trusted. mmap keeps the
    # three large checkpoints out of resident memory during tensor comparison.
    load_kwargs = dict(map_location="cpu", weights_only=False)
    try:
        checkpoint = torch.load(path, mmap=True, **load_kwargs)
    except TypeError:  # PyTorch versions predating torch.load(..., mmap=...).
        checkpoint = torch.load(path, **load_kwargs)

    if not isinstance(checkpoint, Mapping) or "state_dict" not in checkpoint:
        pytest.fail(f"checkpoint has no state_dict mapping: {path}")
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, Mapping):
        pytest.fail(f"checkpoint state_dict is not a mapping: {path}")
    return state_dict


def _changed_keys(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> set[str]:
    return {key for key in left if not torch.equal(left[key], right[key])}


def test_canonical_pidray_three_stage_checkpoint_lineage():
    paths = _resolve_checkpoints()
    foundation = _load_state_dict(paths["foundation"])
    boundary = _load_state_dict(paths["boundary"])
    final = _load_state_dict(paths["final"])

    assert len(foundation) == 953
    assert len(boundary) == 969
    assert len(final) == 969

    foundation_keys = set(foundation)
    boundary_keys = set(boundary)
    final_keys = set(final)
    assert foundation_keys <= boundary_keys
    assert boundary_keys - foundation_keys == _ADDED_EDGE_KEYS
    assert boundary_keys == final_keys
    assert len(_ADDED_EDGE_KEYS) == 16

    foundation_changes = _changed_keys(foundation, boundary)
    assert foundation_changes == set()

    assert all(torch.count_nonzero(boundary[key]).item() == 0 for key in _RESIDUAL_KEYS)

    residual_changes = _changed_keys(boundary, final)
    assert residual_changes == _RESIDUAL_KEYS
    assert len(boundary_keys - residual_changes) == 965
