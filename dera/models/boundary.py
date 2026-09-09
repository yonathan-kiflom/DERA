"""Boundary prediction and supervision helpers for DERA."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def make_side_head(in_channels: int, hidden_channels: int = 32) -> nn.Module:
    """Build one of DERA's two lightweight boundary side heads."""
    if hidden_channels < 1 or hidden_channels % 8:
        raise ValueError("hidden_channels must be a positive multiple of 8")
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        ),
        nn.GroupNorm(8, hidden_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_channels, 1, kernel_size=1),
    )


def inner_boundary(masks: Tensor, kernel_size: int = 3) -> Tensor:
    """Merge instance-wise inner contours into one class-agnostic target.

    Args:
        masks: Binary instance masks shaped ``[instances, height, width]``.
        kernel_size: Positive odd erosion kernel size.

    Returns:
        A tensor shaped ``[1, height, width]``.
    """
    if masks.ndim != 3:
        raise ValueError("masks must have shape [instances, height, width]")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number")
    if masks.shape[0] == 0:
        return masks.new_zeros((1, *masks.shape[-2:]))

    masks = masks.unsqueeze(1)
    padding = kernel_size // 2
    eroded = 1 - F.max_pool2d(
        1 - masks,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )
    boundaries = (masks - eroded).clamp(min=0, max=1)
    return boundaries.amax(dim=0)


def _as_mask_tensor(masks: Any, device: torch.device) -> Tensor:
    if hasattr(masks, "to_tensor"):
        return masks.to_tensor(dtype=torch.float32, device=device)
    if torch.is_tensor(masks):
        return masks.to(device=device, dtype=torch.float32)
    raise TypeError(
        "gt_instances.masks must be an MMDetection mask structure or Tensor; "
        f"got {type(masks)}"
    )


def build_boundary_targets(
    edge_logits: Tensor,
    batch_inputs: Tensor,
    batch_data_samples: Sequence[Any],
    kernel_size: int = 3,
) -> tuple[Tensor, Tensor]:
    """Build boundary targets and ignore padded image regions.

    Masks are needed only by the boundary-training stage. Targets are reduced
    with max pooling so thin contours survive at the edge-head resolution.
    """
    if edge_logits.ndim != 4 or edge_logits.shape[1] != 1:
        raise ValueError("edge_logits must have shape [batch, 1, height, width]")
    if len(batch_data_samples) != edge_logits.shape[0]:
        raise ValueError("one data sample is required for each batch item")

    batch_height, batch_width = batch_inputs.shape[-2:]
    edge_size = edge_logits.shape[-2:]
    targets: list[Tensor] = []
    valid_regions: list[Tensor] = []

    for data_sample in batch_data_samples:
        gt_instances = data_sample.gt_instances
        if "masks" not in gt_instances:
            raise RuntimeError(
                "Boundary training requires gt_instances.masks. Configure "
                "LoadAnnotations(with_mask=True)."
            )
        masks = _as_mask_tensor(gt_instances.masks, edge_logits.device)

        canvas = masks.new_zeros((masks.shape[0], batch_height, batch_width))
        copy_height = min(masks.shape[-2], batch_height)
        copy_width = min(masks.shape[-1], batch_width)
        canvas[:, :copy_height, :copy_width] = masks[:, :copy_height, :copy_width]
        target = inner_boundary(canvas, kernel_size)
        if target.shape[-2:] != edge_size:
            target = F.adaptive_max_pool2d(target.unsqueeze(0), edge_size)[0]
        targets.append(target)

        image_height, image_width = data_sample.img_shape[:2]
        valid = edge_logits.new_zeros((1, 1, batch_height, batch_width))
        valid[
            ..., : min(image_height, batch_height), : min(image_width, batch_width)
        ] = 1
        valid = F.interpolate(valid, size=edge_size, mode="nearest")
        valid_regions.append(valid[0])

    return torch.stack(targets), torch.stack(valid_regions)


def sigmoid_focal_boundary_loss(
    logits: Tensor,
    targets: Tensor,
    valid_regions: Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> Tensor:
    """Compute mask-aware sigmoid focal loss normalized by edge pixels."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if logits.shape != targets.shape or logits.shape != valid_regions.shape:
        raise ValueError("logits, targets, and valid_regions must match")

    logits = logits.float()
    targets = targets.float()
    valid_regions = valid_regions.float()
    probabilities = logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = cross_entropy * alpha_t * (1 - p_t).pow(gamma) * valid_regions
    positive_count = (targets * valid_regions).sum()
    normalizer = torch.where(positive_count > 0, positive_count, valid_regions.sum())
    return loss.sum() / normalizer.clamp_min(1)


__all__ = [
    "build_boundary_targets",
    "inner_boundary",
    "make_side_head",
    "sigmoid_focal_boundary_loss",
]
