"""Tests for boundary target generation and focal supervision."""

from types import SimpleNamespace

import pytest
import torch

from dera.models.boundary import (
    build_boundary_targets,
    inner_boundary,
    sigmoid_focal_boundary_loss,
)


class _Instances(dict):
    """Tiny dict/attribute hybrid matching MMEngine's InstanceData API."""

    def __getattr__(self, name):
        return self[name]


def _sample(masks, image_shape):
    return SimpleNamespace(gt_instances=_Instances(masks=masks), img_shape=image_shape)


def test_inner_boundary_is_a_one_pixel_inner_contour():
    masks = torch.zeros(1, 8, 8)
    masks[:, 2:6, 2:6] = 1
    actual = inner_boundary(masks, kernel_size=3)

    expected = masks.clone()
    expected[:, 3:5, 3:5] = 0
    torch.testing.assert_close(actual, expected)


def test_empty_instances_produce_an_empty_target():
    masks = torch.empty(0, 5, 7)
    target = inner_boundary(masks)
    assert target.shape == (1, 5, 7)
    assert torch.count_nonzero(target) == 0


def test_targets_keep_contours_and_exclude_padding():
    inputs = torch.zeros(1, 3, 8, 10)
    logits = torch.zeros(1, 1, 4, 5)
    masks = torch.zeros(1, 6, 8)
    masks[:, 1:5, 2:7] = 1

    targets, valid = build_boundary_targets(logits, inputs, [_sample(masks, (6, 8))])

    assert targets.shape == valid.shape == logits.shape
    assert torch.count_nonzero(targets) > 0
    assert torch.equal(valid[0, 0, :3, :4], torch.ones(3, 4))
    assert torch.count_nonzero(valid[0, 0, 3:]) == 0
    assert torch.count_nonzero(valid[0, 0, :, 4:]) == 0


def test_boundary_training_requires_masks():
    sample = SimpleNamespace(gt_instances=_Instances(), img_shape=(8, 8))
    with pytest.raises(RuntimeError, match="requires gt_instances.masks"):
        build_boundary_targets(
            torch.zeros(1, 1, 2, 2), torch.zeros(1, 3, 8, 8), [sample]
        )


def test_focal_loss_ignores_invalid_pixels_and_is_finite_without_edges():
    logits = torch.tensor([[[[0.0, 50.0], [-50.0, 0.0]]]])
    targets = torch.zeros_like(logits)
    valid = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

    loss = sigmoid_focal_boundary_loss(logits, targets, valid)
    reference = sigmoid_focal_boundary_loss(torch.zeros_like(logits), targets, valid)

    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, reference)
