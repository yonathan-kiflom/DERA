"""Tests for the minimal attributed PiDiNet implementation."""

import pytest
import torch
from torch.nn import functional as F

from third_party.pidinet import PDCConv2d, PiDiNetPyramid, make_pdc


def _inputs_and_weights():
    inputs = torch.arange(49, dtype=torch.float32).view(1, 1, 7, 7) / 10
    weights = torch.arange(9, dtype=torch.float32).view(1, 1, 3, 3) / 10
    bias = torch.tensor([0.25])
    return inputs, weights, bias


@pytest.mark.parametrize("op_type", ["cv", "cd", "ad", "rd"])
def test_pdc_operators_match_reference_equations(op_type):
    inputs, weights, bias = _inputs_and_weights()
    actual = make_pdc(op_type)(
        inputs, weights, bias, stride=1, padding=1, dilation=1, groups=1
    )

    if op_type == "cv":
        expected = F.conv2d(inputs, weights, bias, padding=1)
    elif op_type == "cd":
        expected = F.conv2d(inputs, weights, bias, padding=1)
        expected -= F.conv2d(inputs, weights.sum((2, 3), keepdim=True))
    elif op_type == "ad":
        permutation = [3, 0, 1, 6, 4, 2, 7, 8, 5]
        flat = weights.view(1, 1, 9)
        expected = F.conv2d(
            inputs, (flat - flat[:, :, permutation]).view_as(weights), bias, padding=1
        )
    else:
        radial = weights.new_zeros(1, 1, 25)
        radial[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights.view(1, 1, 9)[:, :, 1:]
        radial[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights.view(1, 1, 9)[:, :, 1:]
        expected = F.conv2d(inputs, radial.view(1, 1, 5, 5), bias, padding=2)

    torch.testing.assert_close(actual, expected)


def test_invalid_pdc_configuration_is_rejected():
    with pytest.raises(ValueError, match="Unknown PDC operator"):
        make_pdc("unknown")
    with pytest.raises(ValueError, match="3x3"):
        PDCConv2d("rd", 1, 1, kernel_size=5)
    with pytest.raises(ValueError, match="divisible"):
        PDCConv2d("cv", 3, 4, kernel_size=3, groups=2)


def test_pidinet_returns_the_four_canonical_scales():
    model = PiDiNetPyramid(base_channels=4, pdc_arch="carv4").eval()
    with torch.no_grad():
        features = model(torch.randn(2, 3, 32, 40))

    assert [tuple(feature.shape) for feature in features] == [
        (2, 4, 32, 40),
        (2, 8, 16, 20),
        (2, 16, 8, 10),
        (2, 16, 4, 5),
    ]
