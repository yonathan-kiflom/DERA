# Adapted from PiDiNet. Copyright (c) 2021 Zhuo Su.
# Governed by the adjacent LICENSE; modified for DERA.
"""Pixel-difference convolutions adapted from PiDiNet.

The four operators keep PiDiNet's original weight layouts so that DERA's
legacy checkpoints load without conversion.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PDCFunction = Callable[..., Tensor]


def _central_difference(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> Tensor:
    center_weight = weight.sum(dim=(2, 3), keepdim=True)
    center = F.conv2d(x, center_weight, stride=stride, groups=groups)
    ordinary = F.conv2d(
        x,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    return ordinary - center


def _angular_difference(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> Tensor:
    shape = weight.shape
    flat_weight = weight.view(shape[0], shape[1], -1)
    clockwise = flat_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
    difference_weight = (flat_weight - clockwise).view(shape)
    return F.conv2d(
        x,
        difference_weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def _radial_difference(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> Tensor:
    del padding  # PiDiNet's radial operator always uses a dilated 5x5 grid.
    shape = weight.shape
    radial_weight = weight.new_zeros(shape[0], shape[1], 25)
    flat_weight = weight.view(shape[0], shape[1], -1)
    outer = [0, 2, 4, 10, 14, 20, 22, 24]
    inner = [6, 7, 8, 11, 13, 16, 17, 18]
    radial_weight[:, :, outer] = flat_weight[:, :, 1:]
    radial_weight[:, :, inner] = -flat_weight[:, :, 1:]
    return F.conv2d(
        x,
        radial_weight.view(shape[0], shape[1], 5, 5),
        bias,
        stride=stride,
        padding=2 * dilation,
        dilation=dilation,
        groups=groups,
    )


def make_pdc(op_type: str) -> PDCFunction:
    """Return one of PiDiNet's convolution operators.

    Args:
        op_type: ``cv`` (ordinary), ``cd`` (central difference), ``ad``
            (angular difference), or ``rd`` (radial difference).
    """
    operators: dict[str, PDCFunction] = {
        "cv": F.conv2d,
        "cd": _central_difference,
        "ad": _angular_difference,
        "rd": _radial_difference,
    }
    try:
        return operators[op_type]
    except KeyError as error:
        choices = ", ".join(operators)
        raise ValueError(
            f"Unknown PDC operator {op_type!r}; choose one of: {choices}."
        ) from error


class PDCConv2d(nn.Module):
    """A learnable 2-D convolution evaluated as a PDC operator."""

    def __init__(
        self,
        pdc_type: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if in_channels % groups:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups:
            raise ValueError("out_channels must be divisible by groups")
        if pdc_type in {"cd", "ad", "rd"} and kernel_size != 3:
            raise ValueError(f"{pdc_type} requires a 3x3 kernel")

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.pdc = make_pdc(pdc_type)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use the same initialization as ``torch.nn.Conv2d``."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1] * self.weight.shape[2] ** 2
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return self.pdc(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


__all__ = ["PDCConv2d", "PDCFunction", "make_pdc"]
