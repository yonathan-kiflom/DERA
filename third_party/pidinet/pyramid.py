# Adapted from PiDiNet. Copyright (c) 2021 Zhuo Su.
# Governed by the adjacent LICENSE; modified for DERA.
"""Minimal four-level PiDiNet encoder used by DERA."""

from __future__ import annotations

from torch import Tensor, nn

from .pdc import PDCConv2d

PDC_ARCHITECTURES: dict[str, tuple[str, ...]] = {
    "baseline": ("cv",) * 16,
    "carv4": (
        "cd",
        "ad",
        "rd",
        "cv",
        "cd",
        "ad",
        "rd",
        "cv",
        "cd",
        "ad",
        "rd",
        "cv",
        "cd",
        "ad",
        "rd",
        "cv",
    ),
}


class PDCBlock(nn.Module):
    """PiDiNet depthwise PDC block with a residual shortcut."""

    def __init__(
        self,
        pdc_type: str,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("PDCBlock stride must be 1 or 2")
        self.stride = stride
        if stride == 2:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv1 = PDCConv2d(
            pdc_type,
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 2:
            x = self.pool(x)
        residual = self.conv2(self.relu(self.conv1(x)))
        if self.stride == 2:
            x = self.shortcut(x)
        return x + residual


class PiDiNetPyramid(nn.Module):
    """Return PiDiNet features at input, half, quarter, and eighth scale."""

    def __init__(self, base_channels: int = 30, pdc_arch: str = "carv4") -> None:
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        try:
            pdcs = PDC_ARCHITECTURES[pdc_arch]
        except KeyError as error:
            raise ValueError(
                f"Unknown PDC architecture {pdc_arch!r}; choose one of "
                f"{tuple(PDC_ARCHITECTURES)}."
            ) from error

        self.init_block = PDCConv2d(pdcs[0], 3, base_channels, kernel_size=3, padding=1)

        self.block1_1 = PDCBlock(pdcs[1], base_channels, base_channels)
        self.block1_2 = PDCBlock(pdcs[2], base_channels, base_channels)
        self.block1_3 = PDCBlock(pdcs[3], base_channels, base_channels)

        channels_2 = base_channels * 2
        self.block2_1 = PDCBlock(pdcs[4], base_channels, channels_2, stride=2)
        self.block2_2 = PDCBlock(pdcs[5], channels_2, channels_2)
        self.block2_3 = PDCBlock(pdcs[6], channels_2, channels_2)
        self.block2_4 = PDCBlock(pdcs[7], channels_2, channels_2)

        channels_3 = base_channels * 4
        self.block3_1 = PDCBlock(pdcs[8], channels_2, channels_3, stride=2)
        self.block3_2 = PDCBlock(pdcs[9], channels_3, channels_3)
        self.block3_3 = PDCBlock(pdcs[10], channels_3, channels_3)
        self.block3_4 = PDCBlock(pdcs[11], channels_3, channels_3)

        self.block4_1 = PDCBlock(pdcs[12], channels_3, channels_3, stride=2)
        self.block4_2 = PDCBlock(pdcs[13], channels_3, channels_3)
        self.block4_3 = PDCBlock(pdcs[14], channels_3, channels_3)
        self.block4_4 = PDCBlock(pdcs[15], channels_3, channels_3)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.init_block(x)
        level_0 = self.block1_3(self.block1_2(self.block1_1(x)))
        level_1 = self.block2_4(self.block2_3(self.block2_2(self.block2_1(level_0))))
        level_2 = self.block3_4(self.block3_3(self.block3_2(self.block3_1(level_1))))
        level_3 = self.block4_4(self.block4_3(self.block4_2(self.block4_1(level_2))))
        return level_0, level_1, level_2, level_3


__all__ = ["PDC_ARCHITECTURES", "PDCBlock", "PiDiNetPyramid"]
