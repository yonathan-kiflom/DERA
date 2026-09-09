# Copyright (c) OpenMMLab. All rights reserved.
# Modified for DERA under the repository's Apache-2.0 license.
"""Swin backbones for DERA's foundation and final models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from mmdet.models.backbones.swin import SwinTransformer
from mmdet.registry import MODELS
from torch import Tensor, nn
from torch.nn import functional as F

from third_party.pidinet import PiDiNetPyramid

from .boundary import make_side_head


@MODELS.register_module()
class DERAFoundationSwinTransformer(SwinTransformer):
    """Swin Transformer with four additive PiDiNet feature injections.

    PiDiNet runs in parallel with Swin. At each Swin stage, a learned 1x1
    projection aligns the corresponding PiDiNet feature's channels and the
    result is resized, tokenized, and added to the Swin tokens.

    Parameter names intentionally match the research implementation so its
    foundation checkpoints load directly: ``pidinet`` and ``pid_projs``.
    """

    def __init__(
        self,
        pidinet_channels: int = 30,
        pdc_arch: str = "carv4",
        cnn_pyramid_channels: Sequence[int] = (30, 60, 120, 120),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        cnn_pyramid_channels = tuple(cnn_pyramid_channels)
        if len(self.num_features) != 4:
            raise ValueError("DERA requires a four-stage Swin backbone")
        if len(cnn_pyramid_channels) != 4:
            raise ValueError("cnn_pyramid_channels must contain four entries")

        self.pidinet = PiDiNetPyramid(pidinet_channels, pdc_arch)
        self.pid_projs = nn.ModuleList(
            nn.Conv2d(pid_channels, swin_channels, kernel_size=1)
            for pid_channels, swin_channels in zip(
                cnn_pyramid_channels, self.num_features
            )
        )

    def train(self, mode: bool = True) -> "DERAFoundationSwinTransformer":
        """Set training mode while preserving PyTorch's fluent API."""
        super().train(mode)
        return self

    def _project_pid_feature(
        self,
        feature: Tensor,
        stage_index: int,
        output_size: tuple[int, int],
    ) -> Tensor:
        """Align one PiDiNet map with one Swin token sequence."""
        feature = self.pid_projs[stage_index](feature)
        if feature.shape[-2:] != output_size:
            feature = F.interpolate(
                feature,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return feature.flatten(2).transpose(1, 2)

    def _foundation_tokens(
        self,
        swin_tokens: Tensor,
        pid_feature: Tensor,
        stage_index: int,
        output_size: tuple[int, int],
    ) -> Tensor:
        """Apply DERA's foundation fusion: elementwise addition."""
        return swin_tokens + self._project_pid_feature(
            pid_feature, stage_index, output_size
        )

    def _format_stage_output(
        self,
        tokens: Tensor,
        stage_index: int,
        output_size: tuple[int, int],
    ) -> Tensor:
        norm = getattr(self, f"norm{stage_index}")
        tokens = norm(tokens)
        return (
            tokens.view(-1, *output_size, self.num_features[stage_index])
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        pid_features = self.pidinet(x)
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outputs: list[Tensor] = []
        for stage_index, stage in enumerate(self.stages):
            for block in stage.blocks:
                x = block(x, hw_shape)

            # Foundation fusion is fixed at all four hierarchical levels.
            x = self._foundation_tokens(
                x, pid_features[stage_index], stage_index, hw_shape
            )

            stage_output = x
            stage_output_size = hw_shape
            if stage.downsample is not None:
                x, hw_shape = stage.downsample(x, hw_shape)

            if stage_index in self.out_indices:
                outputs.append(
                    self._format_stage_output(
                        stage_output, stage_index, stage_output_size
                    )
                )
        return outputs


@MODELS.register_module()
class DERASwinTransformer(DERAFoundationSwinTransformer):
    """Final DERA backbone with a two-level edge-gated residual pathway.

    The four foundation injections remain unchanged. A frozen boundary head
    predicts one spatial prior from PiDiNet levels 0 and 1. Two independent,
    zero-initialized 1x1 projections add gated residuals only at Swin stages
    0 and 1.
    """

    EDGE_STAGES = (0, 1)

    def __init__(
        self,
        cnn_pyramid_channels: Sequence[int] = (30, 60, 120, 120),
        edge_head_channels: int = 32,
        **kwargs,
    ) -> None:
        cnn_pyramid_channels = tuple(cnn_pyramid_channels)
        super().__init__(
            cnn_pyramid_channels=cnn_pyramid_channels,
            **kwargs,
        )

        self.edge_side_heads = nn.ModuleList(
            make_side_head(cnn_pyramid_channels[stage], edge_head_channels)
            for stage in self.EDGE_STAGES
        )
        self.edge_fuse = nn.Conv2d(len(self.EDGE_STAGES), 1, kernel_size=1)
        nn.init.constant_(self.edge_fuse.weight, 1 / len(self.EDGE_STAGES))
        nn.init.zeros_(self.edge_fuse.bias)

        self.edge_residual_projs = nn.ModuleDict()
        for stage in self.EDGE_STAGES:
            projection = nn.Conv2d(
                cnn_pyramid_channels[stage],
                self.num_features[stage],
                kernel_size=1,
            )
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
            self.edge_residual_projs[str(stage)] = projection

    def edge_head_modules(self) -> tuple[nn.Module, ...]:
        """Return exactly the modules updated during boundary training."""
        return self.edge_side_heads, self.edge_fuse

    def edge_residual_modules(self) -> tuple[nn.Module, ...]:
        """Return exactly the modules updated during residual training."""
        return (self.edge_residual_projs,)

    def _predict_edge_logits(
        self,
        pid_features: tuple[Tensor, Tensor, Tensor, Tensor],
        output_size: tuple[int, int],
    ) -> Tensor:
        side_logits: list[Tensor] = []
        for stage, head in zip(self.EDGE_STAGES, self.edge_side_heads):
            feature = pid_features[stage].detach()
            if feature.shape[-2:] != output_size:
                feature = F.interpolate(
                    feature,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            side_logits.append(head(feature))

        # The side predictions are concatenated; the fusion layer learns how
        # to combine them into a single boundary logit map.
        return self.edge_fuse(torch.cat(side_logits, dim=1))

    def _edge_residual_tokens(
        self,
        pid_feature: Tensor,
        edge_prior: Tensor,
        stage_index: int,
        output_size: tuple[int, int],
    ) -> Tensor:
        feature = pid_feature.detach()
        if feature.shape[-2:] != output_size:
            feature = F.interpolate(
                feature,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        if edge_prior.shape[-2:] != output_size:
            edge_prior = F.interpolate(
                edge_prior,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        # Multiplication performs boundary gating; addition happens below.
        gated_feature = feature * edge_prior
        residual = self.edge_residual_projs[str(stage_index)](gated_feature)
        return residual.flatten(2).transpose(1, 2)

    def _forward_impl(
        self, x: Tensor, return_edge_logits: bool
    ) -> list[Tensor] | tuple[list[Tensor], Tensor]:
        pid_features = self.pidinet(x)
        x, hw_shape = self.patch_embed(x)
        edge_logits = self._predict_edge_logits(pid_features, hw_shape)
        edge_prior = edge_logits.sigmoid().detach()

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outputs: list[Tensor] = []
        for stage_index, stage in enumerate(self.stages):
            for block in stage.blocks:
                x = block(x, hw_shape)

            # Keep the original four-level foundation fusion.
            x = self._foundation_tokens(
                x, pid_features[stage_index], stage_index, hw_shape
            )

            # Add a separate residual only at the first two stages.
            if stage_index in self.EDGE_STAGES:
                x = x + self._edge_residual_tokens(
                    pid_features[stage_index],
                    edge_prior,
                    stage_index,
                    hw_shape,
                )

            stage_output = x
            stage_output_size = hw_shape
            if stage.downsample is not None:
                x, hw_shape = stage.downsample(x, hw_shape)

            if stage_index in self.out_indices:
                outputs.append(
                    self._format_stage_output(
                        stage_output, stage_index, stage_output_size
                    )
                )

        if return_edge_logits:
            return outputs, edge_logits
        return outputs

    def forward(self, x: Tensor) -> list[Tensor]:
        outputs = self._forward_impl(x, return_edge_logits=False)
        assert isinstance(outputs, list)
        return outputs

    def forward_with_edge_logits(self, x: Tensor) -> tuple[list[Tensor], Tensor]:
        result = self._forward_impl(x, return_edge_logits=True)
        assert isinstance(result, tuple)
        return result

    def forward_edge_logits(self, x: Tensor) -> Tensor:
        """Run the frozen PiDiNet path and the trainable boundary head."""
        with torch.no_grad():
            pid_features = self.pidinet(x)
            _, hw_shape = self.patch_embed(x)
        return self._predict_edge_logits(pid_features, hw_shape)


__all__ = ["DERAFoundationSwinTransformer", "DERASwinTransformer"]
