# Copyright (c) OpenMMLab. All rights reserved.
# Modified for DERA under the repository's Apache-2.0 license.
"""Grounding DINO wrapper for DERA's three-stage training recipe."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.registry import MODELS
from torch import Tensor, nn

from .boundary import build_boundary_targets, sigmoid_focal_boundary_loss
from .phases import (
    TrainingPhase,
    activate_only,
    set_requires_grad,
    trainable_parameter_names,
)


@MODELS.register_module()
class DERAGroundingDINO(GroundingDINO):
    """Grounding DINO with explicit DERA training phases.

    ``foundation`` trains the detector and four-level PiDiNet/Swin fusion but
    keeps BERT frozen. ``boundary`` trains only the two side heads and their
    fusion layer. ``residual`` trains only the two zero-initialized residual
    projections. Boundary masks are required only in the second phase.
    """

    def __init__(
        self,
        *args,
        training_phase: str = "residual",
        edge_loss_weight: float = 1.0,
        edge_kernel_size: int = 3,
        edge_focal_alpha: float = 0.75,
        edge_focal_gamma: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.training_phase = TrainingPhase.parse(training_phase)
        if edge_loss_weight < 0:
            raise ValueError("edge_loss_weight must be non-negative")
        if edge_kernel_size < 1 or edge_kernel_size % 2 == 0:
            raise ValueError("edge_kernel_size must be a positive odd number")
        if not 0 <= edge_focal_alpha <= 1:
            raise ValueError("edge_focal_alpha must be in [0, 1]")
        if edge_focal_gamma < 0:
            raise ValueError("edge_focal_gamma must be non-negative")

        self.edge_loss_weight = float(edge_loss_weight)
        self.edge_kernel_size = edge_kernel_size
        self.edge_focal_alpha = float(edge_focal_alpha)
        self.edge_focal_gamma = float(edge_focal_gamma)
        self._validate_phase_backbone()
        self._configure_trainable_parameters()

    def _validate_phase_backbone(self) -> None:
        if not hasattr(self.backbone, "pidinet") or not hasattr(
            self.backbone, "pid_projs"
        ):
            raise TypeError("DERA requires a PiDiNet/Swin foundation backbone")
        if self.training_phase in {
            TrainingPhase.BOUNDARY,
            TrainingPhase.RESIDUAL,
        } and not hasattr(self.backbone, "forward_edge_logits"):
            raise TypeError("Boundary and residual phases require DERASwinTransformer")

    def _phase_modules(self) -> tuple[nn.Module, ...]:
        if self.training_phase is TrainingPhase.BOUNDARY:
            return tuple(self.backbone.edge_head_modules())
        if self.training_phase is TrainingPhase.RESIDUAL:
            return tuple(self.backbone.edge_residual_modules())
        raise RuntimeError("foundation training does not use phase modules")

    def _configure_trainable_parameters(self) -> None:
        """Enforce the parameter ownership of the selected training phase."""
        if self.training_phase is TrainingPhase.FOUNDATION:
            set_requires_grad(self, True)
            set_requires_grad(self.language_model, False)
            # A final DERA backbone is accepted for convenience, but its later
            # modules must not leak into foundation adaptation.
            if hasattr(self.backbone, "edge_head_modules"):
                for module in self.backbone.edge_head_modules():
                    set_requires_grad(module, False)
            if hasattr(self.backbone, "edge_residual_modules"):
                for module in self.backbone.edge_residual_modules():
                    set_requires_grad(module, False)
        else:
            activate_only(self, self._phase_modules())

        self.declared_trainable_names = trainable_parameter_names(self)

    def set_training_phase(self, phase: str | TrainingPhase) -> None:
        """Select a phase and immediately reapply its strict update scope."""
        self.training_phase = TrainingPhase.parse(phase)
        self._validate_phase_backbone()
        self._configure_trainable_parameters()

    def train(self, mode: bool = True) -> "DERAGroundingDINO":
        super().train(mode)
        if not mode:
            return self

        if self.training_phase is TrainingPhase.FOUNDATION:
            # BERT weights are frozen, but its training/evaluation mode follows
            # Grounding DINO exactly so the released recipe remains numerically
            # compatible with the original foundation run.
            if hasattr(self.backbone, "edge_head_modules"):
                for module in self.backbone.edge_head_modules():
                    module.eval()
            if hasattr(self.backbone, "edge_residual_modules"):
                for module in self.backbone.edge_residual_modules():
                    module.eval()
            return self

        # Retain self.training=True for Grounding DINO's denoising-query path,
        # while every frozen component remains deterministic.
        for module in self.modules():
            module.training = False
        self.training = True
        for module in self._phase_modules():
            module.train(True)
        return self

    def _boundary_loss(
        self,
        batch_inputs: Tensor,
        batch_data_samples: Sequence[Any],
    ) -> Tensor:
        edge_logits = self.backbone.forward_edge_logits(batch_inputs)
        targets, valid_regions = build_boundary_targets(
            edge_logits,
            batch_inputs,
            batch_data_samples,
            kernel_size=self.edge_kernel_size,
        )
        return sigmoid_focal_boundary_loss(
            edge_logits,
            targets,
            valid_regions,
            alpha=self.edge_focal_alpha,
            gamma=self.edge_focal_gamma,
        )

    def loss(
        self,
        batch_inputs: Tensor,
        batch_data_samples: Sequence[Any],
    ) -> dict[str, Tensor]:
        if self.training_phase is not TrainingPhase.BOUNDARY:
            return super().loss(batch_inputs, batch_data_samples)
        return {
            "loss_edge_focal": self.edge_loss_weight
            * self._boundary_loss(batch_inputs, batch_data_samples)
        }


__all__ = ["DERAGroundingDINO"]
