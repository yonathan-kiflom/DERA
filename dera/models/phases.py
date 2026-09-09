"""Small utilities that make DERA's three training scopes explicit."""

from __future__ import annotations

from enum import Enum
from typing import Iterable

from torch import nn


class TrainingPhase(str, Enum):
    """The three sequential stages in the DERA training recipe."""

    FOUNDATION = "foundation"
    BOUNDARY = "boundary"
    RESIDUAL = "residual"

    @classmethod
    def parse(cls, value: str | "TrainingPhase") -> "TrainingPhase":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as error:
            choices = ", ".join(phase.value for phase in cls)
            raise ValueError(
                f"Unknown training phase {value!r}; choose one of: {choices}."
            ) from error


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    """Enable or disable gradients for every parameter in ``module``."""
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def activate_only(root: nn.Module, modules: Iterable[nn.Module]) -> None:
    """Freeze ``root``, then enable only the explicitly supplied modules."""
    set_requires_grad(root, False)
    for module in modules:
        set_requires_grad(module, True)


def trainable_parameter_names(module: nn.Module) -> tuple[str, ...]:
    """Return a deterministic manifest of parameters updated by an optimizer."""
    return tuple(
        sorted(
            name
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        )
    )


__all__ = [
    "TrainingPhase",
    "activate_only",
    "set_requires_grad",
    "trainable_parameter_names",
]
