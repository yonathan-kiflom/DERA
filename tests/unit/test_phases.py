"""Tests proving the strict trainable scope of every DERA phase."""

import pytest
from torch import nn

from dera.models import DERAGroundingDINO, TrainingPhase


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = nn.Linear(1, 1, bias=False)
        self.pidinet = nn.Linear(1, 1, bias=False)
        self.pid_projs = nn.ModuleList([nn.Linear(1, 1, bias=False)])
        self.edge_side_heads = nn.ModuleList(
            [nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)]
        )
        self.edge_fuse = nn.Linear(2, 1, bias=False)
        self.edge_residual_projs = nn.ModuleDict(
            {
                "0": nn.Linear(1, 1, bias=False),
                "1": nn.Linear(1, 1, bias=False),
            }
        )

    def edge_head_modules(self):
        return self.edge_side_heads, self.edge_fuse

    def edge_residual_modules(self):
        return (self.edge_residual_projs,)

    def forward_edge_logits(self, inputs):
        return inputs[:, :1]


def _bare_detector(phase):
    model = DERAGroundingDINO.__new__(DERAGroundingDINO)
    nn.Module.__init__(model)
    model.backbone = _Backbone()
    model.language_model = nn.Linear(1, 1, bias=False)
    model.neck = nn.Linear(1, 1, bias=False)
    model.bbox_head = nn.Linear(1, 1, bias=False)
    model.training_phase = TrainingPhase.parse(phase)
    model._configure_trainable_parameters()
    return model


@pytest.mark.parametrize(
    "phase, expected",
    [
        (
            "foundation",
            {
                "backbone.swin.weight",
                "backbone.pidinet.weight",
                "backbone.pid_projs.0.weight",
                "neck.weight",
                "bbox_head.weight",
            },
        ),
        (
            "boundary",
            {
                "backbone.edge_side_heads.0.weight",
                "backbone.edge_side_heads.1.weight",
                "backbone.edge_fuse.weight",
            },
        ),
        (
            "residual",
            {
                "backbone.edge_residual_projs.0.weight",
                "backbone.edge_residual_projs.1.weight",
            },
        ),
    ],
)
def test_exact_parameter_ownership(phase, expected):
    model = _bare_detector(phase)
    actual = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    assert actual == expected
    assert set(model.declared_trainable_names) == expected
    assert not model.language_model.weight.requires_grad


@pytest.mark.parametrize(
    "phase, active_names",
    [
        ("boundary", {"edge_side_heads", "edge_fuse"}),
        ("residual", {"edge_residual_projs"}),
    ],
)
def test_late_stage_train_mode_keeps_frozen_modules_in_eval(phase, active_names):
    model = _bare_detector(phase)
    model.train()

    modules = {
        "swin": model.backbone.swin,
        "pidinet": model.backbone.pidinet,
        "pid_projs": model.backbone.pid_projs,
        "edge_side_heads": model.backbone.edge_side_heads,
        "edge_fuse": model.backbone.edge_fuse,
        "edge_residual_projs": model.backbone.edge_residual_projs,
        "language_model": model.language_model,
        "neck": model.neck,
        "bbox_head": model.bbox_head,
    }
    assert model.training
    for name, module in modules.items():
        assert module.training is (name in active_names)


def test_foundation_keeps_bert_weights_and_later_modules_frozen():
    model = _bare_detector("foundation")
    model.train()

    assert model.backbone.swin.training
    assert model.backbone.pidinet.training
    assert model.language_model.training
    assert not model.language_model.weight.requires_grad
    assert not model.backbone.edge_side_heads.training
    assert not model.backbone.edge_fuse.training
    assert not model.backbone.edge_residual_projs.training


def test_phase_names_are_strict_and_clear():
    with pytest.raises(ValueError, match="Unknown training phase"):
        TrainingPhase.parse("edge_pretrain")
