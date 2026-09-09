"""Architectural invariant and checkpoint-compatibility tests."""

import torch

from dera.models import DERAFoundationSwinTransformer, DERASwinTransformer


def _tiny_kwargs():
    return dict(
        embed_dims=8,
        depths=(1, 1, 1, 1),
        num_heads=(1, 2, 4, 8),
        window_size=2,
        mlp_ratio=2,
        drop_path_rate=0,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        pidinet_channels=4,
        cnn_pyramid_channels=(4, 8, 16, 16),
    )


def test_foundation_fuses_all_four_hierarchical_levels():
    model = DERAFoundationSwinTransformer(**_tiny_kwargs()).eval()
    projection_calls = [0, 0, 0, 0]
    handles = []
    for index, projection in enumerate(model.pid_projs):
        handles.append(
            projection.register_forward_hook(
                lambda _module, _inputs, _output, index=index: (
                    projection_calls.__setitem__(index, projection_calls[index] + 1)
                )
            )
        )

    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 32, 32))
    for handle in handles:
        handle.remove()

    assert projection_calls == [1, 1, 1, 1]
    assert [tuple(output.shape) for output in outputs] == [
        (1, 8, 8, 8),
        (1, 16, 4, 4),
        (1, 32, 2, 2),
        (1, 64, 1, 1),
    ]


def test_final_backbone_uses_two_boundary_heads_and_two_adapters():
    model = DERASwinTransformer(edge_head_channels=8, **_tiny_kwargs())

    assert len(model.edge_side_heads) == 2
    assert tuple(model.edge_residual_projs) == ("0", "1")
    assert model.EDGE_STAGES == (0, 1)
    assert all(
        torch.count_nonzero(parameter) == 0
        for parameter in model.edge_residual_projs.parameters()
    )

    state_keys = tuple(model.state_dict())
    for required_prefix in (
        "pidinet.",
        "pid_projs.",
        "edge_side_heads.",
        "edge_fuse.",
        "edge_residual_projs.",
    ):
        assert any(key.startswith(required_prefix) for key in state_keys)


def test_zero_residuals_exactly_preserve_the_foundation_outputs():
    foundation = DERAFoundationSwinTransformer(**_tiny_kwargs()).eval()
    final = DERASwinTransformer(edge_head_channels=8, **_tiny_kwargs()).eval()
    incompatible = final.load_state_dict(foundation.state_dict(), strict=False)

    assert not incompatible.unexpected_keys
    assert len(incompatible.missing_keys) == 16
    assert all(
        key.startswith(("edge_side_heads.", "edge_fuse.", "edge_residual_projs."))
        for key in incompatible.missing_keys
    )

    inputs = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        expected = foundation(inputs)
        actual, edge_logits = final.forward_with_edge_logits(inputs)

    assert edge_logits.shape == (1, 1, 8, 8)
    for expected_level, actual_level in zip(expected, actual):
        torch.testing.assert_close(actual_level, expected_level, rtol=0, atol=0)
