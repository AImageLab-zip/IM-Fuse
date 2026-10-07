from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimose.models.simple_unet import SimpleUnet
from mimose.nas.flop_count import exact_macs, expected_macs, stage_volumes

# Same awkward width vectors as test_nas_param_count: non-monotone,
# non-power-of-two, prime, and a bottleneck narrower than the stage above it.
WIDTH_CASES = [
    (64, 128, 256, 512, 1024),
    (8, 16, 32, 64, 128),
    (4, 8, 16, 32, 64),
    (12, 40, 32, 100, 192),
    (64, 128, 256, 512, 64),
    (16, 32, 48, 64),
]

INPUT_SHAPE = (32, 32, 32)


def measured_macs(widths: tuple[int, ...], shape: tuple[int, int, int]) -> int:
    """Ground truth: hook every conv and count C_in * C_out * k**3 * out_volume.

    Counts what the network *evaluates*, which is the point -- with
    deep_supervision=False the seg layers are all allocated but only the last is
    run, so a formula derived from the parameter inventory overcounts.
    """
    model = SimpleUnet(features_per_stage=list(widths), input_shape=shape).eval()
    total = 0

    def hook(module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        nonlocal total
        k = 1
        for d in module.kernel_size:
            k *= d
        volume = 1
        for d in output.shape[2:]:
            volume *= d
        if isinstance(module, nn.ConvTranspose3d):
            # k**3 MACs per *input* voxel; for kernel == stride == 2 the output
            # volume is 8x the input, so divide it back out.
            volume //= k
        total += module.in_channels * module.out_channels * k * volume

    handles = [
        m.register_forward_hook(hook)
        for m in model.modules()
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d))
    ]
    with torch.no_grad():
        model(torch.zeros(1, 4, *shape), torch.ones(1, 4, dtype=torch.bool))
    for h in handles:
        h.remove()
    return total


@pytest.mark.parametrize("widths", WIDTH_CASES)
def test_exact_macs_matches_a_hooked_forward(widths: tuple[int, ...]) -> None:
    """The formula must match the convolutions the model actually runs."""
    assert exact_macs(widths, input_shape=INPUT_SHAPE) == measured_macs(
        widths, INPUT_SHAPE
    )


@pytest.mark.parametrize("widths", WIDTH_CASES)
def test_expected_macs_agrees_with_exact_on_integer_widths(
    widths: tuple[int, ...],
) -> None:
    soft = torch.tensor(widths, dtype=torch.float64)
    expected = float(expected_macs(soft, input_shape=INPUT_SHAPE))
    assert expected == pytest.approx(
        float(exact_macs(widths, input_shape=INPUT_SHAPE)), rel=1e-9
    )


def test_expected_macs_is_differentiable_and_weights_shallow_stages_most() -> None:
    """The whole reason this module exists.

    Under a parameter budget the bottleneck dominates the gradient, so the search
    spends its cuts there -- where they buy no speed. Under a MAC budget the
    ordering reverses by orders of magnitude, which is what pushes cuts into the
    high-resolution stages.
    """
    widths = torch.tensor([64.0, 128.0, 256.0, 512.0, 1024.0], requires_grad=True)
    expected_macs(widths).backward()
    grad = widths.grad
    assert torch.isfinite(grad).all()
    # Strictly decreasing cost-per-channel as resolution halves ...
    assert (grad[:-1] > grad[1:]).all()
    # ... and the shallow end dominates by more than two orders of magnitude.
    # The exact ratio depends on the width vector -- it is ~811x at full width
    # and ~2000x at the widths the trained pareto_23 supernet actually realizes,
    # because a narrower bottleneck shrinks the deepest stage's share further.
    assert float(grad[0] / grad[-1]) > 500.0


def test_stage_volumes_halves_each_axis() -> None:
    assert stage_volumes((128, 128, 128), 5) == [
        128**3,
        64**3,
        32**3,
        16**3,
        8**3,
    ]


def test_batched_soft_widths_are_supported() -> None:
    """The trainer evaluates one budget at a time, but the penalty path keeps a
    leading batch axis; a formula that silently reduces over it would be wrong."""
    batch = torch.tensor([[8.0, 16, 32, 64, 128], [16.0, 32, 64, 128, 256]])
    out = expected_macs(batch, input_shape=INPUT_SHAPE)
    assert out.shape == (2,)
    assert float(out[0]) == pytest.approx(
        float(exact_macs((8, 16, 32, 64, 128), input_shape=INPUT_SHAPE))
    )
