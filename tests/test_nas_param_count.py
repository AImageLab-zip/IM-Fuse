from __future__ import annotations

import pytest
import torch

from mimose.models.simple_unet import SimpleUnet
from mimose.nas.arch_config import ArchitectureConfig, build_network
from mimose.nas.param_count import exact_parameters, expected_parameters

# Deliberately awkward width vectors: non-monotone, non-power-of-two, prime, and
# a bottleneck narrower than the stage above it. A formula that only handles
# doubling ladders passes on the easy cases and fails here.
WIDTH_CASES = [
    (64, 128, 256, 512, 1024),
    (8, 16, 32, 64, 128),
    (4, 8, 16, 32, 64),
    (12, 40, 32, 100, 192),
    (64, 128, 256, 512, 64),
    (16, 32, 48, 64),
    (8, 12, 20, 28, 36, 44),
]


@pytest.mark.parametrize("widths", WIDTH_CASES)
def test_exact_parameters_matches_simple_unet_numel(widths: tuple[int, ...]) -> None:
    """The integer formula must match PyTorch exactly, not approximately.

    This is what pins down the easily-missed terms: every ConvDropoutNormReLU's
    conv bias and InstanceNorm affine pair, the doubled decoder input from the
    skip concat, and the fact that all n_stages-1 seg layers are allocated even
    though deep_supervision is False.
    """
    model = SimpleUnet(features_per_stage=list(widths), input_shape=(32, 32, 32))
    assert exact_parameters(widths) == sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("widths", WIDTH_CASES)
def test_expected_parameters_agrees_with_exact_on_integer_widths(
    widths: tuple[int, ...],
) -> None:
    """The differentiable count reduces to the exact one at integer widths.

    Only checked for even bottlenecks: the mask-conditioning hidden width is
    ``bottleneck // 2``, and the soft version cannot represent that floor
    division at an odd bottleneck.
    """
    if widths[-1] % 2:
        pytest.skip("soft hidden width cannot express floor division at odd bottleneck")
    soft = expected_parameters(torch.tensor(widths, dtype=torch.float64))
    assert float(soft) == pytest.approx(float(exact_parameters(widths)), rel=1e-12)


def test_expected_parameters_is_differentiable_and_increasing() -> None:
    widths = torch.tensor([16.0, 32.0, 64.0, 128.0], requires_grad=True)
    total = expected_parameters(widths)
    total.backward()
    assert widths.grad is not None
    # Widening any stage can only add parameters.
    assert (widths.grad > 0).all()


def test_expected_parameters_supports_batched_widths() -> None:
    batch = torch.tensor([[8.0, 16.0, 32.0, 64.0], [16.0, 32.0, 64.0, 128.0]])
    out = expected_parameters(batch)
    assert out.shape == (2,)
    assert float(out[1]) > float(out[0])


def test_parameter_count_rejects_too_few_stages() -> None:
    with pytest.raises(ValueError, match="at least four stages"):
        exact_parameters((8, 16, 32))


def test_architecture_config_round_trips_through_build_network() -> None:
    config = ArchitectureConfig.uniform((8, 16, 32, 64))
    model = build_network(config, input_shape=(32, 32, 32))
    assert model.features_per_stage == (8, 16, 32, 64)
    assert config.parameter_count() == sum(p.numel() for p in model.parameters())


def test_architecture_config_validates_shapes() -> None:
    with pytest.raises(ValueError, match="at least four stages"):
        ArchitectureConfig.uniform((8, 16, 32))
    with pytest.raises(ValueError, match="n_blocks_decoder"):
        ArchitectureConfig(
            n_stages=4,
            features_per_stage=(8, 16, 32, 64),
            n_blocks_encoder=(2, 2, 2, 2),
            n_blocks_decoder=(2, 2, 2, 2),  # should be n_stages - 1
        )


def test_standard_configuration_reproduces_the_existing_baseline() -> None:
    """Phase 0 requirement: the stock config must match the shipped baseline.

    ``mimosa_base`` is one of the hand-written sweep points, so this is a direct
    check that the search space contains the architectures already in the paper.
    The total splits as 22,585,712 in the PlainConvUNet backbone plus 264,448 in
    the mask-conditioning MLP -- the latter is easy to forget when quoting
    "nnU-Net parameter counts", and is large enough to matter.
    """
    baseline = (32, 64, 128, 256, 512)
    config = ArchitectureConfig.uniform(baseline)
    model = build_network(config, input_shape=(32, 32, 32))

    assert config.parameter_count() == exact_parameters(baseline)
    assert exact_parameters(baseline) == 22_850_160
    assert sum(p.numel() for p in model.backbone.parameters()) == 22_585_712
    assert sum(p.numel() for p in model.mask_conditioning.parameters()) == 264_448
