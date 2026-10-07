from __future__ import annotations

import pytest
import torch
from torch import nn

from mimose.models.elastic_unet import ElasticUnet
from mimose.models.simple_unet import SimpleUnet
from mimose.nas.budget import budget_to_z
from mimose.nas.calibration import calibrate_thresholds_analytic
from mimose.nas.extract import active_indices, concat_indices, extract_subnet

INPUT_SHAPE = (32, 32, 32)


def build_supernet(group_size: int = 2) -> ElasticUnet:
    torch.manual_seed(0)
    supernet = ElasticUnet(
        features_per_stage=(8, 16, 32, 64),
        input_shape=INPUT_SHAPE,
        group_size=group_size,
        tau=0.05,
    )
    calibrate_thresholds_analytic(supernet.gates)
    return supernet.eval()


def sample_inputs(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    images = torch.randn(batch, 4, *INPUT_SHAPE)
    # Include a sample with a missing modality so the conditioning path is live.
    mask = torch.tensor([[1, 0, 1, 1], [1, 1, 1, 1]], dtype=torch.bool)[:batch]
    return images, mask


def budgets_across_range(supernet: ElasticUnet, n: int = 7) -> list[float]:
    lo, hi = supernet.gates.budget_min, supernet.gates.budget_max
    return [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]


# ------------------------------------------------------- the load-bearing test


@pytest.mark.parametrize("index", range(7))
def test_gated_supernet_matches_extracted_model(index: int) -> None:
    """The supernet's gated forward must equal the standalone network it yields.

    Everything else rests on this: if it fails, the architecture a budget
    selects is not the architecture that gets evaluated. Widths here are
    non-uniform and not powers of two, because a uniform scale ratio can hide an
    indexing bug that a ragged one exposes.
    """
    supernet = build_supernet()
    budget = budgets_across_range(supernet)[index]
    images, mask = sample_inputs()

    supernet.set_budget(budget)
    with torch.no_grad():
        supernet_output = supernet(images, mask)

    subnet = extract_subnet(supernet, budget).eval()
    with torch.no_grad():
        subnet_output = subnet(images, mask)

    assert subnet_output.shape == supernet_output.shape
    assert torch.allclose(supernet_output, subnet_output, atol=1e-5)


@pytest.mark.parametrize("index", range(7))
def test_extracted_parameter_count_matches_prediction(index: int) -> None:
    supernet = build_supernet()
    budget = budgets_across_range(supernet)[index]
    supernet.set_budget(budget)

    subnet = extract_subnet(supernet, budget)
    assert sum(p.numel() for p in subnet.parameters()) == supernet.actual_parameters()
    assert subnet.features_per_stage == supernet.current_widths()


# ------------------------------------------------------- regression guards
# These exist to stop someone "simplifying" the two-block gathers into
# contiguous slices. Every other test in this file still passes if they do.


def test_concat_indices_selects_from_both_halves() -> None:
    idx = torch.arange(3)
    assert concat_indices(idx, 8).tolist() == [0, 1, 2, 8, 9, 10]
    # The wrong answer a contiguous slice would give:
    assert concat_indices(idx, 8).tolist() != list(range(6))


def test_contiguous_decoder_slice_would_be_wrong() -> None:
    """A contiguous slice of the decoder's concat axis must change the output.

    The decoder stage consumes ``cat(upsampled, skip)``, so its live input
    channels are two disjoint blocks. Slicing ``[:2c]`` instead reinterprets
    part of the upsampled features as skip features -- which still runs, still
    loads, and is silently wrong.
    """
    supernet = build_supernet()
    budget = budgets_across_range(supernet)[3]
    images, mask = sample_inputs()

    correct = extract_subnet(supernet, budget).eval()

    # Rebuild the same subnet but corrupt exactly one decoder input gather.
    broken = extract_subnet(supernet, budget).eval()
    z = budget_to_z(budget, supernet.gates.budget_min, supernet.gates.budget_max)
    idx = active_indices(supernet.gates, z)
    n_stages = len(supernet.max_features)
    level = n_stages - 2
    skip_idx = idx[level]
    src_conv = supernet.supernet.backbone.decoder.stages[0].convs[0].conv
    with torch.no_grad():
        broken.backbone.decoder.stages[0].convs[0].conv.weight.copy_(
            src_conv.weight.index_select(0, skip_idx).index_select(
                1, torch.arange(2 * skip_idx.numel())  # the tempting shortcut
            )
        )

    with torch.no_grad():
        assert (correct(images, mask) - broken(images, mask)).abs().max() > 1e-2


def test_contiguous_mask_conditioning_slice_would_be_wrong() -> None:
    """Same guard for the conditioning MLP's ``chunk(2)`` scale|bias split."""
    supernet = build_supernet()
    budget = budgets_across_range(supernet)[3]
    images, mask = sample_inputs()

    correct = extract_subnet(supernet, budget).eval()
    broken = extract_subnet(supernet, budget).eval()

    z = budget_to_z(budget, supernet.gates.budget_min, supernet.gates.budget_max)
    bottleneck_idx = active_indices(supernet.gates, z)[-1]
    src = supernet.supernet.mask_conditioning.mlp[2]
    hidden_idx = torch.arange(broken.mask_conditioning.mlp[0].out_features)
    with torch.no_grad():
        wrong = torch.arange(2 * bottleneck_idx.numel())
        broken.mask_conditioning.mlp[2].weight.copy_(
            src.weight.index_select(0, wrong).index_select(1, hidden_idx)
        )
        broken.mask_conditioning.mlp[2].bias.copy_(src.bias.index_select(0, wrong))

    with torch.no_grad():
        assert (correct(images, mask) - broken(images, mask)).abs().max() > 1e-4


# ------------------------------------------------------------ gating placement


def test_inactive_channels_are_exactly_zero_after_every_block() -> None:
    """Gates must sit after ``conv -> norm -> nonlin``, never before the norm.

    See the companion test below for why: an InstanceNorm applied to a zeroed
    channel emits its bias, not zero, so gating earlier would leak constants
    into the live channels of the next convolution.
    """
    from dynamic_network_architectures.building_blocks.simple_conv_blocks import (
        ConvDropoutNormReLU,
    )

    supernet = build_supernet()
    budget = budgets_across_range(supernet)[2]
    supernet.set_budget(budget)
    widths = supernet.current_widths()
    images, mask = sample_inputs()

    violations: list[str] = []
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            channels = output.shape[1]
            # Match this activation to the stage width it belongs to.
            if channels in widths:
                live = widths[widths.index(channels)]
                if live < channels and output[:, live:].abs().max() > 0:
                    violations.append(name)

        return hook

    for name, module in supernet.named_modules():
        if isinstance(module, ConvDropoutNormReLU):
            handles.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            supernet(images, mask)
    finally:
        for handle in handles:
            handle.remove()

    assert not violations, f"gate applied too early in: {violations}"


def test_instance_norm_maps_a_zeroed_channel_to_its_bias() -> None:
    """Documents the library behaviour the gate placement is designed around.

    A zeroed channel has zero variance, so normalization yields 0 -- and then the
    affine transform adds the bias. The channel comes out constant and non-zero,
    not zero, and not NaN.
    """
    norm = nn.InstanceNorm3d(4, eps=1e-5, affine=True)
    with torch.no_grad():
        norm.bias.copy_(torch.tensor([1.5, -0.25, -2.0, 0.5]))
        norm.weight.fill_(1.0)

    x = torch.randn(1, 4, 8, 8, 8)
    x[:, 2:] = 0.0
    out = norm(x)

    assert not torch.isnan(out).any()
    assert out[0, 2].unique().tolist() == pytest.approx([-2.0])
    assert out[0, 3].unique().tolist() == pytest.approx([0.5])


def test_transpconv_weight_axis_order_is_in_out() -> None:
    """``ConvTranspose3d`` stores ``[in, out, ...]`` -- the opposite of ``Conv3d``.

    Extraction slices those axes, so a library change here would silently
    transpose the inheritance.
    """
    model = SimpleUnet(features_per_stage=(8, 16, 32, 64), input_shape=INPUT_SHAPE)
    # Decoder stage 0 upsamples the bottleneck (64) to the stage below it (32).
    transpconv = model.backbone.decoder.transpconvs[0]
    assert tuple(transpconv.weight.shape) == (64, 32, 2, 2, 2)
    # And the decoder stage consumes the skip concat, hence twice the width.
    assert model.backbone.decoder.stages[0].convs[0].conv.in_channels == 64


# ----------------------------------------------------------------- robustness


def test_gradients_are_finite_at_min_budget_with_all_modalities_missing() -> None:
    """Narrowest subnet plus an all-zero input: the two degenerate paths at once."""
    supernet = build_supernet()
    supernet.set_budget(supernet.gates.budget_min)
    images, _ = sample_inputs(batch=1)
    mask = torch.zeros(1, 4, dtype=torch.bool)

    output = supernet(images, mask)
    assert torch.isfinite(output).all()

    output.sum().backward()
    grads = [p.grad for p in supernet.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_export_preserves_the_supernet_not_one_operating_point() -> None:
    """The exported checkpoint must retain the whole family.

    Exporting an extracted subnet here would reduce the final artifact to
    whichever budget happened to be set last, and the dense Pareto sweep would
    have nothing to extract from.
    """
    supernet = build_supernet()
    supernet.set_budget(budgets_across_range(supernet)[2])

    exported = supernet.model_for_export()
    assert exported is supernet

    # And the recorded kwargs must rebuild the supernet, since the HF config
    # round-trip relies on them.
    config = supernet.get_hf_config()
    assert config["model_class"] == "ElasticUnet"
    rebuilt = ElasticUnet(**config["model_kwargs"])
    assert rebuilt.max_features == supernet.max_features
    assert sum(p.numel() for p in rebuilt.parameters()) == sum(
        p.numel() for p in supernet.parameters()
    )


def test_extract_for_budget_returns_a_deployable_subnet() -> None:
    supernet = build_supernet()
    budget = budgets_across_range(supernet)[2]
    supernet.set_budget(budget)

    subnet = supernet.extract_for_budget()
    assert isinstance(subnet, SimpleUnet)
    assert subnet.features_per_stage == supernet.current_widths()

    # Reconstructible from its own recorded kwargs.
    rebuilt = SimpleUnet(**subnet._mimose_model_kwargs)
    assert rebuilt.features_per_stage == subnet.features_per_stage


def test_budget_buffer_is_updated_in_place() -> None:
    """torch.compile's CUDA-graph mode captures addresses; rebinding breaks it."""
    supernet = build_supernet()
    before = supernet._budget
    supernet.set_budget(supernet.gates.budget_min)
    assert supernet._budget is before


def test_setting_a_budget_changes_the_selected_architecture() -> None:
    supernet = build_supernet()
    supernet.set_budget(supernet.gates.budget_min)
    small = supernet.current_widths()
    supernet.set_budget(supernet.gates.budget_max)
    large = supernet.current_widths()

    assert small != large
    assert all(a <= b for a, b in zip(small, large))
