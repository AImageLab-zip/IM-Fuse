from __future__ import annotations

import math

import pytest
import torch

from mimose.nas.budget import budget_to_z, sample_log_uniform_budget, z_to_budget
from mimose.nas.calibration import (
    alpha_for_budget,
    budget_param_curve,
    calibrate_thresholds_analytic,
    uniform_width_schedule,
)
from mimose.nas.gates import WidthGates, straight_through
from mimose.nas.param_count import exact_parameters

MAX_FEATURES = (64, 128, 256, 512, 1024)
BUDGET_MIN = float(exact_parameters((8, 16, 32, 64, 128)))
BUDGET_MAX = float(exact_parameters(MAX_FEATURES))


def build_gates(group_size: int = 4, tau: float = 0.05) -> WidthGates:
    return WidthGates(
        MAX_FEATURES,
        budget_min=BUDGET_MIN,
        budget_max=BUDGET_MAX,
        group_size=group_size,
        tau=tau,
    )


def log_spaced_budgets(n: int) -> list[float]:
    lo, hi = math.log(BUDGET_MIN), math.log(BUDGET_MAX)
    return [math.exp(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


# ---------------------------------------------------------------- budget maths


def test_budget_z_round_trip() -> None:
    for budget in log_spaced_budgets(16):
        z = budget_to_z(budget, BUDGET_MIN, BUDGET_MAX)
        assert float(z_to_budget(z, BUDGET_MIN, BUDGET_MAX)) == pytest.approx(budget)


def test_budget_endpoints_map_to_unit_interval() -> None:
    assert float(budget_to_z(BUDGET_MIN, BUDGET_MIN, BUDGET_MAX)) == pytest.approx(0.0)
    assert float(budget_to_z(BUDGET_MAX, BUDGET_MIN, BUDGET_MAX)) == pytest.approx(1.0)


def test_log_uniform_sampling_is_uniform_in_log_space() -> None:
    generator = torch.Generator().manual_seed(0)
    budgets = sample_log_uniform_budget(
        20_000, BUDGET_MIN, BUDGET_MAX, generator=generator
    )
    z = budget_to_z(budgets, BUDGET_MIN, BUDGET_MAX)
    # Uniform on [0, 1] => mean 1/2. Linear-space sampling would sit near 0.9.
    assert float(z.mean()) == pytest.approx(0.5, abs=0.02)
    assert float(z.min()) >= 0.0 and float(z.max()) <= 1.0


# ---------------------------------------------------------------------- gates


def test_straight_through_is_binary_forward_and_differentiable_backward() -> None:
    soft = torch.tensor([0.1, 0.49, 0.51, 0.9], requires_grad=True)
    hard = straight_through(soft)
    assert hard.tolist() == [0.0, 0.0, 1.0, 1.0]
    hard.sum().backward()
    # Gradient passes through unchanged despite the forward rounding.
    assert soft.grad is not None
    assert torch.allclose(soft.grad, torch.ones_like(soft.grad))


def test_thresholds_are_ordered_within_each_stage() -> None:
    gates = build_gates()
    torch.nn.init.normal_(gates.base_threshold, std=0.5)
    for delta in gates.delta:
        torch.nn.init.normal_(delta, std=0.5)

    # Ordering must hold for arbitrary parameter values, not just calibrated
    # ones -- it is what makes the active set a prefix.
    for stage_thresholds in gates.thresholds():
        diffs = stage_thresholds[1:] - stage_thresholds[:-1]
        assert (diffs >= 0).all()


def test_first_group_is_always_active() -> None:
    gates = build_gates()
    calibrate_thresholds_analytic(gates)
    # Even far below the range, no stage may collapse to zero channels.
    for budget in (BUDGET_MIN * 1e-6, BUDGET_MIN, BUDGET_MAX):
        z = budget_to_z(budget, BUDGET_MIN, BUDGET_MAX)
        assert (gates.hard_widths(z) >= gates.group_size).all()


def test_widths_are_monotone_in_budget() -> None:
    gates = build_gates()
    calibrate_thresholds_analytic(gates)
    widths = [
        gates.hard_widths(budget_to_z(b, BUDGET_MIN, BUDGET_MAX)).reshape(-1)
        for b in log_spaced_budgets(200)
    ]
    for smaller, larger in zip(widths, widths[1:]):
        assert (larger >= smaller).all(), "a larger budget removed channels"


def test_active_sets_are_nested_prefixes() -> None:
    """The core guarantee: b1 < b2 implies A(b1) is contained in A(b2)."""
    from mimose.nas.extract import active_indices

    gates = build_gates()
    calibrate_thresholds_analytic(gates)
    previous: list[torch.Tensor] | None = None
    for budget in log_spaced_budgets(64):
        current = active_indices(gates, budget_to_z(budget, BUDGET_MIN, BUDGET_MAX))
        for stage, idx in enumerate(current):
            # Prefix structure: the active set is exactly arange(width).
            assert torch.equal(idx, torch.arange(idx.numel()))
            if previous is not None:
                assert idx.numel() >= previous[stage].numel()
        previous = current


def test_soft_widths_are_differentiable_wrt_thresholds() -> None:
    gates = build_gates()
    calibrate_thresholds_analytic(gates)
    z = budget_to_z(BUDGET_MAX / 4, BUDGET_MIN, BUDGET_MAX)
    gates.soft_widths(z).sum().backward()
    assert gates.base_threshold.grad is not None
    assert torch.isfinite(gates.base_threshold.grad).all()
    assert float(gates.base_threshold.grad.abs().sum()) > 0.0


def test_group_size_must_divide_every_stage_and_be_even() -> None:
    with pytest.raises(ValueError, match="not a multiple"):
        WidthGates((64, 100), budget_min=1.0, budget_max=2.0, group_size=8)
    # Divisible by 3, so the divisibility check passes and the parity check --
    # which the mask-conditioning hidden width (group_size // 2) depends on --
    # is the one that must fire.
    with pytest.raises(ValueError, match="group_size must be even"):
        WidthGates((9, 27), budget_min=1.0, budget_max=2.0, group_size=3)


# ---------------------------------------------------------------- calibration


def test_alpha_for_budget_inverts_the_parameter_count() -> None:
    for target in (2e6, 1e7, 5e7, 9e7):
        alpha = alpha_for_budget(target, MAX_FEATURES)
        widths = uniform_width_schedule(MAX_FEATURES, alpha)
        assert exact_parameters(widths) == pytest.approx(target, rel=0.05)


def test_calibration_puts_budget_versus_parameters_on_the_diagonal() -> None:
    """The milestone gate: requested budget must track realized parameters.

    Residual error is gate quantization (one group of 4 channels per stage), so
    a few percent is expected; anything larger means the thresholds are wrong.
    """
    gates = build_gates()
    calibrate_thresholds_analytic(gates)

    rows = budget_param_curve(gates, n=200)
    errors = [abs(hard - budget) / budget for budget, hard, _ in rows]
    assert sum(errors) / len(errors) < 0.02
    assert max(errors) < 0.06

    realized = [hard for _, hard, _ in rows]
    assert all(a <= b for a, b in zip(realized, realized[1:]))


def test_calibration_spans_the_requested_range() -> None:
    gates = build_gates()
    calibrate_thresholds_analytic(gates)
    rows = budget_param_curve(gates, n=64)
    assert rows[0][1] == pytest.approx(BUDGET_MIN, rel=0.05)
    assert rows[-1][1] == pytest.approx(BUDGET_MAX, rel=0.05)
