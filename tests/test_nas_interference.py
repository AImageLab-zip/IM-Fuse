from __future__ import annotations

import math
from contextlib import nullcontext

import pytest
import torch

from mimose.losses.tiny_mimosa import TinyMimosaLoss
from mimose.models.elastic_unet import ElasticUnet
from mimose.nas.interference import (
    InterferenceResult,
    _cosine,
    _masked_cosine,
    _shared_parameters,
    measure_gradient_interference,
)
from mimose.nas.report import summarize_interference, write_interference_report
from mimose.training.trainers import ParetoTrainer

INPUT_SHAPE = (24, 24, 24)


def make_trainer() -> ParetoTrainer:
    """A ParetoTrainer stub with just enough state for gradient measurement."""
    trainer = ParetoTrainer.__new__(ParetoTrainer)
    model = ElasticUnet(
        features_per_stage=(8, 16, 32, 64),
        input_shape=INPUT_SHAPE,
        group_size=2,
    )
    model.is_training = True

    trainer.model = model
    trainer.device = torch.device("cpu")
    trainer.rank = 0
    trainer.model_config = None
    trainer.distributed = False
    trainer.amp_enabled = False
    trainer.precision = None
    trainer.loss_fn = TinyMimosaLoss(num_classes=4)
    trainer._autocast_context = lambda: nullcontext()  # type: ignore[method-assign]
    trainer._loss_impl = lambda: trainer.loss_fn  # type: ignore[method-assign]
    return trainer


def attach_batches(trainer: ParetoTrainer, count: int) -> list[dict]:
    torch.manual_seed(0)
    batches = [
        {
            "images": torch.randn(1, 4, *INPUT_SHAPE),
            "seg": torch.randint(0, 4, (1, 1, *INPUT_SHAPE)),
            "mask": torch.ones(1, 4, dtype=torch.bool),
        }
        for _ in range(count)
    ]
    iterator = iter(batches)
    trainer._next_train_batch = lambda: next(iterator)  # type: ignore[method-assign]
    return batches


def log_spaced(model: ElasticUnet, count: int) -> list[float]:
    low, high = model.gates.budget_min, model.gates.budget_max
    return [low * (high / low) ** (i / (count - 1)) for i in range(count)]


# ------------------------------------------------------------------ primitives


def test_cosine_matches_torch() -> None:
    a = torch.randn(500)
    b = torch.randn(500)
    expected = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    assert _cosine(a, b) == pytest.approx(expected, abs=1e-6)


def test_cosine_of_identical_vectors_is_one() -> None:
    a = torch.randn(100)
    assert _cosine(a, a) == pytest.approx(1.0, abs=1e-6)
    assert _cosine(a, -a) == pytest.approx(-1.0, abs=1e-6)


def test_cosine_of_a_zero_vector_is_nan_not_a_crash() -> None:
    assert math.isnan(_cosine(torch.zeros(10), torch.randn(10)))


def test_masked_cosine_ignores_coordinates_dead_in_either_vector() -> None:
    """Support mismatch must not be mistaken for disagreement.

    ``a`` and ``b`` agree perfectly on the two coordinates they share, but ``b``
    is zero everywhere else. The full cosine is dragged down by that; the masked
    one should see the agreement.
    """
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([1.0, 2.0, 0.0, 0.0])

    masked, fraction = _masked_cosine(a, b)
    assert masked == pytest.approx(1.0, abs=1e-6)
    assert fraction == pytest.approx(0.5)
    assert _cosine(a, b) < masked


def test_masked_cosine_reports_zero_overlap_without_dividing_by_zero() -> None:
    a = torch.tensor([1.0, 2.0, 0.0, 0.0])
    b = torch.tensor([0.0, 0.0, 3.0, 4.0])
    masked, fraction = _masked_cosine(a, b)
    assert math.isnan(masked)
    assert fraction == pytest.approx(0.0)


def test_shared_parameters_excludes_the_gates() -> None:
    """Gate gradients describe the conditioning, not conflict over weights."""
    trainer = make_trainer()
    model = trainer.model
    shared = _shared_parameters(model)

    shared_ids = {id(p) for p in shared}
    for gate_param in model.gates.parameters():
        assert id(gate_param) not in shared_ids
    assert len(shared) == len(list(model.parameters())) - len(
        list(model.gates.parameters())
    )


# ----------------------------------------------------------------- measurement


def test_measure_produces_symmetric_matrices_with_unit_diagonal() -> None:
    trainer = make_trainer()
    attach_batches(trainer, 4)
    budgets = log_spaced(trainer.model, 3)

    result = measure_gradient_interference(trainer, budgets, n_batches=3)

    n = len(budgets)
    for matrix in (result.cosine_full, result.cosine_overlap, result.overlap_fraction):
        assert len(matrix) == n
        for i in range(n):
            assert matrix[i][i] == pytest.approx(1.0)
            for j in range(n):
                assert matrix[i][j] == pytest.approx(matrix[j][i], abs=1e-9)


def test_identical_budgets_on_the_same_batch_give_cosine_one() -> None:
    """A pair of identical architectures is the measurement's own control."""
    trainer = make_trainer()
    attach_batches(trainer, 3)
    budget = log_spaced(trainer.model, 3)[1]

    result = measure_gradient_interference(trainer, [budget, budget], n_batches=2)
    assert result.cosine_full[0][1] == pytest.approx(1.0, abs=1e-4)
    assert result.cosine_overlap[0][1] == pytest.approx(1.0, abs=1e-4)
    assert result.overlap_fraction[0][1] == pytest.approx(1.0, abs=1e-6)


def test_noise_floor_is_below_one_and_reported_per_budget() -> None:
    """Different batches disagree even for one architecture -- that's the floor.

    It is per-budget on purpose: wider subnetworks have noisier gradients, so a
    single global floor would misjudge pairs at opposite ends of the range.
    """
    trainer = make_trainer()
    attach_batches(trainer, 5)
    budgets = log_spaced(trainer.model, 3)

    result = measure_gradient_interference(trainer, budgets, n_batches=4)
    assert len(result.noise_floor) == len(budgets)
    for value in result.noise_floor:
        assert -1.0 <= value < 1.0


def test_overlap_fraction_shrinks_as_budgets_separate() -> None:
    """Distant budgets barely share weights -- the real constraint, arguably."""
    trainer = make_trainer()
    attach_batches(trainer, 3)
    budgets = log_spaced(trainer.model, 4)

    result = measure_gradient_interference(trainer, budgets, n_batches=2)
    adjacent = result.overlap_fraction[0][1]
    extreme = result.overlap_fraction[0][-1]
    assert extreme < adjacent


def test_measurement_leaves_no_gradients_behind() -> None:
    trainer = make_trainer()
    attach_batches(trainer, 3)
    measure_gradient_interference(trainer, log_spaced(trainer.model, 3), n_batches=2)
    assert all(p.grad is None for p in trainer.model.parameters())


def test_measurement_restores_the_training_flag() -> None:
    trainer = make_trainer()
    trainer.model.is_training = False
    attach_batches(trainer, 3)
    measure_gradient_interference(trainer, log_spaced(trainer.model, 3), n_batches=2)
    assert trainer.model.is_training is False


def test_measure_rejects_degenerate_arguments() -> None:
    trainer = make_trainer()
    attach_batches(trainer, 3)
    with pytest.raises(ValueError, match="at least two budgets"):
        measure_gradient_interference(trainer, [1e6], n_batches=2)
    with pytest.raises(ValueError, match="at least two batches"):
        measure_gradient_interference(trainer, [1e6, 2e6], n_batches=1)


# --------------------------------------------------------------------- report


def build_result() -> InterferenceResult:
    trainer = make_trainer()
    attach_batches(trainer, 5)
    return measure_gradient_interference(
        trainer, log_spaced(trainer.model, 4), n_batches=4
    )


def test_pairs_covers_the_upper_triangle_with_relative_values() -> None:
    result = build_result()
    rows = result.pairs()
    n = len(result.budgets)
    assert len(rows) == n * (n - 1) // 2

    for row in rows:
        assert row["budget_i"] < row["budget_j"]
        assert row["log_distance"] > 0
        # relative_* is the interpretable quantity: cosine against the floor.
        expected = row["cosine_overlap"] / row["noise_floor"]
        assert row["relative_overlap"] == pytest.approx(expected)


def test_summary_reports_the_decay_correlation() -> None:
    summary = summarize_interference(build_result())
    for key in (
        "corr_distance_vs_full",
        "corr_distance_vs_overlap",
        "corr_distance_vs_overlap_fraction",
        "min_relative_overlap",
        "n_pairs",
    ):
        assert key in summary
    # Support necessarily diverges with distance, so this correlation must be
    # negative regardless of whether real interference exists.
    assert summary["corr_distance_vs_overlap_fraction"] < 0


def test_write_report_emits_all_artifacts(tmp_path) -> None:
    result = build_result()
    directory = write_interference_report(result, tmp_path / "interference")

    for name in (
        "interference_pairs.csv",
        "interference_matrices.csv",
        "interference_summary.csv",
        "interference_heatmaps.png",
        "interference_vs_distance.png",
    ):
        assert (directory / name).is_file(), name

    # The tidy CSV must be re-readable without rerunning the GPU measurement.
    import csv

    with (directory / "interference_pairs.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result.pairs())
    assert "relative_overlap" in rows[0]
