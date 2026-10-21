from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mimose.enums import LossKind, ModelKind, TrainerKind
from mimose.losses.adamm import AdaMMLoss, global_style_loss
from mimose.losses.config import build_loss_config
from mimose.models.adamm import AdaMM, input_patch_size, mask_key
from mimose.models.config import build_model_config
from mimose.utils.cli_overrides import load_yaml_config

# AdaMM has no fixed-size components, so most tests run on small volumes.
SIZE = 32


def _random_target(batch: int, num_classes: int, size: int) -> torch.Tensor:
    labels = torch.randint(0, num_classes, (batch, size, size, size))
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def test_adamm_inference_forward_returns_segmentation_shape() -> None:
    torch.manual_seed(0)
    model = AdaMM(num_cls=4).eval()
    images = torch.randn(1, 4, SIZE, SIZE, SIZE)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 4, SIZE, SIZE, SIZE)


def test_adamm_training_forward_returns_student_and_teacher_tuples() -> None:
    torch.manual_seed(0)
    model = AdaMM(num_cls=4).eval()
    model.is_training = True
    images = torch.randn(2, 4, SIZE, SIZE, SIZE)
    mask = torch.tensor([[True, False, True, True], [True, True, True, True]])

    with torch.no_grad():
        student_out, teacher_out = model(images, mask)

    bottleneck = SIZE // 8
    for branch in (student_out, teacher_out):
        uout, style, content, global_gs, logit, cls_probs = branch
        assert uout.shape == logit.shape == (2, 4, SIZE, SIZE, SIZE)
        assert style.shape == content.shape == (2, 128, bottleneck, bottleneck, bottleneck)
        assert global_gs.shape == (2, 384, bottleneck, bottleneck, bottleneck)
        assert cls_probs.shape == (2, 3)
        assert ((cls_probs >= 0) & (cls_probs <= 1)).all()


def test_adamm_teacher_has_no_missing_modality_modules() -> None:
    model = AdaMM(num_cls=4)

    assert not hasattr(model.teacher, "enc_adapters")
    assert not hasattr(model.teacher, "adagem")
    # Every non-empty, non-full combination of 4 modalities gets its own adapters.
    assert len(model.student.enc_adapters) == 14
    assert "1111" not in model.student.enc_adapters
    assert "0000" not in model.student.enc_adapters


def test_adamm_mixed_mask_batch_matches_per_sample_forward() -> None:
    torch.manual_seed(0)
    model = AdaMM(num_cls=4).eval()
    images = torch.randn(3, 4, SIZE, SIZE, SIZE)
    mask = torch.tensor([[True, False, True, True], [True, True, True, True], [False, True, False, False]])

    with torch.no_grad():
        batched = model(images, mask)
        singles = torch.cat([model(images[i : i + 1], mask[i : i + 1]) for i in range(3)])

    assert torch.allclose(batched, singles, atol=1e-5)


def test_adamm_lpgm_gating_zeroes_absent_tumor_classes() -> None:
    torch.manual_seed(0)
    model = AdaMM(num_cls=4).eval()
    images = torch.randn(1, 4, SIZE, SIZE, SIZE)
    mask = torch.tensor([[True, True, False, True]])

    model.lpgm_gating = False
    with torch.no_grad():
        ungated = model(images, mask)
        model.is_training = True
        student_out, _ = model(images, mask)
    model.is_training = False
    model.lpgm_gating = True
    with torch.no_grad():
        gated = model(images, mask)

    present = student_out[5][0] > 0.5
    assert torch.equal(gated[:, 0], ungated[:, 0])
    for tumor_class in range(3):
        expected = ungated[:, tumor_class + 1] if present[tumor_class] else torch.zeros_like(ungated[:, 0])
        assert torch.equal(gated[:, tumor_class + 1], expected)


def test_adamm_mask_key_uses_legacy_modality_order() -> None:
    images = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1)
    mask = torch.tensor([[False, True, False, True]])

    remapped_images, remapped_mask = AdaMM._remap_input_order(images, mask)

    # legacy order is [flair, t1, t1ce, t2] = external indices [2, 1, 0, 3]
    assert remapped_images.flatten().tolist() == [2.0, 1.0, 0.0, 3.0]
    assert mask_key(remapped_mask[0]) == "0101"


@pytest.mark.parametrize(
    ("images", "mask"),
    [
        (torch.zeros(1, 3, 16, 16, 16), torch.ones(1, 4, dtype=torch.bool)),
        (torch.zeros(1, 4, 16, 16, 16), torch.ones(1, 3, dtype=torch.bool)),
    ],
)
def test_adamm_rejects_invalid_modality_or_mask_shape(images: torch.Tensor, mask: torch.Tensor) -> None:
    model = AdaMM(num_cls=4).eval()

    with pytest.raises(RuntimeError, match="AdaMM expects"):
        model(images, mask)


def test_adamm_window_starts_use_50_percent_overlap() -> None:
    assert AdaMM._window_starts(input_patch_size) == [0]
    assert AdaMM._window_starts(input_patch_size + 64) == [0, 64]

    with pytest.raises(RuntimeError, match=f"at least {input_patch_size}"):
        AdaMM._window_starts(input_patch_size - 1)


def test_adamm_resolves_from_config_builders_and_enums() -> None:
    model_config = build_model_config("adamm", {"num_cls": 4})
    loss_config = build_loss_config("adamm", {"num_classes": 4, "weight_gsm": 100})

    assert model_config.model_class is AdaMM
    assert loss_config.loss_class is AdaMMLoss
    assert loss_config.kwargs == {"num_classes": 4, "weight_gsm": 100.0}
    assert TrainerKind("adamm") and ModelKind("adamm") and LossKind("adamm")


def test_adamm_training_loss_end_to_end() -> None:
    torch.manual_seed(4)
    model = AdaMM(num_cls=4)
    model.is_training = True
    images = torch.randn(2, 4, SIZE, SIZE, SIZE)
    mask = torch.tensor([[True, False, True, True], [False, False, True, False]])
    target = _random_target(batch=2, num_classes=4, size=SIZE)

    outputs = model(images, mask)
    loss_fn = AdaMMLoss(num_classes=4)
    metrics = loss_fn.training_loss(outputs, target, epoch=0)

    expected = (
        (1.0 - loss_fn.weight_mispath) * metrics["loss_dc"]
        + loss_fn.weight_mispath * metrics["loss_miss_dc"]
        + loss_fn._consistency_weight(0) * metrics["consistency"]
        + loss_fn.weight_content * metrics["content"]
        + metrics["style"]
        + loss_fn.weight_presence * (metrics["presence_full"] + metrics["presence_missing"])
        + loss_fn.weight_gsm * metrics["gsm"]
    )
    assert torch.allclose(metrics["loss"], expected, atol=1e-5, rtol=1e-5)
    assert all(torch.isfinite(value) for value in metrics.values())

    metrics["loss"].backward()
    assert model.student.cls_conv2.weight.grad is not None
    assert model.teacher.cls_conv2.weight.grad is not None
    assert model.student.adagem.pred.weight.grad is not None


def test_adamm_presence_weight_raises_dice_for_wrong_presence() -> None:
    loss_fn = AdaMMLoss(num_classes=4)
    target = _random_target(batch=1, num_classes=4, size=8)
    prediction = torch.rand(1, 4, 8, 8, 8)

    exact = loss_fn._presence_weighted_dice(prediction, target, torch.ones(1, 3))
    penalised = loss_fn._presence_weighted_dice(prediction, target, torch.full((1, 3), 2.0))

    assert penalised > exact


def test_adamm_global_style_loss_pairs_each_sample_with_itself_per_sample() -> None:
    torch.manual_seed(0)
    full = torch.randn(2, 6, 4, 4, 4)
    missing = torch.randn(2, 6, 4, 4, 4)

    per_sample = global_style_loss(full, missing, per_sample=True)
    first_only = global_style_loss(full[:1], missing[:1], per_sample=True)
    second_only = global_style_loss(full[1:], missing[1:], per_sample=True)

    assert torch.allclose(per_sample, (first_only + second_only) / 2)
    assert global_style_loss(full, full, per_sample=True) == 0


def test_adamm_consistency_weight_ramps_up_over_epochs() -> None:
    loss_fn = AdaMMLoss(num_classes=4, consistency_max_weight=10.0, consistency_rampup_epochs=20.0)

    assert loss_fn._consistency_weight(0) == pytest.approx(0.0674, abs=1e-3)
    assert loss_fn._consistency_weight(0) < loss_fn._consistency_weight(10) < loss_fn._consistency_weight(20)
    assert loss_fn._consistency_weight(20) == pytest.approx(10.0, abs=1e-6)


@pytest.mark.parametrize("config_name", ["adamm_18.yaml", "adamm_23.yaml", "adamm_25.yaml"])
def test_packaged_adamm_configs_use_adamm_training_contract(config_name: str) -> None:
    config = load_yaml_config(Path(config_name))

    assert config["trainer"] == "adamm"
    assert config["model"] == "adamm"
    assert config["loss"] == "adamm"
    assert config["custom_trainer_kwargs"]["patch_size"] == 128
    assert config["custom_trainer_kwargs"]["transform_kind"] == "imfuse"
    assert config["scheduler"] == "poly"
    assert config["optimizer"] == "adam"
    assert config["lr"] == pytest.approx(1e-4)
    assert config["num_epochs"] == 400
    assert config["batch_size"] == 1


def _make_epoch_ready_trainer(steps: int = 2):
    """An AdaMMTrainer that can run ``train_epoch`` on CPU without BaseTrainer's
    CUDA/dataset setup (same harness as tests/test_nas_pareto_trainer.py)."""
    from contextlib import nullcontext

    from mimose.training.trainers import AdaMMTrainer

    trainer = AdaMMTrainer.__new__(AdaMMTrainer)
    model = AdaMM(num_cls=4)
    trainer.model = model
    trainer.device = torch.device("cpu")
    trainer.rank = 0
    trainer.num_epochs = 10
    trainer.distributed = False
    trainer.use_grad_scaler = False
    trainer.grad_scaler = None
    trainer.amp_enabled = False
    trainer.precision = None
    trainer.model_config = None  # num_classes falls back to its default of 4
    trainer.skip_nonfinite_loss = True
    trainer.grad_clip_norm = None
    trainer.loss_fn = AdaMMLoss(num_classes=4)
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer._autocast_context = lambda: nullcontext()  # type: ignore[method-assign]
    trainer._loss_impl = lambda: trainer.loss_fn  # type: ignore[method-assign]

    torch.manual_seed(0)
    batches = [
        {
            "images": torch.randn(2, 4, SIZE, SIZE, SIZE),
            "seg": torch.randint(0, 4, (2, 1, SIZE, SIZE, SIZE)),
            "mask": torch.tensor([[True, False, True, True], [False, True, False, False]]),
        }
        for _ in range(steps)
    ]
    iterator = iter(batches)
    trainer.train_loader = batches
    trainer.iter_per_epoch = None
    trainer.debug = False
    trainer._next_train_batch = lambda: next(iterator)  # type: ignore[method-assign]
    return trainer


def test_adamm_trainer_train_epoch_accumulates_every_loss_term() -> None:
    import math

    trainer = _make_epoch_ready_trainer(steps=2)
    result = trainer.train_epoch(epoch=0)

    for key in (
        "loss",
        "loss_dc",
        "loss_miss_dc",
        "consistency",
        "content",
        "style",
        "gsm",
        "presence_full",
        "presence_missing",
    ):
        assert key in result
        assert math.isfinite(result[key])
