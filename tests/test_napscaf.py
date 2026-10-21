from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path

import pytest
import torch

from mimose.enums import LossKind, ModelKind, TrainerKind
from mimose.losses.config import build_loss_config
from mimose.losses.napscaf import NAPSCAFLoss
from mimose.models.config import build_model_config
from mimose.models.napscaf import NAPSCAF
from mimose.utils.cli_overrides import load_yaml_config


def _model() -> NAPSCAF:
    # Skip the ImageNet download: the prior branch's weights do not matter for these checks.
    return NAPSCAF(num_cls=4, pretrained_prior=False)


def test_napscaf_matches_paper_parameter_count() -> None:
    params = sum(p.numel() for p in _model().parameters())

    # README: 58.3M parameters (DenseNet121 classifier excluded).
    assert params / 1e6 == pytest.approx(58.3, abs=0.05)


def test_napscaf_training_forward_returns_slice_logits() -> None:
    torch.manual_seed(0)
    model = _model().train()
    model.is_training = True
    slices = torch.randn(3, 4, 64, 64)
    mask = torch.tensor([[True, False, True, True]] * 3)

    logits = model(slices, mask)

    assert logits.shape == (3, 4, 64, 64)


def test_napscaf_volume_inference_handles_non_multiple_of_32_sizes() -> None:
    torch.manual_seed(0)
    model = _model().eval()
    volume = torch.randn(2, 4, 70, 90, 5)
    mask = torch.tensor([[True, True, True, True], [False, False, True, False]])

    with torch.no_grad():
        probabilities = model.predict(volume, mask)

    assert probabilities.shape == (2, 4, 70, 90, 5)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2, 70, 90, 5), atol=1e-5)


def test_napscaf_prior_branch_receives_no_gradient() -> None:
    model = _model().train()
    model.is_training = True

    model(torch.randn(2, 4, 64, 64), torch.ones(2, 4, dtype=torch.bool)).mean().backward()

    assert all(p.grad is None for p in model.net.nap_path.vae.parameters())
    assert model.net.convout.weight.grad is not None


def test_napscaf_zero_fills_missing_modalities() -> None:
    torch.manual_seed(0)
    model = _model().eval()
    model.is_training = True
    slices = torch.randn(1, 4, 64, 64)
    # External order [t1c, t1n, t2f, t2w]; dropping t1c must equal zeroing it by hand.
    mask = torch.tensor([[False, True, True, True]])
    zeroed = slices.clone()
    zeroed[:, 0] = 0

    torch.manual_seed(1)
    with torch.no_grad():
        masked = model(slices, mask)
    torch.manual_seed(1)
    with torch.no_grad():
        manual = model(zeroed, torch.ones(1, 4, dtype=torch.bool))

    assert torch.allclose(masked, manual, atol=1e-5)


@pytest.mark.parametrize(
    ("images", "mask"),
    [
        (torch.zeros(1, 3, 32, 32, 2), torch.ones(1, 4, dtype=torch.bool)),
        (torch.zeros(1, 4, 32, 32, 2), torch.ones(1, 3, dtype=torch.bool)),
    ],
)
def test_napscaf_rejects_invalid_modality_or_mask_shape(images: torch.Tensor, mask: torch.Tensor) -> None:
    with pytest.raises(RuntimeError, match="NAPSCAF expects"):
        _model().eval()(images, mask)


def test_napscaf_resolves_from_config_builders_and_enums() -> None:
    model_config = build_model_config("napscaf", {"num_cls": 4})
    loss_config = build_loss_config("napscaf", {"num_classes": 4, "ce_weight": 0.5})

    assert model_config.model_class is NAPSCAF
    assert loss_config.loss_class is NAPSCAFLoss
    assert loss_config.kwargs == {"num_classes": 4, "ce_weight": 0.5}
    assert TrainerKind("napscaf") and ModelKind("napscaf") and LossKind("napscaf")


def test_napscaf_loss_combines_ce_and_dice() -> None:
    torch.manual_seed(0)
    loss_fn = NAPSCAFLoss(num_classes=4)
    logits = torch.randn(2, 4, 16, 16)
    labels = torch.randint(0, 4, (2, 16, 16))

    metrics = loss_fn.training_loss(logits, labels)

    assert torch.allclose(metrics["loss"], 0.5 * metrics["ce"] + 0.5 * metrics["dice"])
    assert torch.allclose(metrics["ce"], torch.nn.functional.cross_entropy(logits, labels))
    one_hot = torch.nn.functional.one_hot(labels, 4).movedim(-1, 1).float()
    assert loss_fn.training_loss(one_hot * 50.0, labels)["dice"] < 1e-4


def _make_epoch_ready_trainer(steps: int = 2, slices_per_patch: int = 4):
    from mimose.training.trainers import NAPSCAFTrainer

    trainer = NAPSCAFTrainer.__new__(NAPSCAFTrainer)
    model = _model()
    trainer.model = model
    trainer.device = torch.device("cpu")
    trainer.rank = 0
    trainer.num_epochs = 10
    trainer.distributed = False
    trainer.use_grad_scaler = False
    trainer.grad_scaler = None
    trainer.amp_enabled = False
    trainer.precision = None
    trainer.model_config = None
    trainer.skip_nonfinite_loss = True
    trainer.grad_clip_norm = None
    trainer.slices_per_patch = slices_per_patch
    trainer.slice_sampling = "tumor"
    trainer.loss_fn = NAPSCAFLoss(num_classes=4)
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer._autocast_context = lambda: nullcontext()  # type: ignore[method-assign]
    trainer._loss_impl = lambda: trainer.loss_fn  # type: ignore[method-assign]

    torch.manual_seed(0)
    batches = [
        {
            "images": torch.randn(2, 4, 32, 32, 8),
            "seg": torch.randint(0, 4, (2, 1, 32, 32, 8)),
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


def test_napscaf_trainer_prefers_tumor_slices() -> None:
    trainer = _make_epoch_ready_trainer(slices_per_patch=3)
    seg = torch.zeros(8, 8, 10, dtype=torch.long)
    seg[2:4, 2:4, [1, 6]] = 2

    for _ in range(5):
        indices = trainer._sample_slice_indices(seg)
        assert len(indices) == 3
        assert {1, 6} <= set(indices.tolist())


def test_napscaf_trainer_slices_patches_with_their_masks() -> None:
    trainer = _make_epoch_ready_trainer(slices_per_patch=4)
    images = torch.randn(2, 4, 32, 32, 8)
    seg = torch.randint(0, 4, (2, 1, 32, 32, 8))
    mask = torch.tensor([[True, False, True, True], [False, True, False, False]])

    slices, labels, slice_mask = trainer._slice_batch(images, seg, mask)

    assert slices.shape == (8, 4, 32, 32)
    assert labels.shape == (8, 32, 32)
    assert torch.equal(slice_mask[:4], mask[0].expand(4, -1))
    assert torch.equal(slice_mask[4:], mask[1].expand(4, -1))


def test_napscaf_trainer_train_epoch_reports_loss_terms() -> None:
    trainer = _make_epoch_ready_trainer(steps=2)
    result = trainer.train_epoch(epoch=0)

    for key in ("loss", "ce", "dice"):
        assert key in result
        assert math.isfinite(result[key])


@pytest.mark.parametrize("config_name", ["napscaf_18.yaml", "napscaf_23.yaml", "napscaf_25.yaml"])
def test_packaged_napscaf_configs_use_napscaf_training_contract(config_name: str) -> None:
    config = load_yaml_config(Path(config_name))

    assert config["trainer"] == "napscaf"
    assert config["model"] == "napscaf"
    assert config["loss"] == "napscaf"
    assert config["custom_trainer_kwargs"]["patch_size"] == 128
    assert config["custom_trainer_kwargs"]["slices_per_patch"] == 16
    assert config["optimizer"] == "adamw"
    assert tuple(config["betas"]) == (0.9, 0.99)
    assert config["scheduler"] == "step"
    assert config["step_gamma"] == 1.0
    assert config["lr"] == pytest.approx(1e-4)
    assert config["weight_decay"] == pytest.approx(1e-4)
