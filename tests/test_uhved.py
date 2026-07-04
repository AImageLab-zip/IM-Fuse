from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mimose.enums import TransformKind
from mimose.losses.config import build_loss_config
from mimose.losses.uhved import UHVEDLoss, compute_kld, dice_loss, softmax_loss
from mimose.models.config import build_model_config
from mimose.models.uhved import UHVED, input_patch_size
from mimose.training.transforms import (
    IMFuseTransformManager,
    build_transform_manager,
)
from mimose.utils.cli_overrides import load_yaml_config


def _random_probabilities(batch: int, num_classes: int, size: int) -> torch.Tensor:
    logits = torch.randn(batch, num_classes, size, size, size)
    return torch.softmax(logits, dim=1)


def _random_target(batch: int, num_classes: int, size: int) -> torch.Tensor:
    labels = torch.randint(0, num_classes, (batch, size, size, size))
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def test_uhved_inference_forward_returns_segmentation_shape() -> None:
    model = UHVED(num_cls=4).eval()
    images = torch.zeros(1, 4, 16, 16, 16)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 4, 16, 16, 16)


def test_uhved_training_forward_returns_outputs_and_posterior_params() -> None:
    model = UHVED(num_cls=4).eval()
    model.is_training = True
    images = torch.zeros(1, 4, 16, 16, 16)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        outputs, post_param = model(images, mask)

    assert outputs["seg"].shape == (1, 4, 16, 16, 16)
    for key in ("T1", "T1c", "T2", "Flair"):
        assert outputs[key].shape == (1, 1, 16, 16, 16)
    assert len(post_param) == 4
    for level in post_param:
        assert set(level["mu"]) == {"T1", "T1c", "T2", "Flair"}
        assert set(level["logvar"]) == {"T1", "T1c", "T2", "Flair"}


def test_uhved_remaps_dataset_modalities_to_legacy_order() -> None:
    images = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1)
    mask = torch.tensor([[False, True, False, True]])

    remapped_images, remapped_mask = UHVED._remap_input_order(images, mask)

    assert remapped_images.flatten().tolist() == [1.0, 0.0, 3.0, 2.0]
    assert remapped_mask.tolist() == [[True, False, True, False]]


def test_uhved_window_starts_use_50_percent_overlap() -> None:
    assert UHVED._window_starts(input_patch_size) == [0]
    assert UHVED._window_starts(input_patch_size + 56) == [0, 56]

    with pytest.raises(RuntimeError, match=f"at least {input_patch_size}"):
        UHVED._window_starts(input_patch_size - 1)


@pytest.mark.parametrize(
    ("images", "mask"),
    [
        (torch.zeros(1, 3, 16, 16, 16), torch.ones(1, 4, dtype=torch.bool)),
        (torch.zeros(1, 4, 16, 16, 16), torch.ones(1, 3, dtype=torch.bool)),
    ],
)
def test_uhved_rejects_invalid_modality_or_mask_shape(
    images: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    model = UHVED(num_cls=4).eval()

    with pytest.raises(RuntimeError, match="UHVED expects"):
        model(images, mask)


def test_uhved_resolves_from_model_config_builder() -> None:
    model_config = build_model_config("uhved", {"num_cls": 4})

    assert model_config.model_class is UHVED
    assert model_config.kwargs == {"num_cls": 4}


def test_uhved_loss_resolves_from_loss_config_builder() -> None:
    loss_config = build_loss_config("uhved", {"num_classes": 4})

    assert loss_config.loss_class is UHVEDLoss
    assert loss_config.kwargs == {"num_classes": 4}


def test_uhved_transform_manager_resolves_from_kind() -> None:
    transform_manager = build_transform_manager(TransformKind.IMFUSE, crop_size=(112, 112, 112))

    assert isinstance(transform_manager, IMFuseTransformManager)
    assert transform_manager.crop_size == (112, 112, 112)


def test_uhved_dice_and_softmax_losses_are_zero_for_perfect_prediction() -> None:
    target = _random_target(batch=1, num_classes=4, size=6)

    assert dice_loss(target, target, num_cls=4).item() == pytest.approx(0.0, abs=1e-5)
    assert softmax_loss(target, target, num_cls=4).item() == pytest.approx(0.0, abs=1e-3)


def test_uhved_compute_kld_is_finite_and_nonnegative() -> None:
    torch.manual_seed(5)
    shape = (2, 4, 4, 4, 4)
    means = {mod: torch.randn(shape) * 0.1 for mod in ("T1", "T1c", "T2", "Flair")}
    logvars = {mod: torch.randn(shape) * 0.1 for mod in ("T1", "T1c", "T2", "Flair")}
    mask = torch.ones(2, 4, dtype=torch.bool)

    inter_kld, prior_kld = compute_kld(means, logvars, mask)

    assert torch.isfinite(inter_kld)
    assert torch.isfinite(prior_kld)
    assert inter_kld.item() >= -1e-4
    assert prior_kld.item() >= -1e-4


def test_uhved_training_loss_end_to_end() -> None:
    torch.manual_seed(3)
    model = UHVED(num_cls=4)
    model.is_training = True
    images = torch.randn(1, 4, 32, 32, 32)
    mask = torch.ones(1, 4, dtype=torch.bool)
    target = _random_target(batch=1, num_classes=4, size=32)

    outputs, post_param = model(images, mask)
    loss_fn = UHVEDLoss(num_classes=4)
    metrics = loss_fn.training_loss(outputs, post_param, target, mask)

    expected = metrics["cross"] + metrics["dice"] + 0.1 * metrics["kld"] + 0.1 * metrics["recon"]
    assert torch.allclose(metrics["loss"], expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("config_name", ["uhved_18.yaml", "uhved_23.yaml", "uhved_25.yaml"])
def test_packaged_uhved_configs_use_imfuse_training_contract(config_name: str) -> None:
    config = load_yaml_config(Path(config_name))

    assert config["trainer"] == "uhved"
    assert config["model"] == "uhved"
    assert config["loss"] == "uhved"
    assert config["custom_trainer_kwargs"]["patch_size"] == 112
    assert config["custom_trainer_kwargs"]["transform_kind"] == "imfuse"
    assert config["scheduler"] == "plateau"
