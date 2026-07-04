from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mimose.enums import TransformKind
from mimose.losses.config import build_loss_config
from mimose.losses.mam import MaMLoss
from mimose.models.config import build_model_config
from mimose.models.mam import MaM, input_patch_size
from mimose.training.transforms import (
    IMFuseTransformManager,
    build_transform_manager,
)
from mimose.utils.cli_overrides import load_yaml_config


def _random_target(batch: int, num_classes: int, size: int) -> torch.Tensor:
    labels = torch.randint(0, num_classes, (batch, size, size, size))
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def test_mam_inference_forward_returns_segmentation_shape() -> None:
    torch.manual_seed(0)
    model = MaM(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)


def test_mam_training_forward_returns_bottleneck_and_ground_truth() -> None:
    torch.manual_seed(0)
    model = MaM(num_cls=4).eval()
    model.is_training = True
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        seg_pred, bottleneck, ground_truth = model(images, mask)

    assert seg_pred.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert bottleneck.shape == (1, 4, 128, 4, 4, 4)
    assert ground_truth.shape == bottleneck.shape
    assert not ground_truth.requires_grad


def test_mam_handles_missing_modalities_and_heterogeneous_batch_masks() -> None:
    torch.manual_seed(0)
    model = MaM(num_cls=4).eval()
    images = torch.randn(2, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.tensor([[True, False, True, False], [False, True, False, True]])

    with torch.no_grad():
        output = model(images, mask)

    assert output.shape == (2, 4, input_patch_size, input_patch_size, input_patch_size)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("images", "mask"),
    [
        (torch.zeros(1, 3, input_patch_size, input_patch_size, input_patch_size), torch.ones(1, 4, dtype=torch.bool)),
        (torch.zeros(1, 4, input_patch_size, input_patch_size, input_patch_size), torch.ones(1, 3, dtype=torch.bool)),
    ],
)
def test_mam_rejects_invalid_modality_or_mask_shape(
    images: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    model = MaM(num_cls=4).eval()

    with pytest.raises(RuntimeError, match="MaM expects"):
        model(images, mask)


def test_mam_window_starts_use_50_percent_overlap() -> None:
    assert MaM._window_starts(input_patch_size) == [0]
    assert MaM._window_starts(input_patch_size + 64) == [0, 64]

    with pytest.raises(RuntimeError, match=f"at least {input_patch_size}"):
        MaM._window_starts(input_patch_size - 1)


def test_mam_resolves_from_model_config_builder() -> None:
    model_config = build_model_config("mam", {"num_cls": 4})

    assert model_config.model_class is MaM
    assert model_config.kwargs == {"num_cls": 4}


def test_mam_loss_resolves_from_loss_config_builder() -> None:
    loss_config = build_loss_config("mam", {"num_classes": 4})

    assert loss_config.loss_class is MaMLoss
    assert loss_config.kwargs == {"num_classes": 4}


def test_mam_transform_manager_resolves_from_kind() -> None:
    transform_manager = build_transform_manager(TransformKind.IMFUSE, crop_size=(128, 128, 128))

    assert isinstance(transform_manager, IMFuseTransformManager)
    assert transform_manager.crop_size == (128, 128, 128)


def test_mam_reconstruction_loss_only_covers_missing_modalities() -> None:
    torch.manual_seed(4)
    model = MaM(num_cls=4)
    model.is_training = True
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask_all_present = torch.ones(1, 4, dtype=torch.bool)
    target = _random_target(batch=1, num_classes=4, size=input_patch_size)

    outputs = model(images, mask_all_present)
    loss_fn = MaMLoss(num_classes=4)
    metrics = loss_fn.training_loss(outputs, target, mask_all_present)

    # no missing modalities -> reconstruction term is exactly zero
    assert metrics["recon"].item() == pytest.approx(0.0, abs=1e-8)
    expected = metrics["cross"] + metrics["dice"]
    assert torch.allclose(metrics["loss"], expected, atol=1e-6, rtol=1e-6)


def test_mam_training_loss_end_to_end_with_missing_modality() -> None:
    torch.manual_seed(4)
    model = MaM(num_cls=4)
    model.is_training = True
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.tensor([[True, True, False, True]])
    target = _random_target(batch=1, num_classes=4, size=input_patch_size)

    outputs = model(images, mask)
    loss_fn = MaMLoss(num_classes=4)
    metrics = loss_fn.training_loss(outputs, target, mask)

    expected = metrics["cross"] + metrics["dice"] + metrics["recon"]
    assert torch.allclose(metrics["loss"], expected, atol=1e-6, rtol=1e-6)
    assert metrics["recon"].item() > 0.0


@pytest.mark.parametrize("config_name", ["mam_18.yaml", "mam_23.yaml", "mam_25.yaml"])
def test_packaged_mam_configs_use_mam_training_contract(config_name: str) -> None:
    config = load_yaml_config(Path(config_name))

    assert config["trainer"] == "mam"
    assert config["model"] == "mam"
    assert config["loss"] == "mam"
    assert config["custom_trainer_kwargs"]["patch_size"] == 128
    assert config["custom_trainer_kwargs"]["transform_kind"] == "imfuse"
    assert config["scheduler"] == "poly"
    assert config["optimizer"] == "sgd"
    assert config["nesterov"] is True
