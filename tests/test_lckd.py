from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mimose.enums import TransformKind
from mimose.losses.config import build_loss_config
from mimose.losses.lckd import LCKDLoss
from mimose.models.config import build_model_config
from mimose.models.lckd import LCKD, input_patch_size
from mimose.training.transforms import (
    IMFuseTransformManager,
    build_transform_manager,
)
from mimose.utils.cli_overrides import load_yaml_config


def _random_target(batch: int, num_classes: int, size: int) -> torch.Tensor:
    labels = torch.randint(0, num_classes, (batch, size, size, size))
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def test_lckd_inference_forward_returns_segmentation_shape() -> None:
    torch.manual_seed(0)
    model = LCKD(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert torch.isfinite(output).all()


def test_lckd_training_forward_returns_prediction_and_kd_loss() -> None:
    torch.manual_seed(0)
    model = LCKD(num_cls=4)
    model.is_training = True
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.tensor([[True, False, True, True]])

    prediction, kd_loss = model(images, mask)

    assert prediction.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert kd_loss.ndim == 0
    assert torch.isfinite(kd_loss)


def test_lckd_handles_missing_modalities() -> None:
    torch.manual_seed(0)
    model = LCKD(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.tensor([[True, False, True, False]])

    with torch.no_grad():
        output = model(images, mask)

    assert output.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert torch.isfinite(output).all()


def test_lckd_remaps_dataset_modalities_to_legacy_order() -> None:
    images = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1)
    mask = torch.tensor([[False, True, False, True]])

    remapped_images, remapped_mask = LCKD._remap_input_order(images, mask)

    # legacy order is [flair, t1, t1ce, t2] = external indices [2, 1, 0, 3]
    assert remapped_images.flatten().tolist() == [2.0, 1.0, 0.0, 3.0]
    assert remapped_mask.tolist() == [[False, True, False, True]]


def test_lckd_window_starts_use_50_percent_overlap() -> None:
    assert LCKD._window_starts(input_patch_size) == [0]
    assert LCKD._window_starts(input_patch_size + 64) == [0, 64]

    with pytest.raises(RuntimeError, match=f"at least {input_patch_size}"):
        LCKD._window_starts(input_patch_size - 1)


@pytest.mark.parametrize(
    ("images", "mask"),
    [
        (torch.zeros(1, 3, 16, 16, 16), torch.ones(1, 4, dtype=torch.bool)),
        (torch.zeros(1, 4, 16, 16, 16), torch.ones(1, 3, dtype=torch.bool)),
    ],
)
def test_lckd_rejects_invalid_modality_or_mask_shape(
    images: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    model = LCKD(num_cls=4).eval()

    with pytest.raises(RuntimeError, match="LCKD expects"):
        model(images, mask)


def test_lckd_resolves_from_model_config_builder() -> None:
    model_config = build_model_config("lckd", {"num_cls": 4})

    assert model_config.model_class is LCKD
    assert model_config.kwargs == {"num_cls": 4}


def test_lckd_loss_resolves_from_loss_config_builder() -> None:
    loss_config = build_loss_config("lckd", {"num_classes": 4})

    assert loss_config.loss_class is LCKDLoss
    assert loss_config.kwargs == {"num_classes": 4}


def test_lckd_transform_manager_resolves_from_kind() -> None:
    transform_manager = build_transform_manager(TransformKind.IMFUSE, crop_size=(128, 128, 128))

    assert isinstance(transform_manager, IMFuseTransformManager)
    assert transform_manager.crop_size == (128, 128, 128)


def test_lckd_training_loss_end_to_end() -> None:
    torch.manual_seed(4)
    model = LCKD(num_cls=4)
    model.is_training = True
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.tensor([[True, False, True, True]])
    target = _random_target(batch=1, num_classes=4, size=input_patch_size)

    outputs = model(images, mask)
    loss_fn = LCKDLoss(num_classes=4)
    metrics = loss_fn.training_loss(outputs, target)

    expected = metrics["cross"] + metrics["dice"] + loss_fn.kd_weight * metrics["kd"]
    assert torch.allclose(metrics["loss"], expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("config_name", ["lckd_18.yaml", "lckd_23.yaml", "lckd_25.yaml"])
def test_packaged_lckd_configs_use_lckd_training_contract(config_name: str) -> None:
    config = load_yaml_config(Path(config_name))

    assert config["trainer"] == "lckd"
    assert config["model"] == "lckd"
    assert config["loss"] == "lckd"
    assert config["custom_trainer_kwargs"]["patch_size"] == 128
    assert config["custom_trainer_kwargs"]["transform_kind"] == "imfuse"
    assert config["scheduler"] == "poly"
    assert config["optimizer"] == "sgd"
