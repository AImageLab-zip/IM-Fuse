from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mimose.enums import TransformKind
from mimose.losses.a2fseg import A2FSegLoss
from mimose.losses.config import build_loss_config
from mimose.models.a2fseg import A2FSeg, input_patch_size
from mimose.training.trainers import A2FSegTrainer
from mimose.training.transforms import (
    IMFuseTransformManager,
    build_transform_manager,
)
from mimose.utils.cli_overrides import load_yaml_config


def test_a2fseg_inference_forward_returns_segmentation_shape() -> None:
    model = A2FSeg(num_cls=4).eval()
    images = torch.zeros(1, 4, 16, 16, 16)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 4, 16, 16, 16)
    assert torch.allclose(output.sum(dim=1), torch.ones_like(output.sum(dim=1)), atol=1e-4)


def test_a2fseg_training_forward_returns_fuse_sep_and_fusion_ds_predictions() -> None:
    model = A2FSeg(num_cls=4).eval()
    model.is_training = True
    images = torch.zeros(1, 4, 16, 16, 16)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        fuse_pred, sep_preds, fusion_ds_preds = model(images, mask)

    assert fuse_pred.shape == (1, 4, 16, 16, 16)
    assert len(sep_preds) == 4
    assert len(fusion_ds_preds) == 3
    assert all(len(preds) == 3 for preds in sep_preds)
    assert all(pred.shape == (1, 4, 16, 16, 16) for preds in sep_preds for pred in preds)
    assert all(pred.shape == (1, 4, 16, 16, 16) for pred in fusion_ds_preds)


def test_a2fseg_handles_dropped_modalities() -> None:
    model = A2FSeg(num_cls=4).eval()
    images = torch.randn(2, 4, 16, 16, 16)
    mask = torch.ones(2, 4, dtype=torch.bool)
    mask[0, 1] = False
    mask[1, 2] = False

    with torch.no_grad():
        output = model(images, mask)

    assert output.shape == (2, 4, 16, 16, 16)
    assert torch.isfinite(output).all()


def test_a2fseg_remaps_dataset_modalities_to_legacy_order() -> None:
    images = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1)
    mask = torch.tensor([[False, True, False, True]])

    remapped_images, remapped_mask = A2FSeg._remap_input_order(images, mask)

    assert remapped_images.flatten().tolist() == [2.0, 0.0, 1.0, 3.0]
    assert remapped_mask.tolist() == [[False, False, True, True]]


def test_a2fseg_window_starts_use_50_percent_overlap() -> None:
    assert A2FSeg._window_starts(input_patch_size) == [0]
    assert A2FSeg._window_starts(input_patch_size + 40) == [0, 40]
    assert A2FSeg._window_starts(input_patch_size + 62) == [0, 40, 62]

    with pytest.raises(RuntimeError, match="at least 80"):
        A2FSeg._window_starts(input_patch_size - 1)


@pytest.mark.parametrize(
    ("images", "mask"),
    [
        (torch.zeros(1, 3, 16, 16, 16), torch.ones(1, 4, dtype=torch.bool)),
        (torch.zeros(1, 4, 16, 16, 16), torch.ones(1, 3, dtype=torch.bool)),
    ],
)
def test_a2fseg_rejects_invalid_modality_or_mask_shape(
    images: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    model = A2FSeg(num_cls=4).eval()

    with pytest.raises(RuntimeError, match="A2FSeg expects"):
        model(images, mask)


def test_a2fseg_loss_resolves_from_loss_config_builder() -> None:
    loss_config = build_loss_config("a2fseg", {"num_classes": 4})

    assert loss_config.loss_class is A2FSegLoss
    assert loss_config.kwargs == {"num_classes": 4}


def test_a2fseg_loss_training_loss_masks_absent_modalities() -> None:
    model = A2FSeg(num_cls=4).eval()
    model.is_training = True
    images = torch.randn(2, 4, 16, 16, 16)
    mask = torch.ones(2, 4, dtype=torch.bool)
    mask[0, 1] = False
    target = torch.nn.functional.one_hot(
        torch.randint(0, 4, (2, 16, 16, 16)), num_classes=4
    ).permute(0, 4, 1, 2, 3).float()

    with torch.no_grad():
        outputs = model(images, mask)

    loss_fn = A2FSegLoss(num_cls=4)
    metrics = loss_fn.training_loss(outputs, target, mask=mask)

    assert torch.isfinite(metrics["loss"])
    assert metrics["loss"].ndim == 0


def test_a2fseg_transform_manager_resolves_from_kind() -> None:
    transform_manager = build_transform_manager(TransformKind.IMFUSE, crop_size=(80, 80, 80))

    assert isinstance(transform_manager, IMFuseTransformManager)
    assert transform_manager.crop_size == (80, 80, 80)


def test_a2fseg_trainer_is_a_base_trainer_subclass() -> None:
    from mimose.training.trainers.base_trainer import BaseTrainer

    assert issubclass(A2FSegTrainer, BaseTrainer)


@pytest.mark.parametrize("config_name", ["a2fseg_18.yaml", "a2fseg_23.yaml", "a2fseg_25.yaml"])
def test_packaged_a2fseg_configs_use_expected_training_contract(config_name: str) -> None:
    config = load_yaml_config(Path(config_name))

    assert config["trainer"] == "a2fseg"
    assert config["model"] == "a2fseg"
    assert config["loss"] == "a2fseg"
    assert config["lr"] == 0.0004
    assert config["weight_decay"] == 0.0001
    assert config["fp16"] is True
    assert config["seed"] == 999
    assert config["crop_min_size"] == (128, 128, 128)
    assert config["custom_trainer_kwargs"]["patch_size"] == 80
    assert config["custom_trainer_kwargs"]["transform_kind"] == "imfuse"
