from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mimose.enums import TransformKind
from mimose.losses.config import build_loss_config
from mimose.losses.srmnet import SRMNetLoss
from mimose.models.config import build_model_config
from mimose.models.srmnet import SRMNet, input_patch_size
from mimose.training.transforms import (
    IMFuseTransformManager,
    build_transform_manager,
)
from mimose.utils.cli_overrides import load_yaml_config


def _random_target(batch: int, num_classes: int, size: int) -> torch.Tensor:
    labels = torch.randint(0, num_classes, (batch, size, size, size))
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def test_srmnet_inference_forward_returns_segmentation_shape() -> None:
    torch.manual_seed(0)
    model = SRMNet(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)


def test_srmnet_training_forward_returns_pred_deep_supervision_and_recons() -> None:
    torch.manual_seed(0)
    model = SRMNet(num_cls=4).eval()
    model.is_training = True
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        pred, preds, recs, remapped_images = model(images, mask)

    expected_seg_shape = (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert pred.shape == expected_seg_shape
    assert len(preds) == 4
    assert all(p.shape == expected_seg_shape for p in preds)
    assert len(recs) == 4
    assert all(r.shape == (1, 1, input_patch_size, input_patch_size, input_patch_size) for r in recs)
    assert remapped_images.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)


def test_srmnet_handles_missing_modalities() -> None:
    torch.manual_seed(0)
    model = SRMNet(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.tensor([[True, False, True, False]])

    with torch.no_grad():
        output = model(images, mask)

    assert output.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert torch.isfinite(output).all()


def test_srmnet_remaps_dataset_modalities_to_legacy_order() -> None:
    images = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1)
    mask = torch.tensor([[False, True, False, True]])

    remapped_images, remapped_mask = SRMNet._remap_input_order(images, mask)

    assert remapped_images.flatten().tolist() == [2.0, 0.0, 1.0, 3.0]
    assert remapped_mask.tolist() == [[False, False, True, True]]


def test_srmnet_window_starts_use_50_percent_overlap() -> None:
    assert SRMNet._window_starts(input_patch_size) == [0]
    assert SRMNet._window_starts(input_patch_size + 64) == [0, 64]

    with pytest.raises(RuntimeError, match=f"at least {input_patch_size}"):
        SRMNet._window_starts(input_patch_size - 1)


@pytest.mark.parametrize(
    ("images", "mask"),
    [
        (torch.zeros(1, 3, 16, 16, 16), torch.ones(1, 4, dtype=torch.bool)),
        (torch.zeros(1, 4, 16, 16, 16), torch.ones(1, 3, dtype=torch.bool)),
    ],
)
def test_srmnet_rejects_invalid_modality_or_mask_shape(
    images: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    model = SRMNet(num_cls=4).eval()

    with pytest.raises(RuntimeError, match="SRMNet expects"):
        model(images, mask)


def test_srmnet_resolves_from_model_config_builder() -> None:
    model_config = build_model_config("srmnet", {"num_cls": 4})

    assert model_config.model_class is SRMNet
    assert model_config.kwargs == {"num_cls": 4}


def test_srmnet_loss_resolves_from_loss_config_builder() -> None:
    loss_config = build_loss_config("srmnet", {"num_classes": 4})

    assert loss_config.loss_class is SRMNetLoss
    assert loss_config.kwargs == {"num_classes": 4}


def test_srmnet_transform_manager_resolves_from_kind() -> None:
    transform_manager = build_transform_manager(TransformKind.IMFUSE, crop_size=(128, 128, 128))

    assert isinstance(transform_manager, IMFuseTransformManager)
    assert transform_manager.crop_size == (128, 128, 128)


def test_srmnet_training_loss_end_to_end() -> None:
    torch.manual_seed(4)
    model = SRMNet(num_cls=4)
    model.is_training = True
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)
    target = _random_target(batch=1, num_classes=4, size=input_patch_size)

    outputs = model(images, mask)
    loss_fn = SRMNetLoss(num_classes=4)
    metrics = loss_fn.training_loss(outputs, target)

    expected = (
        metrics["pred_cross"]
        + metrics["pred_dice"]
        + metrics["preds_cross"]
        + metrics["preds_dice"]
        + 0.1 * metrics["recon"]
    )
    assert torch.allclose(metrics["loss"], expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("config_name", ["srmnet_18.yaml", "srmnet_23.yaml", "srmnet_25.yaml"])
def test_packaged_srmnet_configs_use_srmnet_training_contract(config_name: str) -> None:
    config = load_yaml_config(Path(config_name))

    assert config["trainer"] == "srmnet"
    assert config["model"] == "srmnet"
    assert config["loss"] == "srmnet"
    assert config["custom_trainer_kwargs"]["patch_size"] == 128
    assert config["custom_trainer_kwargs"]["transform_kind"] == "imfuse"
    assert config["scheduler"] == "poly"
    assert config["optimizer"] == "adam"
