from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mimose.enums import TransformKind
from mimose.models.unetmfi import UNETMFI, input_patch_size
from mimose.training.transforms import (
    IMFuseTransformManager,
    build_transform_manager,
)
from mimose.utils.cli_overrides import load_yaml_config


def test_unetmfi_inference_forward_returns_segmentation_shape() -> None:
    torch.manual_seed(0)
    model = UNETMFI(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)


def test_unetmfi_training_forward_returns_fuse_sep_and_prm_predictions() -> None:
    torch.manual_seed(0)
    model = UNETMFI(num_cls=4).eval()
    model.is_training = True
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        fuse_pred, sep_preds, prm_preds = model(images, mask)

    expected_shape = (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert fuse_pred.shape == expected_shape
    assert len(sep_preds) == 4
    assert len(prm_preds) == 8
    assert all(pred.shape == expected_shape for pred in sep_preds)
    assert all(pred.shape == expected_shape for pred in prm_preds)


def test_unetmfi_handles_missing_modalities() -> None:
    torch.manual_seed(0)
    model = UNETMFI(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.tensor([[True, False, True, False]])

    with torch.no_grad():
        output = model(images, mask)

    assert output.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert torch.isfinite(output).all()


def test_unetmfi_handles_heterogeneous_per_sample_masks() -> None:
    torch.manual_seed(0)
    model = UNETMFI(num_cls=4).eval()
    images = torch.randn(2, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.tensor([[True, False, True, True], [False, True, True, False]])

    with torch.no_grad():
        output = model(images, mask)

    assert output.shape == (2, 4, input_patch_size, input_patch_size, input_patch_size)
    assert torch.isfinite(output).all()


def test_unetmfi_remaps_dataset_modalities_to_legacy_order() -> None:
    images = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1)
    mask = torch.tensor([[False, True, False, True]])

    remapped_images, remapped_mask = UNETMFI._remap_input_order(images, mask)

    assert remapped_images.flatten().tolist() == [2.0, 0.0, 1.0, 3.0]
    assert remapped_mask.tolist() == [[False, False, True, True]]


def test_unetmfi_window_starts_use_50_percent_overlap() -> None:
    assert UNETMFI._window_starts(input_patch_size) == [0]
    assert UNETMFI._window_starts(input_patch_size + 64) == [0, 64]

    with pytest.raises(RuntimeError, match=f"at least {input_patch_size}"):
        UNETMFI._window_starts(input_patch_size - 1)


@pytest.mark.parametrize(
    ("images", "mask"),
    [
        (torch.zeros(1, 3, 16, 16, 16), torch.ones(1, 4, dtype=torch.bool)),
        (torch.zeros(1, 4, 16, 16, 16), torch.ones(1, 3, dtype=torch.bool)),
    ],
)
def test_unetmfi_rejects_invalid_modality_or_mask_shape(
    images: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    model = UNETMFI(num_cls=4).eval()

    with pytest.raises(RuntimeError, match="UNETMFI expects"):
        model(images, mask)


def test_unetmfi_resolves_from_model_config_builder() -> None:
    from mimose.models.config import build_model_config

    model_config = build_model_config("unetmfi", {"num_cls": 4})

    assert model_config.model_class is UNETMFI
    assert model_config.kwargs == {"num_cls": 4}


def test_unetmfi_transform_manager_resolves_from_kind() -> None:
    transform_manager = build_transform_manager(TransformKind.IMFUSE, crop_size=(128, 128, 128))

    assert isinstance(transform_manager, IMFuseTransformManager)
    assert transform_manager.crop_size == (128, 128, 128)


@pytest.mark.parametrize("config_name", ["unetmfi_18.yaml", "unetmfi_23.yaml", "unetmfi_25.yaml"])
def test_packaged_unetmfi_configs_use_imfuse_training_contract(config_name: str) -> None:
    config = load_yaml_config(Path(config_name))

    assert config["trainer"] == "imfuse"
    assert config["model"] == "unetmfi"
    assert config["loss"] == "imfuse"
    assert config["lr"] == 0.00005
    assert config["weight_decay"] == 0.00001
    assert config["num_epochs"] == 1200
    assert config["batch_size"] == 1
    assert config["fp16"] is False
    assert config["seed"] == 999
    assert config["crop_min_size"] == (128, 128, 128)
    assert config["custom_trainer_kwargs"]["patch_size"] == 128
    assert config["custom_trainer_kwargs"]["region_fusion_start_epoch"] == 0
    assert config["custom_trainer_kwargs"]["transform_kind"] == "imfuse"
    assert config["optimizer"] == "adam"
    assert config["scheduler"] == "poly"
