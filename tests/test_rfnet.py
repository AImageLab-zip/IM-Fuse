from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mimose.enums import TransformKind
from mimose.flops import resolve_2023_config_paths
from mimose.models.config import build_model_config
from mimose.models.rfnet import RFNet, input_patch_size
from mimose.training.transforms import (
    IMFuseTransformManager,
    build_transform_manager,
)
from mimose.utils.cli_overrides import load_yaml_config


def test_rfnet_inference_forward_returns_segmentation_shape() -> None:
    model = RFNet(num_cls=4).eval()
    images = torch.zeros(1, 4, 16, 16, 16)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 4, 16, 16, 16)


def test_rfnet_training_forward_returns_fuse_sep_and_prm_predictions() -> None:
    model = RFNet(num_cls=4).eval()
    model.is_training = True
    images = torch.zeros(1, 4, 16, 16, 16)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        fuse_pred, sep_preds, prm_preds = model(images, mask)

    assert fuse_pred.shape == (1, 4, 16, 16, 16)
    assert len(sep_preds) == 4
    assert len(prm_preds) == 4
    assert all(pred.shape == (1, 4, 16, 16, 16) for pred in sep_preds)
    assert all(pred.shape == (1, 4, 16, 16, 16) for pred in prm_preds)


def test_rfnet_remaps_dataset_modalities_to_legacy_order() -> None:
    images = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1)
    mask = torch.tensor([[False, True, False, True]])

    remapped_images, remapped_mask = RFNet._remap_input_order(images, mask)

    assert remapped_images.flatten().tolist() == [2.0, 0.0, 1.0, 3.0]
    assert remapped_mask.tolist() == [[False, False, True, True]]


def test_rfnet_window_starts_use_50_percent_overlap() -> None:
    assert RFNet._window_starts(input_patch_size) == [0]
    assert RFNet._window_starts(input_patch_size + 40) == [0, 40]
    assert RFNet._window_starts(input_patch_size + 62) == [0, 40, 62]

    with pytest.raises(RuntimeError, match="at least 80"):
        RFNet._window_starts(input_patch_size - 1)


@pytest.mark.parametrize(
    ("images", "mask"),
    [
        (torch.zeros(1, 3, 16, 16, 16), torch.ones(1, 4, dtype=torch.bool)),
        (torch.zeros(1, 4, 16, 16, 16), torch.ones(1, 3, dtype=torch.bool)),
    ],
)
def test_rfnet_rejects_invalid_modality_or_mask_shape(
    images: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    model = RFNet(num_cls=4).eval()

    with pytest.raises(RuntimeError, match="RFNet expects"):
        model(images, mask)


def test_rfnet_resolves_from_model_config_builder() -> None:
    model_config = build_model_config("rfnet", {"num_cls": 4})

    assert model_config.model_class is RFNet
    assert model_config.kwargs == {"num_cls": 4}


def test_rfnet_transform_manager_resolves_from_kind() -> None:
    transform_manager = build_transform_manager(TransformKind.IMFUSE, crop_size=(80, 80, 80))

    assert isinstance(transform_manager, IMFuseTransformManager)
    assert transform_manager.crop_size == (80, 80, 80)


@pytest.mark.parametrize("config_name", ["rfnet_18.yaml", "rfnet_23.yaml"])
def test_packaged_rfnet_configs_use_imfuse_training_contract(config_name: str) -> None:
    config = load_yaml_config(Path(config_name))

    assert config["trainer"] == "imfuse"
    assert config["model"] == "rfnet"
    assert config["loss"] == "imfuse"
    assert config["lr"] == 0.0002
    assert config["weight_decay"] == 0.0001
    assert config["num_epochs"] == 300
    assert config["fp16"] is False
    assert config["seed"] == 999
    assert config["crop_min_size"] == (128, 128, 128)
    assert config["custom_trainer_kwargs"]["patch_size"] == 80
    assert config["custom_trainer_kwargs"]["region_fusion_start_epoch"] == 20
    assert config["custom_trainer_kwargs"]["transform_kind"] == "imfuse"


def test_packaged_2023_configs_include_rfnet_for_flops_all() -> None:
    config_names = {path.name for path in resolve_2023_config_paths()}

    assert "rfnet_23.yaml" in config_names
