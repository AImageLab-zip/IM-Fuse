from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mimose.enums import TransformKind
from mimose.losses.config import build_loss_config
from mimose.losses.rfl import RFLLoss
from mimose.models.config import build_model_config
from mimose.models.rfl import RFL, input_patch_size
from mimose.training.transforms import (
    IMFuseTransformManager,
    build_transform_manager,
)
from mimose.utils.cli_overrides import load_yaml_config


def _random_target(batch: int, num_classes: int, size: int) -> torch.Tensor:
    labels = torch.randint(0, num_classes, (batch, size, size, size))
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def test_rfl_inference_forward_returns_segmentation_shape() -> None:
    torch.manual_seed(0)
    model = RFL(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)


def test_rfl_training_forward_returns_fuse_sep_prm_and_rfim_outputs() -> None:
    torch.manual_seed(0)
    model = RFL(num_cls=4).eval()
    model.is_training = True
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        outputs = model(images, mask)

    fuse_pred, sep_preds, prm_preds, forward_outputs, inverse_outputs, intra_tokens, global_tokens_used = outputs
    expected_shape = (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert fuse_pred.shape == expected_shape
    assert len(sep_preds) == 4
    assert len(prm_preds) == 4
    assert all(pred.shape == expected_shape for pred in sep_preds)
    assert all(pred.shape == expected_shape for pred in prm_preds)
    for tensors in (forward_outputs, inverse_outputs, intra_tokens, global_tokens_used):
        assert len(tensors) == 4
        assert all(t.shape == (1, 512, 512) for t in tensors)


def test_rfl_handles_missing_modalities() -> None:
    torch.manual_seed(0)
    model = RFL(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.tensor([[True, False, True, False]])

    with torch.no_grad():
        output = model(images, mask)

    assert output.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert torch.isfinite(output).all()


def test_rfl_remaps_dataset_modalities_to_legacy_order() -> None:
    images = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1)
    mask = torch.tensor([[False, True, False, True]])

    remapped_images, remapped_mask = RFL._remap_input_order(images, mask)

    assert remapped_images.flatten().tolist() == [2.0, 0.0, 1.0, 3.0]
    assert remapped_mask.tolist() == [[False, False, True, True]]


def test_rfl_window_starts_use_50_percent_overlap() -> None:
    assert RFL._window_starts(input_patch_size) == [0]
    assert RFL._window_starts(input_patch_size + 64) == [0, 64]

    with pytest.raises(RuntimeError, match=f"at least {input_patch_size}"):
        RFL._window_starts(input_patch_size - 1)


@pytest.mark.parametrize(
    ("images", "mask"),
    [
        (torch.zeros(1, 3, 16, 16, 16), torch.ones(1, 4, dtype=torch.bool)),
        (torch.zeros(1, 4, 16, 16, 16), torch.ones(1, 3, dtype=torch.bool)),
    ],
)
def test_rfl_rejects_invalid_modality_or_mask_shape(
    images: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    model = RFL(num_cls=4).eval()

    with pytest.raises(RuntimeError, match="RFL expects"):
        model(images, mask)


def test_rfl_resolves_from_model_config_builder() -> None:
    model_config = build_model_config("rfl", {"num_cls": 4})

    assert model_config.model_class is RFL
    assert model_config.kwargs == {"num_cls": 4}


def test_rfl_loss_resolves_from_loss_config_builder() -> None:
    loss_config = build_loss_config("rfl", {"num_classes": 4})

    assert loss_config.loss_class is RFLLoss
    assert loss_config.kwargs == {"num_classes": 4}


def test_rfl_transform_manager_resolves_from_kind() -> None:
    transform_manager = build_transform_manager(TransformKind.IMFUSE, crop_size=(128, 128, 128))

    assert isinstance(transform_manager, IMFuseTransformManager)
    assert transform_manager.crop_size == (128, 128, 128)


def test_rfl_training_loss_end_to_end() -> None:
    torch.manual_seed(4)
    model = RFL(num_cls=4)
    model.is_training = True
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.tensor([[True, False, True, True]])
    target = _random_target(batch=1, num_classes=4, size=input_patch_size)

    outputs = model(images, mask)
    loss_fn = RFLLoss(num_classes=4)
    metrics = loss_fn.training_loss(outputs, target, include_fuse=True)

    expected = (
        metrics["fusecross"]
        + metrics["fusedice"]
        + metrics["sepcross"]
        + metrics["sepdice"]
        + metrics["prmcross"]
        + metrics["prmdice"]
        + metrics["forwardmse"]
        + metrics["rflmse"]
    )
    assert torch.allclose(metrics["loss"], expected, atol=1e-6, rtol=1e-6)
    metrics["loss"].backward()


def test_rfl_training_loss_zeroes_fuse_term_when_excluded() -> None:
    torch.manual_seed(4)
    model = RFL(num_cls=4)
    model.is_training = True
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)
    target = _random_target(batch=1, num_classes=4, size=input_patch_size)

    outputs = model(images, mask)
    loss_fn = RFLLoss(num_classes=4)
    metrics = loss_fn.training_loss(outputs, target, include_fuse=False)

    expected = (
        metrics["sepcross"]
        + metrics["sepdice"]
        + metrics["prmcross"]
        + metrics["prmdice"]
        + metrics["forwardmse"]
        + metrics["rflmse"]
    )
    assert torch.allclose(metrics["loss"], expected, atol=1e-6, rtol=1e-6)
    assert metrics["fusecross"].item() == pytest.approx(0.0)
    assert metrics["fusedice"].item() == pytest.approx(0.0)


@pytest.mark.parametrize("config_name", ["rfl_18.yaml", "rfl_23.yaml", "rfl_25.yaml"])
def test_packaged_rfl_configs_use_rfl_training_contract(config_name: str) -> None:
    config = load_yaml_config(Path(config_name))

    assert config["trainer"] == "rfl"
    assert config["model"] == "rfl"
    assert config["loss"] == "rfl"
    assert config["custom_trainer_kwargs"]["patch_size"] == 128
    assert config["custom_trainer_kwargs"]["region_fusion_start_epoch"] == 0
    assert config["custom_trainer_kwargs"]["transform_kind"] == "imfuse"
    assert config["scheduler"] == "poly"
    assert config["optimizer"] == "adam"
