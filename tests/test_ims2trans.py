from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mimose.enums import TransformKind
from mimose.losses.config import build_loss_config
from mimose.losses.ims2trans import IMS2TransLoss
from mimose.models.config import build_model_config
from mimose.models.ims2trans import IMS2Trans, input_patch_size
from mimose.training.transforms import (
    IMFuseTransformManager,
    build_transform_manager,
)
from mimose.utils.cli_overrides import load_yaml_config


def _random_target(batch: int, num_classes: int, size: int) -> torch.Tensor:
    labels = torch.randint(0, num_classes, (batch, size, size, size))
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def test_ims2trans_inference_forward_returns_segmentation_shape() -> None:
    torch.manual_seed(0)
    model = IMS2Trans(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)


def test_ims2trans_training_forward_returns_fuse_embeddings_and_prm_predictions() -> None:
    torch.manual_seed(0)
    model = IMS2Trans(num_cls=4).eval()
    model.is_training = True
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        fuse_pred, embeddings, prm_preds = model(images, mask)

    expected_shape = (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert fuse_pred.shape == expected_shape
    assert len(embeddings) == 5
    assert all(embedding.shape == (1, 512 * 8**3) for embedding in embeddings)
    assert len(prm_preds) == 4
    assert all(pred.shape == expected_shape for pred in prm_preds)


def test_ims2trans_handles_missing_modalities() -> None:
    torch.manual_seed(0)
    model = IMS2Trans(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.tensor([[True, False, True, False]])

    with torch.no_grad():
        output = model(images, mask)

    assert output.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert torch.isfinite(output).all()


def test_ims2trans_remaps_dataset_modalities_to_legacy_order() -> None:
    images = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1)
    mask = torch.tensor([[False, True, False, True]])

    remapped_images, remapped_mask = IMS2Trans._remap_input_order(images, mask)

    assert remapped_images.flatten().tolist() == [2.0, 0.0, 1.0, 3.0]
    assert remapped_mask.tolist() == [[False, False, True, True]]


def test_ims2trans_window_starts_use_50_percent_overlap() -> None:
    assert IMS2Trans._window_starts(input_patch_size) == [0]
    assert IMS2Trans._window_starts(input_patch_size + 64) == [0, 64]

    with pytest.raises(RuntimeError, match=f"at least {input_patch_size}"):
        IMS2Trans._window_starts(input_patch_size - 1)


@pytest.mark.parametrize(
    ("images", "mask"),
    [
        (torch.zeros(1, 3, 16, 16, 16), torch.ones(1, 4, dtype=torch.bool)),
        (torch.zeros(1, 4, 16, 16, 16), torch.ones(1, 3, dtype=torch.bool)),
    ],
)
def test_ims2trans_rejects_invalid_modality_or_mask_shape(
    images: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    model = IMS2Trans(num_cls=4).eval()

    with pytest.raises(RuntimeError, match="IMS2Trans expects"):
        model(images, mask)


def test_ims2trans_resolves_from_model_config_builder() -> None:
    model_config = build_model_config("ims2trans", {"num_cls": 4})

    assert model_config.model_class is IMS2Trans
    assert model_config.kwargs == {"num_cls": 4}


def test_ims2trans_loss_resolves_from_loss_config_builder() -> None:
    loss_config = build_loss_config("ims2trans", {"num_classes": 4})

    assert loss_config.loss_class is IMS2TransLoss
    assert loss_config.kwargs == {"num_classes": 4}


def test_ims2trans_transform_manager_resolves_from_kind() -> None:
    transform_manager = build_transform_manager(TransformKind.IMFUSE, crop_size=(128, 128, 128))

    assert isinstance(transform_manager, IMFuseTransformManager)
    assert transform_manager.crop_size == (128, 128, 128)


def test_ims2trans_training_loss_end_to_end() -> None:
    torch.manual_seed(4)
    model = IMS2Trans(num_cls=4)
    model.is_training = True
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)
    target = _random_target(batch=1, num_classes=4, size=input_patch_size)

    outputs = model(images, mask)
    loss_fn = IMS2TransLoss(num_classes=4)
    metrics = loss_fn.training_loss(outputs, target, include_fuse=True)

    expected = (
        metrics["fusecross"]
        + metrics["fusedice"]
        + metrics["prmcross"]
        + metrics["prmdice"]
        + 0.1 * metrics["dis"]
    )
    assert torch.allclose(metrics["loss"], expected, atol=1e-6, rtol=1e-6)


def test_ims2trans_training_loss_zeroes_fuse_term_when_excluded() -> None:
    torch.manual_seed(4)
    model = IMS2Trans(num_cls=4)
    model.is_training = True
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)
    target = _random_target(batch=1, num_classes=4, size=input_patch_size)

    outputs = model(images, mask)
    loss_fn = IMS2TransLoss(num_classes=4)
    metrics = loss_fn.training_loss(outputs, target, include_fuse=False)

    expected = metrics["prmcross"] + metrics["prmdice"] + 0.1 * metrics["dis"]
    assert torch.allclose(metrics["loss"], expected, atol=1e-6, rtol=1e-6)
    assert metrics["fusecross"].item() == pytest.approx(0.0)
    assert metrics["fusedice"].item() == pytest.approx(0.0)


@pytest.mark.parametrize("config_name", ["ims2trans_18.yaml", "ims2trans_23.yaml", "ims2trans_25.yaml"])
def test_packaged_ims2trans_configs_use_ims2trans_training_contract(config_name: str) -> None:
    config = load_yaml_config(Path(config_name))

    assert config["trainer"] == "ims2trans"
    assert config["model"] == "ims2trans"
    assert config["loss"] == "ims2trans"
    assert config["custom_trainer_kwargs"]["patch_size"] == 128
    assert config["custom_trainer_kwargs"]["region_fusion_start_epoch"] == 0
    assert config["custom_trainer_kwargs"]["transform_kind"] == "imfuse"
    assert config["scheduler"] == "poly"
    assert config["optimizer"] == "adam"
    assert config["batch_size"] == 1
