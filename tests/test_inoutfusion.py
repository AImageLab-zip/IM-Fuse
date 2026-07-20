from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mimose.enums import TransformKind
from mimose.losses.config import build_loss_config
from mimose.losses.inoutfusion import InOutFusionLoss
from mimose.models.config import build_model_config
from mimose.models.inoutfusion import InOutFusion, input_patch_size
from mimose.training.transforms import (
    IMFuseTransformManager,
    build_transform_manager,
)
from mimose.utils.cli_overrides import load_yaml_config


def _random_target(batch: int, num_classes: int, size: int) -> torch.Tensor:
    labels = torch.randint(0, num_classes, (batch, size, size, size))
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def test_inoutfusion_inference_forward_returns_segmentation_shape() -> None:
    torch.manual_seed(0)
    model = InOutFusion(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)


def test_inoutfusion_training_forward_returns_fuse_pred_with_empty_aux_slots() -> None:
    torch.manual_seed(0)
    model = InOutFusion(num_cls=4).eval()
    model.is_training = True
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        fuse_pred, sep_preds, prm_preds = model(images, mask)

    assert fuse_pred.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert sep_preds == ()
    assert prm_preds == ()


def test_inoutfusion_handles_missing_modalities() -> None:
    torch.manual_seed(0)
    model = InOutFusion(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.tensor([[True, False, True, False]])

    with torch.no_grad():
        output = model(images, mask)

    assert output.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert torch.isfinite(output).all()


def test_inoutfusion_handles_heterogeneous_per_sample_masks() -> None:
    torch.manual_seed(0)
    model = InOutFusion(num_cls=4).eval()
    images = torch.randn(2, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.tensor([[True, False, True, True], [False, True, True, False]])

    with torch.no_grad():
        output = model(images, mask)

    assert output.shape == (2, 4, input_patch_size, input_patch_size, input_patch_size)
    assert torch.isfinite(output).all()


def test_inoutfusion_remaps_dataset_modalities_to_legacy_order() -> None:
    images = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1)
    mask = torch.tensor([[False, True, False, True]])

    remapped_images, remapped_mask = InOutFusion._remap_input_order(images, mask)

    assert remapped_images.flatten().tolist() == [0.0, 1.0, 3.0, 2.0]
    assert remapped_mask.tolist() == [[False, True, True, False]]


def test_inoutfusion_window_starts_use_50_percent_overlap() -> None:
    assert InOutFusion._window_starts(input_patch_size) == [0]
    assert InOutFusion._window_starts(input_patch_size + 64) == [0, 64]

    with pytest.raises(RuntimeError, match=f"at least {input_patch_size}"):
        InOutFusion._window_starts(input_patch_size - 1)


@pytest.mark.parametrize(
    ("images", "mask"),
    [
        (torch.zeros(1, 3, 16, 16, 16), torch.ones(1, 4, dtype=torch.bool)),
        (torch.zeros(1, 4, 16, 16, 16), torch.ones(1, 3, dtype=torch.bool)),
    ],
)
def test_inoutfusion_rejects_invalid_modality_or_mask_shape(
    images: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    model = InOutFusion(num_cls=4).eval()

    with pytest.raises(RuntimeError, match="InOutFusion expects"):
        model(images, mask)


def test_inoutfusion_resolves_from_model_config_builder() -> None:
    model_config = build_model_config("inoutfusion", {"num_cls": 4})

    assert model_config.model_class is InOutFusion
    assert model_config.kwargs == {"num_cls": 4}


def test_inoutfusion_loss_resolves_from_loss_config_builder() -> None:
    loss_config = build_loss_config("inoutfusion", {"num_classes": 4})

    assert loss_config.loss_class is InOutFusionLoss
    assert loss_config.kwargs == {"num_classes": 4}


def test_inoutfusion_transform_manager_resolves_from_kind() -> None:
    transform_manager = build_transform_manager(TransformKind.IMFUSE, crop_size=(128, 128, 128))

    assert isinstance(transform_manager, IMFuseTransformManager)
    assert transform_manager.crop_size == (128, 128, 128)


def test_inoutfusion_training_loss_end_to_end() -> None:
    torch.manual_seed(4)
    model = InOutFusion(num_cls=4)
    model.is_training = True
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.tensor([[True, False, True, True]])
    target = _random_target(batch=1, num_classes=4, size=input_patch_size)

    outputs = model(images, mask)
    loss_fn = InOutFusionLoss(num_classes=4)
    metrics = loss_fn.training_loss(outputs, target, include_fuse=True)

    assert torch.allclose(metrics["loss"], metrics["fusedice"], atol=1e-6, rtol=1e-6)
    assert metrics["fusecross"].item() == pytest.approx(0.0)
    assert metrics["sepcross"].item() == pytest.approx(0.0)
    assert metrics["sepdice"].item() == pytest.approx(0.0)
    assert metrics["prmcross"].item() == pytest.approx(0.0)
    assert metrics["prmdice"].item() == pytest.approx(0.0)


def test_inoutfusion_training_loss_zeroes_when_fuse_excluded() -> None:
    torch.manual_seed(4)
    model = InOutFusion(num_cls=4)
    model.is_training = True
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)
    target = _random_target(batch=1, num_classes=4, size=input_patch_size)

    outputs = model(images, mask)
    loss_fn = InOutFusionLoss(num_classes=4)
    metrics = loss_fn.training_loss(outputs, target, include_fuse=False)

    assert metrics["loss"].item() == pytest.approx(0.0)
    assert metrics["fusedice"].item() == pytest.approx(0.0)


@pytest.mark.parametrize("config_name", ["inoutfusion_18.yaml", "inoutfusion_23.yaml", "inoutfusion_25.yaml"])
def test_packaged_inoutfusion_configs_use_imfuse_training_contract(config_name: str) -> None:
    config = load_yaml_config(Path(config_name))

    assert config["trainer"] == "imfuse"
    assert config["model"] == "inoutfusion"
    assert config["loss"] == "inoutfusion"
    assert config["optimizer"] == "adamw"
    assert config["lr"] == 0.0003
    assert config["weight_decay"] == 0.0001
    assert config["batch_size"] == 1
    assert config["num_epochs"] == 200
    assert config["custom_trainer_kwargs"]["patch_size"] == 128
    assert config["custom_trainer_kwargs"]["transform_kind"] == "imfuse"
    assert config["scheduler"] == "poly"
