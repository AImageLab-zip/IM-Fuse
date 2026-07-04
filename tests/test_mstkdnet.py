from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mimose.enums import TransformKind
from mimose.losses.config import build_loss_config
from mimose.losses.mstkdnet import MSTKDNetLoss
from mimose.models.config import build_model_config
from mimose.models.mstkdnet import MSTKDNet, input_patch_size
from mimose.training.transforms import (
    IMFuseTransformManager,
    build_transform_manager,
)
from mimose.utils.cli_overrides import load_yaml_config


def _random_target(batch: int, num_classes: int, size: int) -> torch.Tensor:
    labels = torch.randint(0, num_classes, (batch, size, size, size))
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def test_mstkdnet_inference_forward_returns_segmentation_shape() -> None:
    torch.manual_seed(0)
    model = MSTKDNet(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)


def test_mstkdnet_training_forward_returns_student_and_teacher_tuples() -> None:
    torch.manual_seed(0)
    model = MSTKDNet(num_cls=4).eval()
    model.is_training = True
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        student_out, teacher_out = model(images, mask)

    expected_shape = (1, 4, input_patch_size, input_patch_size, input_patch_size)
    for branch in (student_out, teacher_out):
        uout, style, content, unetr_fs, extract_weights, global_style, logit = branch
        assert uout.shape == expected_shape
        assert logit.shape == expected_shape
        assert style.shape == (1, 128, 16, 16, 16)
        assert content.shape == (1, 128, 16, 16, 16)
        assert len(unetr_fs) == 4
        assert len(extract_weights) == 4
        assert len(global_style) == 3


def test_mstkdnet_handles_missing_modalities() -> None:
    torch.manual_seed(0)
    model = MSTKDNet(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.tensor([[True, False, True, False]])

    with torch.no_grad():
        output = model(images, mask)

    assert output.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert torch.isfinite(output).all()


def test_mstkdnet_remaps_dataset_modalities_to_legacy_order() -> None:
    images = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1)
    mask = torch.tensor([[False, True, False, True]])

    remapped_images, remapped_mask = MSTKDNet._remap_input_order(images, mask)

    # legacy order is [flair, t1, t1ce, t2] = external indices [2, 1, 0, 3]
    assert remapped_images.flatten().tolist() == [2.0, 1.0, 0.0, 3.0]
    assert remapped_mask.tolist() == [[False, True, False, True]]


def test_mstkdnet_window_starts_use_50_percent_overlap() -> None:
    assert MSTKDNet._window_starts(input_patch_size) == [0]
    assert MSTKDNet._window_starts(input_patch_size + 64) == [0, 64]

    with pytest.raises(RuntimeError, match=f"at least {input_patch_size}"):
        MSTKDNet._window_starts(input_patch_size - 1)


@pytest.mark.parametrize(
    ("images", "mask"),
    [
        (torch.zeros(1, 3, 16, 16, 16), torch.ones(1, 4, dtype=torch.bool)),
        (torch.zeros(1, 4, 16, 16, 16), torch.ones(1, 3, dtype=torch.bool)),
    ],
)
def test_mstkdnet_rejects_invalid_modality_or_mask_shape(
    images: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    model = MSTKDNet(num_cls=4).eval()

    with pytest.raises(RuntimeError, match="MSTKDNet expects"):
        model(images, mask)


def test_mstkdnet_resolves_from_model_config_builder() -> None:
    model_config = build_model_config("mstkdnet", {"num_cls": 4})

    assert model_config.model_class is MSTKDNet
    assert model_config.kwargs == {"num_cls": 4}


def test_mstkdnet_loss_resolves_from_loss_config_builder() -> None:
    loss_config = build_loss_config("mstkdnet", {"num_classes": 4})

    assert loss_config.loss_class is MSTKDNetLoss
    assert loss_config.kwargs == {"num_classes": 4}


def test_mstkdnet_transform_manager_resolves_from_kind() -> None:
    transform_manager = build_transform_manager(TransformKind.IMFUSE, crop_size=(128, 128, 128))

    assert isinstance(transform_manager, IMFuseTransformManager)
    assert transform_manager.crop_size == (128, 128, 128)


def test_mstkdnet_training_loss_end_to_end() -> None:
    torch.manual_seed(4)
    model = MSTKDNet(num_cls=4)
    model.is_training = True
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.tensor([[True, False, True, True]])
    target = _random_target(batch=1, num_classes=4, size=input_patch_size)

    outputs = model(images, mask)
    loss_fn = MSTKDNetLoss(num_classes=4)
    metrics = loss_fn.training_loss(outputs, target, epoch=0)

    expected = (
        (1.0 - loss_fn.weight_mispath) * metrics["loss_dc"]
        + loss_fn.weight_mispath * metrics["loss_miss_dc"]
        + loss_fn._consistency_weight(0) * metrics["consistency"]
        + loss_fn.weight_content * metrics["content"]
        + metrics["style"]
        + loss_fn.unetr_weight * metrics["unetr"]
        + loss_fn.evd_weight * metrics["evd"]
        + metrics["slkd"]
        + loss_fn.weight_gsm * metrics["gsm"]
    )
    assert torch.allclose(metrics["loss"], expected, atol=1e-4, rtol=1e-4)


def test_mstkdnet_consistency_weight_ramps_up_over_epochs() -> None:
    loss_fn = MSTKDNetLoss(num_classes=4, consistency_max_weight=10.0, consistency_rampup_epochs=20.0)

    # legacy's sigmoid_rampup never reaches exactly 0, even at epoch 0.
    assert loss_fn._consistency_weight(0) == pytest.approx(0.0674, abs=1e-3)
    assert loss_fn._consistency_weight(0) < loss_fn._consistency_weight(10) < loss_fn._consistency_weight(20)
    assert loss_fn._consistency_weight(20) == pytest.approx(10.0, abs=1e-6)


@pytest.mark.parametrize("config_name", ["mstkdnet_18.yaml", "mstkdnet_23.yaml", "mstkdnet_25.yaml"])
def test_packaged_mstkdnet_configs_use_mstkdnet_training_contract(config_name: str) -> None:
    config = load_yaml_config(Path(config_name))

    assert config["trainer"] == "mstkdnet"
    assert config["model"] == "mstkdnet"
    assert config["loss"] == "mstkdnet"
    assert config["custom_trainer_kwargs"]["patch_size"] == 128
    assert config["custom_trainer_kwargs"]["transform_kind"] == "imfuse"
    assert config["scheduler"] == "poly"
    assert config["optimizer"] == "adam"
    assert config["batch_size"] == 1
