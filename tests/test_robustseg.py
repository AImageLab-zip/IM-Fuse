from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mimose.enums import TransformKind
from mimose.losses.config import build_loss_config
from mimose.losses.robustseg import RobustSegLoss, kl_loss
from mimose.models.config import build_model_config
from mimose.models.robustseg import RobustSeg, input_patch_size
from mimose.training.transforms import (
    IMFuseTransformManager,
    build_transform_manager,
)
from mimose.utils.cli_overrides import load_yaml_config


def _random_probabilities(batch: int, num_classes: int, size: int) -> torch.Tensor:
    logits = torch.randn(batch, num_classes, size, size, size)
    return torch.softmax(logits, dim=1)


def _random_target(batch: int, num_classes: int, size: int) -> torch.Tensor:
    labels = torch.randint(0, num_classes, (batch, size, size, size))
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def test_robustseg_inference_forward_returns_segmentation_shape() -> None:
    model = RobustSeg(num_cls=4).eval()
    images = torch.zeros(1, 4, 32, 32, 32)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 4, 32, 32, 32)


def test_robustseg_training_forward_returns_outputs_dict() -> None:
    model = RobustSeg(num_cls=4).eval()
    model.is_training = True
    images = torch.zeros(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        outputs = model(images, mask)

    assert outputs["seg_pred"].shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert outputs["seg_logit"].shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)
    for modality in ("Flair", "T1c", "T1", "T2"):
        assert outputs[f"reconstruct_{modality}"].shape == (1, 1, input_patch_size, input_patch_size, input_patch_size)
        assert outputs[f"mu_{modality}"].shape == (1, 128, 1, 1, 1)
        assert outputs[f"sigma_{modality}"].shape == (1, 128, 1, 1, 1)
        assert outputs["images"][modality].shape == (1, 1, input_patch_size, input_patch_size, input_patch_size)


def test_robustseg_remaps_dataset_modalities_to_legacy_order() -> None:
    images = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1)
    mask = torch.tensor([[False, True, False, True]])

    remapped_images, remapped_mask = RobustSeg._remap_input_order(images, mask)

    assert remapped_images.flatten().tolist() == [2.0, 0.0, 1.0, 3.0]
    assert remapped_mask.tolist() == [[False, False, True, True]]


def test_robustseg_window_starts_use_50_percent_overlap() -> None:
    assert RobustSeg._window_starts(input_patch_size) == [0]
    assert RobustSeg._window_starts(input_patch_size + 40) == [0, 40]

    with pytest.raises(RuntimeError, match=f"at least {input_patch_size}"):
        RobustSeg._window_starts(input_patch_size - 1)


@pytest.mark.parametrize(
    ("images", "mask"),
    [
        (torch.zeros(1, 3, 16, 16, 16), torch.ones(1, 4, dtype=torch.bool)),
        (torch.zeros(1, 4, 16, 16, 16), torch.ones(1, 3, dtype=torch.bool)),
    ],
)
def test_robustseg_rejects_invalid_modality_or_mask_shape(
    images: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    model = RobustSeg(num_cls=4).eval()

    with pytest.raises(RuntimeError, match="RobustSeg expects"):
        model(images, mask)


def test_robustseg_resolves_from_model_config_builder() -> None:
    model_config = build_model_config("robustseg", {"num_cls": 4})

    assert model_config.model_class is RobustSeg
    assert model_config.kwargs == {"num_cls": 4}


def test_robustseg_loss_resolves_from_loss_config_builder() -> None:
    loss_config = build_loss_config("robustseg", {"num_classes": 4})

    assert loss_config.loss_class is RobustSegLoss
    assert loss_config.kwargs == {"num_classes": 4}


def test_robustseg_transform_manager_resolves_from_kind() -> None:
    transform_manager = build_transform_manager(TransformKind.IMFUSE, crop_size=(80, 80, 80))

    assert isinstance(transform_manager, IMFuseTransformManager)
    assert transform_manager.crop_size == (80, 80, 80)


def test_robustseg_kl_loss_is_zero_at_standard_normal() -> None:
    mu = torch.zeros(2, 128, 1, 1, 1)
    logvar = torch.zeros(2, 128, 1, 1, 1)

    assert kl_loss(mu, logvar).item() == pytest.approx(0.0, abs=1e-6)


def test_robustseg_training_loss_end_to_end() -> None:
    torch.manual_seed(4)
    model = RobustSeg(num_cls=4)
    model.is_training = True
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)
    target = _random_target(batch=1, num_classes=4, size=input_patch_size)

    outputs = model(images, mask)
    loss_fn = RobustSegLoss(num_classes=4)
    metrics = loss_fn.training_loss(outputs, target)

    expected = metrics["cross"] + metrics["dice"] + 0.1 * metrics["recon"] + 0.1 * metrics["kld"]
    assert torch.allclose(metrics["loss"], expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("config_name", ["robustseg_18.yaml", "robustseg_23.yaml", "robustseg_25.yaml"])
def test_packaged_robustseg_configs_use_imfuse_training_contract(config_name: str) -> None:
    config = load_yaml_config(Path(config_name))

    assert config["trainer"] == "robustseg"
    assert config["model"] == "robustseg"
    assert config["loss"] == "robustseg"
    assert config["custom_trainer_kwargs"]["patch_size"] == 80
    assert config["custom_trainer_kwargs"]["transform_kind"] == "imfuse"
    assert config["scheduler"] == "poly"
