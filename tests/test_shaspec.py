from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mimose.enums import TransformKind
from mimose.losses.config import build_loss_config
from mimose.losses.shaspec import ShaSpecLoss
from mimose.models.config import build_model_config
from mimose.models.shaspec import ShaSpec, input_patch_size
from mimose.training.transforms import (
    IMFuseTransformManager,
    build_transform_manager,
)
from mimose.utils.cli_overrides import load_yaml_config


def _random_target(batch: int, num_classes: int, size: int) -> torch.Tensor:
    labels = torch.randint(0, num_classes, (batch, size, size, size))
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def test_shaspec_inference_forward_returns_segmentation_shape() -> None:
    torch.manual_seed(0)
    model = ShaSpec(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)


def test_shaspec_training_forward_returns_outputs_dict() -> None:
    torch.manual_seed(0)
    model = ShaSpec(num_cls=4).eval()
    model.is_training = True
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        outputs = model(images, mask)

    assert outputs["seg_pred"].shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert len(outputs["shared_content"]) == 4
    assert all(t.shape == (1, 256, 5, 5, 5) for t in outputs["shared_content"])
    assert outputs["dom_logits"].shape == (4, 4)


def test_shaspec_handles_missing_modalities_and_heterogeneous_batch_masks() -> None:
    torch.manual_seed(0)
    model = ShaSpec(num_cls=4).eval()
    images = torch.randn(2, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.tensor([[True, False, True, False], [False, True, False, True]])

    with torch.no_grad():
        output = model(images, mask)

    assert output.shape == (2, 4, input_patch_size, input_patch_size, input_patch_size)
    assert torch.isfinite(output).all()


def test_shaspec_remaps_dataset_modalities_to_legacy_order() -> None:
    images = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1)
    mask = torch.tensor([[False, True, False, True]])

    remapped_images, remapped_mask = ShaSpec._remap_input_order(images, mask)

    assert remapped_images.flatten().tolist() == [2.0, 1.0, 0.0, 3.0]
    assert remapped_mask.tolist() == [[False, True, False, True]]


def test_shaspec_window_starts_use_50_percent_overlap() -> None:
    assert ShaSpec._window_starts(input_patch_size) == [0]
    assert ShaSpec._window_starts(input_patch_size + 40) == [0, 40]

    with pytest.raises(RuntimeError, match=f"at least {input_patch_size}"):
        ShaSpec._window_starts(input_patch_size - 1)


@pytest.mark.parametrize(
    ("images", "mask"),
    [
        (torch.zeros(1, 3, 16, 16, 16), torch.ones(1, 4, dtype=torch.bool)),
        (torch.zeros(1, 4, 16, 16, 16), torch.ones(1, 3, dtype=torch.bool)),
    ],
)
def test_shaspec_rejects_invalid_modality_or_mask_shape(
    images: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    model = ShaSpec(num_cls=4).eval()

    with pytest.raises(RuntimeError, match="ShaSpec expects"):
        model(images, mask)


def test_shaspec_resolves_from_model_config_builder() -> None:
    model_config = build_model_config("shaspec", {"num_cls": 4})

    assert model_config.model_class is ShaSpec
    assert model_config.kwargs == {"num_cls": 4}


def test_shaspec_loss_resolves_from_loss_config_builder() -> None:
    loss_config = build_loss_config("shaspec", {"num_classes": 4})

    assert loss_config.loss_class is ShaSpecLoss
    assert loss_config.kwargs == {"num_classes": 4}


def test_shaspec_transform_manager_resolves_from_kind() -> None:
    transform_manager = build_transform_manager(TransformKind.IMFUSE, crop_size=(80, 80, 80))

    assert isinstance(transform_manager, IMFuseTransformManager)
    assert transform_manager.crop_size == (80, 80, 80)


def test_shaspec_training_loss_end_to_end() -> None:
    torch.manual_seed(4)
    model = ShaSpec(num_cls=4)
    model.is_training = True
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)
    target = _random_target(batch=1, num_classes=4, size=input_patch_size)

    outputs = model(images, mask)
    loss_fn = ShaSpecLoss(num_classes=4)
    metrics = loss_fn.training_loss(outputs, target)

    expected = metrics["cross"] + metrics["dice"] + 0.1 * metrics["shared_similarity"] + 0.02 * metrics["domain_cls"]
    assert torch.allclose(metrics["loss"], expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("config_name", ["shaspec_18.yaml", "shaspec_23.yaml", "shaspec_25.yaml"])
def test_packaged_shaspec_configs_use_shaspec_training_contract(config_name: str) -> None:
    config = load_yaml_config(Path(config_name))

    assert config["trainer"] == "shaspec"
    assert config["model"] == "shaspec"
    assert config["loss"] == "shaspec"
    assert config["custom_trainer_kwargs"]["patch_size"] == 80
    assert config["custom_trainer_kwargs"]["transform_kind"] == "imfuse"
    assert config["scheduler"] == "poly"
    assert config["optimizer"] == "sgd"
    assert config["nesterov"] is True
