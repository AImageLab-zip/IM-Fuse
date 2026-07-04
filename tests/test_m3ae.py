from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mimose.enums import TransformKind
from mimose.losses.config import build_loss_config
from mimose.losses.m3ae import M3AELoss
from mimose.models.config import build_model_config
from mimose.models.m3ae import M3AE, input_patch_size
from mimose.training.transforms import (
    IMFuseTransformManager,
    build_transform_manager,
)
from mimose.utils.cli_overrides import load_yaml_config


def _random_target(batch: int, num_classes: int, size: int) -> torch.Tensor:
    labels = torch.randint(0, num_classes, (batch, size, size, size))
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def test_m3ae_forward_returns_segmentation_shape() -> None:
    torch.manual_seed(0)
    model = M3AE(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert torch.allclose(output.sum(dim=1), torch.ones_like(output.sum(dim=1)), atol=1e-5)


def test_m3ae_train_and_eval_forward_have_the_same_output_contract() -> None:
    torch.manual_seed(0)
    model = M3AE(num_cls=4)
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    model.train()
    train_out = model(images, mask)
    model.eval()
    with torch.no_grad():
        eval_out = model(images, mask)

    assert train_out.shape == eval_out.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)


def test_m3ae_substitutes_missing_modalities_with_learned_embedding() -> None:
    torch.manual_seed(0)
    model = M3AE(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    all_present = torch.ones(1, 4, dtype=torch.bool)
    one_missing = torch.tensor([[True, True, True, False]])

    with torch.no_grad():
        out_full = model(images, all_present)
        out_missing = model(images, one_missing)

    assert torch.isfinite(out_missing).all()
    assert not torch.allclose(out_full, out_missing)


def test_m3ae_rejects_non_patch_sized_input() -> None:
    model = M3AE(num_cls=4).eval()
    images = torch.zeros(1, 4, 64, 64, 64)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with pytest.raises(RuntimeError, match="M3AE requires spatial dimensions"):
        model(images, mask)


@pytest.mark.parametrize(
    ("images", "mask"),
    [
        (torch.zeros(1, 3, input_patch_size, input_patch_size, input_patch_size), torch.ones(1, 4, dtype=torch.bool)),
        (torch.zeros(1, 4, input_patch_size, input_patch_size, input_patch_size), torch.ones(1, 3, dtype=torch.bool)),
    ],
)
def test_m3ae_rejects_invalid_modality_or_mask_shape(
    images: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    model = M3AE(num_cls=4).eval()

    with pytest.raises(RuntimeError, match="M3AE expects"):
        model(images, mask)


def test_m3ae_window_starts_use_50_percent_overlap() -> None:
    assert M3AE._window_starts(input_patch_size) == [0]
    assert M3AE._window_starts(input_patch_size + 64) == [0, 64]

    with pytest.raises(RuntimeError, match=f"at least {input_patch_size}"):
        M3AE._window_starts(input_patch_size - 1)


def test_m3ae_predict_slides_over_larger_volumes() -> None:
    torch.manual_seed(0)
    model = M3AE(num_cls=4).eval()
    images = torch.randn(1, 4, 160, 160, 160)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        pred = model.predict(images, mask)

    assert pred.shape == (1, 4, 160, 160, 160)
    assert torch.isfinite(pred).all()


def test_m3ae_resolves_from_model_config_builder() -> None:
    model_config = build_model_config("m3ae", {"num_cls": 4})

    assert model_config.model_class is M3AE
    assert model_config.kwargs == {"num_cls": 4}


def test_m3ae_loss_resolves_from_loss_config_builder() -> None:
    loss_config = build_loss_config("m3ae", {"num_classes": 4})

    assert loss_config.loss_class is M3AELoss
    assert loss_config.kwargs == {"num_classes": 4}


def test_m3ae_transform_manager_resolves_from_kind() -> None:
    transform_manager = build_transform_manager(TransformKind.IMFUSE, crop_size=(128, 128, 128))

    assert isinstance(transform_manager, IMFuseTransformManager)
    assert transform_manager.crop_size == (128, 128, 128)


def test_m3ae_training_loss_end_to_end() -> None:
    torch.manual_seed(4)
    model = M3AE(num_cls=4)
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)
    target = _random_target(batch=1, num_classes=4, size=input_patch_size)

    output = model(images, mask)
    loss_fn = M3AELoss(num_classes=4)
    metrics = loss_fn.training_loss(output, target)

    expected = metrics["cross"] + metrics["dice"]
    assert torch.allclose(metrics["loss"], expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("config_name", ["m3ae_18.yaml", "m3ae_23.yaml", "m3ae_25.yaml"])
def test_packaged_m3ae_configs_use_m3ae_training_contract(config_name: str) -> None:
    config = load_yaml_config(Path(config_name))

    assert config["trainer"] == "m3ae"
    assert config["model"] == "m3ae"
    assert config["loss"] == "m3ae"
    assert config["custom_trainer_kwargs"]["patch_size"] == 128
    assert config["custom_trainer_kwargs"]["transform_kind"] == "imfuse"
    assert config["scheduler"] == "cosine"
    assert config["optimizer"] == "adam"
