from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mimose.enums import TransformKind
from mimose.flops import resolve_2023_config_paths
from mimose.losses.config import build_loss_config
from mimose.losses.mambavitakd import MambaVitAKDLoss
from mimose.models.config import build_model_config
from mimose.models.mambavitakd import MambaVitAKD, input_patch_size
from mimose.training.transforms import (
    IMFuseTransformManager,
    build_transform_manager,
)
from mimose.utils.cli_overrides import load_yaml_config


def _random_target(*, batch: int, num_classes: int, size: int) -> torch.Tensor:
    labels = torch.randint(0, num_classes, (batch, 1, size, size, size))
    return torch.cat([(labels == index).float() for index in range(num_classes)], dim=1)


def test_mambavitakd_inference_forward_returns_segmentation_shape() -> None:
    model = MambaVitAKD(num_cls=4).eval()
    images = torch.zeros(1, 4, 16, 16, 16)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 4, 16, 16, 16)


def test_mambavitakd_training_forward_returns_student_teacher_and_attn_loss() -> None:
    model = MambaVitAKD(num_cls=4).eval()
    model.is_training = True
    images = torch.zeros(1, 4, 16, 16, 16)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        student_out, teacher_out, attn_loss = model(images, mask)

    fuse_pred, sep_preds, prm_preds, feature, logits = student_out
    fuse_pred_t, sep_preds_t, prm_preds_t, feature_t, logits_t = teacher_out

    assert fuse_pred.shape == (1, 4, 16, 16, 16)
    assert fuse_pred_t.shape == (1, 4, 16, 16, 16)
    assert len(sep_preds) == 4
    assert len(sep_preds_t) == 4
    assert len(prm_preds) == 4
    assert len(prm_preds_t) == 4
    assert all(pred.shape == (1, 4, 16, 16, 16) for pred in sep_preds)
    assert all(pred.shape == (1, 4, 16, 16, 16) for pred in prm_preds)
    assert feature.shape[0] == 1
    assert feature_t.shape == feature.shape
    assert logits.shape == (1, 4, 16, 16, 16)
    assert logits_t.shape == (1, 4, 16, 16, 16)
    assert attn_loss.ndim == 0


def test_mambavitakd_handles_missing_modalities() -> None:
    torch.manual_seed(0)
    model = MambaVitAKD(num_cls=4).eval()
    images = torch.randn(1, 4, 16, 16, 16)
    mask = torch.tensor([[True, False, True, False]])

    with torch.no_grad():
        output = model(images, mask)

    assert output.shape == (1, 4, 16, 16, 16)
    assert torch.isfinite(output).all()


def test_mambavitakd_remaps_dataset_modalities_to_legacy_order() -> None:
    images = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1)
    mask = torch.tensor([[False, True, False, True]])

    remapped_images, remapped_mask = MambaVitAKD._remap_input_order(images, mask)

    assert remapped_images.flatten().tolist() == [2.0, 0.0, 1.0, 3.0]
    assert remapped_mask.tolist() == [[False, False, True, True]]


def test_mambavitakd_window_starts_use_50_percent_overlap() -> None:
    assert MambaVitAKD._window_starts(input_patch_size) == [0]
    assert MambaVitAKD._window_starts(input_patch_size + 40) == [0, 40]
    assert MambaVitAKD._window_starts(input_patch_size + 62) == [0, 40, 62]

    with pytest.raises(RuntimeError, match="at least 80"):
        MambaVitAKD._window_starts(input_patch_size - 1)


@pytest.mark.parametrize(
    ("images", "mask"),
    [
        (torch.zeros(1, 3, 16, 16, 16), torch.ones(1, 4, dtype=torch.bool)),
        (torch.zeros(1, 4, 16, 16, 16), torch.ones(1, 3, dtype=torch.bool)),
    ],
)
def test_mambavitakd_rejects_invalid_modality_or_mask_shape(
    images: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    model = MambaVitAKD(num_cls=4).eval()

    with pytest.raises(RuntimeError, match="MambaVitAKD expects"):
        model(images, mask)


def test_mambavitakd_resolves_from_model_config_builder() -> None:
    model_config = build_model_config("mambavitakd", {"num_cls": 4})

    assert model_config.model_class is MambaVitAKD
    assert model_config.kwargs == {"num_cls": 4}


def test_mambavitakd_loss_resolves_from_loss_config_builder() -> None:
    loss_config = build_loss_config("mambavitakd", {"num_classes": 4})

    assert loss_config.loss_class is MambaVitAKDLoss
    assert loss_config.kwargs == {"num_classes": 4}


def test_mambavitakd_transform_manager_resolves_from_kind() -> None:
    transform_manager = build_transform_manager(
        TransformKind.IMFUSE, crop_size=(80, 80, 80)
    )

    assert isinstance(transform_manager, IMFuseTransformManager)
    assert transform_manager.crop_size == (80, 80, 80)


def test_mambavitakd_training_loss_end_to_end() -> None:
    torch.manual_seed(4)
    model = MambaVitAKD(num_cls=4)
    model.is_training = True
    images = torch.randn(1, 4, 16, 16, 16)
    mask = torch.tensor([[True, False, True, True]])
    target = _random_target(batch=1, num_classes=4, size=16)

    outputs = model(images, mask)
    loss_fn = MambaVitAKDLoss(num_classes=4)
    metrics = loss_fn.training_loss(outputs, target, include_fuse=True)

    assert torch.isfinite(metrics["loss"])
    metrics["loss"].backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_mambavitakd_training_loss_zeroes_fuse_before_region_fusion_start_epoch() -> (
    None
):
    torch.manual_seed(5)
    model = MambaVitAKD(num_cls=4)
    model.is_training = True
    images = torch.randn(1, 4, 16, 16, 16)
    mask = torch.tensor([[True, True, True, True]])
    target = _random_target(batch=1, num_classes=4, size=16)

    outputs = model(images, mask)
    loss_fn = MambaVitAKDLoss(num_classes=4)
    warmup_metrics = loss_fn.training_loss(outputs, target, include_fuse=False)
    fused_metrics = loss_fn.training_loss(outputs, target, include_fuse=True)

    assert warmup_metrics["loss"] < fused_metrics["loss"]


@pytest.mark.parametrize(
    "config_name", ["mambavitakd_18.yaml", "mambavitakd_23.yaml", "mambavitakd_25.yaml"]
)
def test_packaged_mambavitakd_configs_use_imfuse_training_contract(
    config_name: str,
) -> None:
    config = load_yaml_config(Path(config_name))

    assert config["trainer"] == "imfuse"
    assert config["model"] == "mambavitakd"
    assert config["loss"] == "mambavitakd"
    assert config["fp16"] is True
    assert config["seed"] == 999
    assert config["crop_min_size"] == (128, 128, 128)
    assert config["custom_trainer_kwargs"]["patch_size"] == 80
    assert config["custom_trainer_kwargs"]["transform_kind"] == "imfuse"
    assert config["custom_loss_kwargs"]["kd_weight"] == 10.0
    assert config["custom_loss_kwargs"]["proto_weight"] == 0.1
    assert config["custom_loss_kwargs"]["attn_weight"] == 0.1


def test_packaged_2023_configs_include_mambavitakd_for_flops_all() -> None:
    config_names = {path.name for path in resolve_2023_config_paths()}

    assert "mambavitakd_23.yaml" in config_names
