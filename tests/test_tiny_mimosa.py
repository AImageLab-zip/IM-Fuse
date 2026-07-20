from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mimose.enums import TransformKind
from mimose.models.config import build_model_config
from mimose.models.config import ModelConfig
from mimose.models.tiny_mimosa import TinyMimosa
from mimose.training.transforms import build_transform_manager
from mimose.training.trainers.imfuse import IMFuseTrainer
import mimose.training.trainers.imfuse as imfuse_trainer_module


def build_test_model() -> TinyMimosa:
    model = TinyMimosa(
        num_cls=4,
        input_shape=(16, 16, 16),
        features_per_stage=(4, 8, 16, 32),
    ).eval()
    return model


def test_tiny_mimosa_inference_forward_returns_segmentation_shape() -> None:
    model = build_test_model()
    images = torch.zeros(1, 4, 16, 16, 16)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 4, 16, 16, 16)


def test_tiny_mimosa_forward_returns_only_final_prediction() -> None:
    model = build_test_model()
    model.is_training = True
    images = torch.zeros(1, 4, 16, 16, 16)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 4, 16, 16, 16)


def test_tiny_mimosa_predict_handles_exact_tile_shape() -> None:
    model = build_test_model()
    images = torch.zeros(1, 4, 16, 16, 16)
    mask = torch.tensor([[True, False, True, True]])

    with torch.no_grad():
        output = model.predict(images, mask)

    assert output.shape == (1, 4, 16, 16, 16)


def test_tiny_mimosa_predict_tiles_larger_inputs_back_to_original_shape() -> None:
    model = build_test_model()
    images = torch.zeros(1, 4, 24, 26, 24)
    mask = torch.tensor([[True, True, False, True]])

    with torch.no_grad():
        output = model.predict(images, mask)

    assert output.shape == (1, 4, 24, 26, 24)


def test_tiny_mimosa_preserves_dataset_modality_order() -> None:
    images = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1)
    mask = torch.tensor([[False, True, False, True]])

    remapped_images, remapped_mask = TinyMimosa._remap_input_order(images, mask)

    assert remapped_images.flatten().tolist() == [0.0, 1.0, 2.0, 3.0]
    assert remapped_mask.tolist() == [[False, True, False, True]]


def test_tiny_mimosa_forward_rejects_unpadded_shape() -> None:
    model = build_test_model()
    images = torch.zeros(1, 4, 14, 15, 13)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with pytest.raises(RuntimeError, match="expects spatial shape"):
        model(images, mask)


def test_tiny_mimosa_rounds_tile_shape_up_to_decoder_multiple() -> None:
    model = TinyMimosa(
        num_cls=4,
        input_shape=(182, 218, 182),
        features_per_stage=(8, 16, 16, 32),
    )

    assert model.spatial_multiple == 8
    assert model.tile_shape == (184, 224, 184)


def test_tinymimosa_transform_pads_to_tile_shape() -> None:
    transform_manager = build_transform_manager(
        TransformKind.TINYMIMOSA,
        model_kwargs={
            "input_shape": (182, 218, 182),
            "features_per_stage": (8, 16, 16, 32),
        },
    )
    images = torch.zeros(4, 182, 218, 182)
    labels = torch.zeros(1, 182, 218, 182, dtype=torch.long)

    transformed_images, transformed_labels = transform_manager(images, labels, mode="test")

    assert transformed_images.shape == (4, 184, 224, 184)
    assert transformed_labels.shape == (1, 184, 224, 184)


def test_imfuse_trainer_uses_initialized_transform_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeDataset:
        def __init__(self, **kwargs) -> None:
            captured.setdefault("datasets", []).append(kwargs)

    def initialized_transform_manager(images, labels, mode="train"):
        captured.setdefault("modes", []).append(mode)
        return images, labels

    monkeypatch.setattr(
        imfuse_trainer_module,
        "build_transform_manager",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not rebuild transforms")),
    )
    monkeypatch.setattr(imfuse_trainer_module, "IMFuseDataset", FakeDataset)

    trainer = object.__new__(IMFuseTrainer)
    trainer.train_split = [{"sub": "train"}]
    trainer.val_split = [{"sub": "val"}]
    trainer.transform_kind = TransformKind.IMFUSE
    trainer.transform_manager = initialized_transform_manager
    trainer.model_config = ModelConfig(
        model_class=TinyMimosa,
        kwargs={"input_shape": (182, 218, 182), "features_per_stage": (8, 16, 16, 32)},
    )
    trainer.input_dir = Path("/tmp")
    trainer.train_masking_mode = None
    trainer.val_masking_mode = None

    train_set, val_set = IMFuseTrainer.build_datasets(trainer)

    assert isinstance(train_set, FakeDataset)
    assert isinstance(val_set, FakeDataset)
    train_transform = captured["datasets"][0]["sample_transform"]
    val_transform = captured["datasets"][1]["sample_transform"]
    train_transform(torch.zeros(4, 8, 8, 8), torch.zeros(1, 8, 8, 8))
    val_transform(torch.zeros(4, 8, 8, 8), torch.zeros(1, 8, 8, 8))
    assert captured["modes"] == ["train", "test"]


def test_tiny_mimosa_resolves_from_model_config_builder() -> None:
    model_config = build_model_config("tinymimosa", {"num_cls": 4})

    assert model_config.model_class is TinyMimosa
    assert model_config.kwargs == {"num_cls": 4}


def test_tiny_mimosa_window_starts_cover_full_axis() -> None:
    assert TinyMimosa._window_starts(16, 16, 0.5) == [0]
    assert TinyMimosa._window_starts(25, 16, 0.5) == [0, 8, 9]

    with pytest.raises(RuntimeError, match="at least 16"):
        TinyMimosa._window_starts(15, 16, 0.5)
