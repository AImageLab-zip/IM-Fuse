from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mimose.enums import TransformKind
from mimose.losses.clrs import CLRSLoss
from mimose.losses.config import build_loss_config
from mimose.models.clrs import CLRS, input_patch_size, num_modals
from mimose.training.trainers import CLRSTrainer
from mimose.training.trainers.clrs import _GModel
from mimose.training.transforms import (
    IMFuseTransformManager,
    build_transform_manager,
)
from mimose.utils.cli_overrides import load_yaml_config


def _build_bare_trainer(tmp_path: Path, *, use_cyclic_synthesis: bool) -> CLRSTrainer:
    trainer = CLRSTrainer.__new__(CLRSTrainer)
    trainer.output_dir = tmp_path
    trainer.checkpoint_dir = tmp_path
    trainer.rank = 0
    trainer.device = torch.device("cpu")
    trainer.model = CLRS(num_cls=4)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)
    trainer.scheduler = None
    trainer.best_val_loss = float("inf")
    trainer.best_val_dice = float("-inf")
    trainer.use_cyclic_synthesis = use_cyclic_synthesis
    trainer.synthesis_generators = None
    trainer.synthesis_optimizer = None
    if use_cyclic_synthesis:
        trainer.synthesis_generators = torch.nn.ModuleList(
            [_GModel() for _ in range(num_modals)]
        )
        trainer.synthesis_optimizer = torch.optim.Adam(
            trainer.synthesis_generators.parameters(), lr=1e-4
        )
    return trainer


def test_clrs_inference_forward_returns_segmentation_shape() -> None:
    model = CLRS(num_cls=4).eval()
    images = torch.zeros(1, 4, 32, 32, 32)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 4, 32, 32, 32)
    assert torch.allclose(output.sum(dim=1), torch.ones_like(output.sum(dim=1)), atol=1e-4)


def test_clrs_training_forward_returns_seg_spec_and_generated_outputs() -> None:
    model = CLRS(num_cls=4).eval()
    model.is_training = True
    images = torch.zeros(1, 4, 32, 32, 32)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        fused_logits, spec_logits_cat, spec_info_vector, generated_map = model(images, mask)

    assert fused_logits.shape == (1, 4, 32, 32, 32)
    assert spec_logits_cat.shape == (4, 1, 4)
    assert spec_info_vector.shape == (4, 1, 512)
    assert len(generated_map) == 4
    assert all(g.shape[0] == 1 and g.shape[1] == 64 for g in generated_map)


def test_clrs_handles_dropped_modalities() -> None:
    model = CLRS(num_cls=4).eval()
    images = torch.randn(2, 4, 32, 32, 32)
    mask = torch.ones(2, 4, dtype=torch.bool)
    mask[0, 1] = False
    mask[1, 2] = False

    with torch.no_grad():
        output = model(images, mask)

    assert output.shape == (2, 4, 32, 32, 32)
    assert torch.isfinite(output).all()


def test_clrs_remaps_dataset_modalities_to_legacy_order() -> None:
    images = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1)
    mask = torch.tensor([[False, True, False, True]])

    remapped_images, remapped_mask = CLRS._remap_input_order(images, mask)

    assert remapped_images.flatten().tolist() == [2.0, 0.0, 1.0, 3.0]
    assert remapped_mask.tolist() == [[False, False, True, True]]


def test_clrs_window_starts_use_50_percent_overlap() -> None:
    assert CLRS._window_starts(input_patch_size) == [0]
    assert CLRS._window_starts(input_patch_size + 40) == [0, 40]
    assert CLRS._window_starts(input_patch_size + 62) == [0, 40, 62]

    with pytest.raises(RuntimeError, match="at least 80"):
        CLRS._window_starts(input_patch_size - 1)


@pytest.mark.parametrize(
    ("images", "mask"),
    [
        (torch.zeros(1, 3, 32, 32, 32), torch.ones(1, 4, dtype=torch.bool)),
        (torch.zeros(1, 4, 32, 32, 32), torch.ones(1, 3, dtype=torch.bool)),
    ],
)
def test_clrs_rejects_invalid_modality_or_mask_shape(
    images: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    model = CLRS(num_cls=4).eval()

    with pytest.raises(RuntimeError, match="CLRS expects"):
        model(images, mask)


def test_clrs_loss_resolves_from_loss_config_builder() -> None:
    loss_config = build_loss_config("clrs", {"num_classes": 4})

    assert loss_config.loss_class is CLRSLoss
    assert loss_config.kwargs == {"num_classes": 4}


def test_clrs_loss_training_loss_masks_absent_modalities() -> None:
    model = CLRS(num_cls=4).eval()
    model.is_training = True
    images = torch.randn(2, 4, 32, 32, 32)
    mask = torch.ones(2, 4, dtype=torch.bool)
    mask[0, 1] = False
    target = torch.nn.functional.one_hot(
        torch.randint(0, 4, (2, 32, 32, 32)), num_classes=4
    ).permute(0, 4, 1, 2, 3).float()

    with torch.no_grad():
        outputs = model(images, mask)

    loss_fn = CLRSLoss(num_cls=4)
    metrics = loss_fn.training_loss(outputs, target, mask=mask)

    assert torch.isfinite(metrics["loss"])
    assert metrics["loss"].ndim == 0


def test_clrs_synthesis_generator_cycle_produces_matching_shapes() -> None:
    generators = [_GModel(in_channels=8, base_channels=8, p_dim=8) for _ in range(4)]
    slices = [torch.randn(2, 8, 16, 16, 16) for _ in range(4)]

    current = slices[0]
    for generator in generators:
        prediction = generator(current)
        assert prediction.shape == current.shape
        current = prediction


def test_clrs_transform_manager_resolves_from_kind() -> None:
    transform_manager = build_transform_manager(TransformKind.IMFUSE, crop_size=(80, 80, 80))

    assert isinstance(transform_manager, IMFuseTransformManager)
    assert transform_manager.crop_size == (80, 80, 80)


def test_clrs_trainer_is_a_base_trainer_subclass() -> None:
    from mimose.training.trainers.base_trainer import BaseTrainer

    assert issubclass(CLRSTrainer, BaseTrainer)


@pytest.mark.parametrize("config_name", ["clrs_18.yaml", "clrs_23.yaml", "clrs_25.yaml"])
def test_packaged_clrs_configs_use_expected_training_contract(config_name: str) -> None:
    config = load_yaml_config(Path(config_name))

    assert config["trainer"] == "clrs"
    assert config["model"] == "clrs"
    assert config["loss"] == "clrs"
    assert config["lr"] == 0.0002
    assert config["weight_decay"] == 0.0001
    assert config["num_epochs"] == 200
    assert config["fp16"] is True
    assert config["seed"] == 1024
    assert config["crop_min_size"] == (128, 128, 128)
    assert config["custom_trainer_kwargs"]["patch_size"] == 80
    assert config["custom_trainer_kwargs"]["transform_kind"] == "imfuse"
    assert config["custom_trainer_kwargs"]["use_cyclic_synthesis"] is True


def test_clrs_checkpoint_round_trips_synthesis_generator_state(tmp_path: Path) -> None:
    trainer = _build_bare_trainer(tmp_path, use_cyclic_synthesis=True)
    with torch.no_grad():
        for param in trainer.synthesis_generators.parameters():
            param.fill_(0.5)

    last_path = trainer.save_checkpoint(epoch=3)

    assert last_path.with_stem(last_path.stem + "_synthesis").is_file()

    other_trainer = _build_bare_trainer(tmp_path, use_cyclic_synthesis=True)
    start_epoch = other_trainer.load_checkpoint(last_path)

    assert start_epoch == 4
    for param in other_trainer.synthesis_generators.parameters():
        assert torch.allclose(param, torch.full_like(param, 0.5))


def test_clrs_checkpoint_load_without_synthesis_companion_warns_and_continues(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    writer_trainer = _build_bare_trainer(tmp_path, use_cyclic_synthesis=False)
    last_path = writer_trainer.save_checkpoint(epoch=1)

    assert not last_path.with_stem(last_path.stem + "_synthesis").is_file()

    reader_trainer = _build_bare_trainer(tmp_path, use_cyclic_synthesis=True)
    with caplog.at_level("WARNING"):
        start_epoch = reader_trainer.load_checkpoint(last_path)

    assert start_epoch == 2
    assert "no companion synthesis checkpoint" in caplog.text
