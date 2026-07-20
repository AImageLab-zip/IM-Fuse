from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from mimose.enums import TransformKind
from mimose.losses.config import build_loss_config
from mimose.losses.m3ae import M3AELoss, m3ae_reconstruction_loss
from mimose.models.config import build_model_config
from mimose.models.m3ae import M3AE, input_patch_size
from mimose.training.config import OptimizerConfig, SchedulerConfig
from mimose.training.trainers.m3ae import M3AETrainer
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


def test_m3ae_reconstruct_returns_input_shape() -> None:
    torch.manual_seed(0)
    model = M3AE(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.tensor([[True, True, False, True]])

    with torch.no_grad():
        recon = model.reconstruct(images, mask)

    assert recon.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert torch.isfinite(recon).all()


def test_m3ae_forward_mode_reconstruct_matches_reconstruct_method() -> None:
    torch.manual_seed(0)
    model = M3AE(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        via_forward = model(images, mask, mode="reconstruct")
        via_method = model.reconstruct(images, mask)

    assert torch.equal(via_forward, via_method)


def test_m3ae_forward_default_mode_is_unaffected_by_recon_head() -> None:
    torch.manual_seed(0)
    model = M3AE(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert output.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert torch.allclose(output.sum(dim=1), torch.ones_like(output.sum(dim=1)), atol=1e-5)


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
    assert config["custom_trainer_kwargs"]["pretrain_fraction"] == 0.15
    assert config["custom_trainer_kwargs"]["limage_lr"] == 0.005
    assert config["custom_trainer_kwargs"]["limage_reg_weight"] == 0.005


def test_m3ae_reconstruction_loss_combines_mse_and_smoothness() -> None:
    torch.manual_seed(1)
    output = torch.randn(1, 4, 8, 8, 8)
    target = torch.randn(1, 4, 8, 8, 8)
    limage = torch.randn(1, 4, 8, 8, 8, requires_grad=True)

    metrics = m3ae_reconstruction_loss(output, target, limage, reg_weight=0.005)

    expected_recon = torch.nn.functional.mse_loss(output, target)
    expected_smoothness = torch.norm(limage - limage.mean(dim=(2, 3, 4), keepdim=True), p=2)
    assert torch.allclose(metrics["recon"], expected_recon)
    assert torch.allclose(metrics["smoothness"], expected_smoothness)
    assert torch.allclose(metrics["loss"], expected_recon + 0.005 * expected_smoothness)


def _new_m3ae_trainer_stub(*, pretrain_epochs: int, num_epochs: int = 10) -> M3AETrainer:
    """Constructs a bare, uninitialized ``M3AETrainer`` (bypassing
    ``BaseTrainer.__init__``, which needs real datasets/dataloaders) with
    just the attributes ``_apply_training_stage``/``_build_optimizer``
    touch, so the stage-switch logic can be unit tested in isolation."""
    trainer = M3AETrainer.__new__(M3AETrainer)
    trainer.model = M3AE(num_cls=4)
    trainer.pretrain_epochs = pretrain_epochs
    trainer.limage_lr = 0.005
    trainer.limage_reg_weight = 0.005
    trainer.num_epochs = num_epochs
    trainer._in_pretrain = None
    trainer.optimizer_config = OptimizerConfig(optim_class=torch.optim.Adam, lr=1e-3, weight_decay=1e-5)
    trainer.scheduler_config = SchedulerConfig(scheduler_class=CosineAnnealingLR, kwargs={"T_max": num_epochs, "eta_min": 0.0})
    trainer.optimizer = trainer._build_optimizer()
    trainer.scheduler = None
    return trainer


def test_m3ae_apply_training_stage_is_noop_when_disabled() -> None:
    trainer = _new_m3ae_trainer_stub(pretrain_epochs=0)

    trainer._apply_training_stage(0)

    assert trainer._in_pretrain is None
    assert trainer.model.limage.requires_grad is True


def test_m3ae_apply_training_stage_toggles_limage_requires_grad() -> None:
    trainer = _new_m3ae_trainer_stub(pretrain_epochs=3, num_epochs=10)

    trainer._apply_training_stage(0)
    assert trainer._in_pretrain is True
    assert trainer.model.limage.requires_grad is True

    trainer._apply_training_stage(2)
    assert trainer._in_pretrain is True

    trainer._apply_training_stage(3)
    assert trainer._in_pretrain is False
    assert trainer.model.limage.requires_grad is False


def test_m3ae_apply_training_stage_restarts_scheduler_entering_finetune() -> None:
    trainer = _new_m3ae_trainer_stub(pretrain_epochs=3, num_epochs=10)
    trainer._apply_training_stage(0)
    original_scheduler = trainer.scheduler = trainer._build_scheduler(trainer.optimizer)
    for _ in range(2):
        trainer.scheduler.step()

    trainer._apply_training_stage(3)

    assert trainer.scheduler is not original_scheduler
    assert trainer.scheduler.T_max == 7
    assert trainer.scheduler.last_epoch == 0
    # scheduler_config.kwargs is restored to its original (YAML-declared)
    # T_max after the rebuild, so logging/checkpointing still reflects the
    # configured value rather than the transient "remaining epochs" figure.
    assert trainer.scheduler_config.kwargs["T_max"] == 10


def test_m3ae_build_optimizer_uses_single_group_when_pretrain_disabled() -> None:
    trainer = _new_m3ae_trainer_stub(pretrain_epochs=0)

    assert len(trainer.optimizer.param_groups) == 1


def test_m3ae_build_optimizer_splits_limage_into_own_group_when_enabled() -> None:
    trainer = _new_m3ae_trainer_stub(pretrain_epochs=3)

    assert len(trainer.optimizer.param_groups) == 2
    limage_group = next(g for g in trainer.optimizer.param_groups if g["lr"] == trainer.limage_lr)
    assert limage_group["params"] == [trainer.model.limage]
    other_group = next(g for g in trainer.optimizer.param_groups if g is not limage_group)
    assert not any(param is trainer.model.limage for param in other_group["params"])
