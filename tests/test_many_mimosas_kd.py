from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from mimose.checkpoints import load_weights_only_checkpoint, save_weights_only_checkpoint
from mimose.losses.config import build_loss_config
from mimose.losses.many_mimosas_kd import ManyMimosasKDLoss
from mimose.models.config import build_model_config
from mimose.models.many_mimosas import ALL_MASK_PATTERNS, ManyMimosas
from mimose.models.many_mimosas_kd import ManyMimosasKD


def build_test_model() -> ManyMimosasKD:
    return ManyMimosasKD(
        num_cls=4,
        input_shape=(16, 16, 16),
        student_features_per_stage=(4, 8, 16, 32),
        teacher_features_per_stage=(8, 16, 32, 64),
    )


def test_many_mimosas_kd_has_one_submodel_per_pattern_sized_to_its_channels() -> None:
    model = build_test_model().eval()
    assert len(model.submodels) == 15
    for pattern in ALL_MASK_PATTERNS:
        key = "".join("1" if bool(v) else "0" for v in pattern.tolist())
        assert model.submodels[key].num_modals == int(pattern.sum().item())


def test_many_mimosas_kd_teacher_is_wider_than_students() -> None:
    model = build_test_model()
    assert model.teacher.features_per_stage == (8, 16, 32, 64)
    for submodel in model.submodels.values():
        assert submodel.features_per_stage == (4, 8, 16, 32)


def test_many_mimosas_kd_forward_routes_mixed_batch_when_not_training() -> None:
    model = build_test_model().eval()
    images = torch.randn(2, 4, 16, 16, 16)
    mask = torch.tensor([[True, False, False, False], [True, True, True, True]])

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 4, 16, 16, 16)


def test_many_mimosas_kd_predict_tiles_larger_inputs_back_to_original_shape() -> None:
    model = build_test_model().eval()
    images = torch.zeros(1, 4, 24, 26, 24)
    mask = torch.tensor([[True, True, False, True]])

    with torch.no_grad():
        output = model.predict(images, mask)

    assert output.shape == (1, 4, 24, 26, 24)


def test_many_mimosas_kd_teacher_pretrain_forward() -> None:
    model = build_test_model()
    model.is_training = True
    images = torch.zeros(1, 4, 16, 16, 16)
    mask = torch.tensor([[True, False, True, False]])

    with torch.no_grad():
        output = model(images, mask, mode="teacher_pretrain")

    assert isinstance(output, tuple)
    fuse_pred, aux1, aux2 = output
    assert fuse_pred.shape == (1, 4, 16, 16, 16)
    assert aux1 == () and aux2 == ()


def test_many_mimosas_kd_teacher_forward_hints_shapes() -> None:
    model = build_test_model()
    images = torch.zeros(1, 4, 16, 16, 16)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        hints = model(images, mask, mode="teacher_forward")

    assert [f.shape[1] for f in hints.encoder_feats] == [8, 16, 32, 64]
    assert [f.shape[1] for f in hints.decoder_feats] == [32, 16, 8]
    assert hints.logits.shape == (1, 4, 16, 16, 16)


def test_many_mimosas_kd_distillation_forward_returns_hint_and_kd_losses() -> None:
    model = build_test_model()
    model.is_training = True
    images = torch.randn(1, 4, 16, 16, 16)
    full_mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        teacher_hints = model(images, full_mask, mode="teacher_forward")

    pattern = ALL_MASK_PATTERNS[0].unsqueeze(0)
    fuse_pred, hint_loss, kd_loss = model(images, pattern, teacher_hints=teacher_hints)

    assert fuse_pred.shape == (1, 4, 16, 16, 16)
    assert isinstance(hint_loss, torch.Tensor) and hint_loss.ndim == 0
    assert isinstance(kd_loss, torch.Tensor) and kd_loss.ndim == 0
    assert hint_loss.requires_grad and kd_loss.requires_grad

    (hint_loss + kd_loss).backward()


def test_many_mimosas_kd_model_for_export_is_plain_many_mimosas() -> None:
    model = build_test_model()
    export = model.model_for_export()

    assert isinstance(export, ManyMimosas)
    assert set(export.submodels.state_dict().keys()) == set(model.submodels.state_dict().keys())

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "final_weights_only.safetensors"
        save_weights_only_checkpoint(export, checkpoint_path)

        fresh = ManyMimosas(
            num_cls=4, input_shape=(16, 16, 16), features_per_stage=(4, 8, 16, 32)
        )
        load_weights_only_checkpoint(fresh, checkpoint_path, strict=True)


def test_many_mimosas_kd_resolves_from_model_config_builder() -> None:
    model_config = build_model_config("manymimosaskd", {"num_cls": 4})

    assert model_config.model_class is ManyMimosasKD
    assert model_config.kwargs == {"num_cls": 4}


def test_many_mimosas_kd_loss_resolves_from_loss_config_builder() -> None:
    loss_config = build_loss_config("manymimosaskd", {"hint_weight": 2.0})

    assert loss_config.loss_class is ManyMimosasKDLoss
    assert loss_config.kwargs["hint_weight"] == 2.0
