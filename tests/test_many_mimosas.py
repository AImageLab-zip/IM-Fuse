from __future__ import annotations

import torch

from mimose.models.config import build_model_config
from mimose.models.many_mimosas import ALL_MASK_PATTERNS, ManyMimosas


def build_test_model() -> ManyMimosas:
    return ManyMimosas(
        num_cls=4,
        input_shape=(16, 16, 16),
        features_per_stage=(4, 8, 16, 32),
    ).eval()


def test_all_mask_patterns_covers_every_non_empty_combination() -> None:
    assert ALL_MASK_PATTERNS.shape == (15, 4)
    assert (ALL_MASK_PATTERNS.sum(dim=1) >= 1).all()
    unique_rows = {tuple(row.tolist()) for row in ALL_MASK_PATTERNS}
    assert len(unique_rows) == 15


def test_many_mimosas_has_one_submodel_per_pattern_sized_to_its_channels() -> None:
    model = build_test_model()
    assert len(model.submodels) == 15
    for pattern in ALL_MASK_PATTERNS:
        key = "".join("1" if bool(v) else "0" for v in pattern.tolist())
        assert model.submodels[key].num_modals == int(pattern.sum().item())


def test_many_mimosas_forward_routes_uniform_batch() -> None:
    model = build_test_model()
    images = torch.zeros(2, 4, 16, 16, 16)
    mask = torch.tensor([[True, False, True, True], [True, False, True, True]])

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 4, 16, 16, 16)


def test_many_mimosas_forward_routes_mixed_batch() -> None:
    model = build_test_model()
    images = torch.randn(2, 4, 16, 16, 16)
    mask = torch.tensor([[True, False, False, False], [True, True, True, True]])

    with torch.no_grad():
        output = model(images, mask)

    assert output.shape == (2, 4, 16, 16, 16)


def test_many_mimosas_forward_returns_fuse_pred_tuple_when_training() -> None:
    model = build_test_model()
    model.is_training = True
    images = torch.zeros(1, 4, 16, 16, 16)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, tuple)
    fuse_pred, aux1, aux2 = output
    assert isinstance(fuse_pred, torch.Tensor)
    assert fuse_pred.shape == (1, 4, 16, 16, 16)
    assert aux1 == () and aux2 == ()


def test_many_mimosas_predict_tiles_larger_inputs_back_to_original_shape() -> None:
    model = build_test_model()
    images = torch.zeros(1, 4, 24, 26, 24)
    mask = torch.tensor([[True, True, False, True]])

    with torch.no_grad():
        output = model.predict(images, mask)

    assert output.shape == (1, 4, 24, 26, 24)


def test_many_mimosas_rejects_all_missing_mask() -> None:
    model = build_test_model()
    images = torch.zeros(1, 4, 16, 16, 16)
    mask = torch.zeros(1, 4, dtype=torch.bool)

    try:
        model(images, mask)
    except RuntimeError as exc:
        assert "at least one available modality" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for an all-missing mask")


def test_many_mimosas_resolves_from_model_config_builder() -> None:
    model_config = build_model_config("manymimosas", {"num_cls": 4})

    assert model_config.model_class is ManyMimosas
    assert model_config.kwargs == {"num_cls": 4}
