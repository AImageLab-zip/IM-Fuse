from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from mimose.losses.imfuse import IMFuseLoss, dice_loss, softmax_weighted_loss


def _load_legacy_criterions():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "legacy" / "IMFuse" / "utils" / "criterions.py"
    spec = importlib.util.spec_from_file_location("legacy_imfuse_criterions", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load legacy IMFuse criterions from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _random_probabilities(batch: int, num_classes: int, size: int) -> torch.Tensor:
    logits = torch.randn(batch, num_classes, size, size, size)
    return torch.softmax(logits, dim=1)


def _random_target(batch: int, num_classes: int, size: int) -> torch.Tensor:
    labels = torch.randint(0, num_classes, (batch, size, size, size))
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def test_imfuse_losses_match_legacy_criterions() -> None:
    torch.manual_seed(7)
    legacy = _load_legacy_criterions()

    num_classes = 4
    target = _random_target(batch=2, num_classes=num_classes, size=8)
    fuse_pred = _random_probabilities(batch=2, num_classes=num_classes, size=8)
    sep_preds = [
        _random_probabilities(batch=2, num_classes=num_classes, size=8)
        for _ in range(4)
    ]
    prm_preds = [
        _random_probabilities(batch=2, num_classes=num_classes, size=8)
        for _ in range(3)
    ]

    new_loss = IMFuseLoss(num_classes=num_classes)

    assert torch.allclose(
        softmax_weighted_loss(fuse_pred, target, num_cls=num_classes),
        legacy.softmax_weighted_loss(fuse_pred, target, num_cls=num_classes),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        dice_loss(fuse_pred, target, num_cls=num_classes),
        legacy.dice_loss(fuse_pred, target, num_cls=num_classes),
        atol=1e-6,
        rtol=1e-6,
    )

    metrics = new_loss.training_loss(
        (fuse_pred, sep_preds, prm_preds),
        target,
        include_fuse=True,
    )

    legacy_fusecross = legacy.softmax_weighted_loss(fuse_pred, target, num_cls=num_classes)
    legacy_fusedice = legacy.dice_loss(fuse_pred, target, num_cls=num_classes)
    legacy_sepcross = sum(
        legacy.softmax_weighted_loss(pred, target, num_cls=num_classes)
        for pred in sep_preds
    )
    legacy_sepdice = sum(
        legacy.dice_loss(pred, target, num_cls=num_classes)
        for pred in sep_preds
    )
    legacy_prmcross = sum(
        legacy.softmax_weighted_loss(pred, target, num_cls=num_classes)
        for pred in prm_preds
    )
    legacy_prmdice = sum(
        legacy.dice_loss(pred, target, num_cls=num_classes)
        for pred in prm_preds
    )
    legacy_total = (
        legacy_fusecross
        + legacy_fusedice
        + legacy_sepcross
        + legacy_sepdice
        + legacy_prmcross
        + legacy_prmdice
    )

    assert torch.allclose(metrics["fusecross"], legacy_fusecross, atol=1e-6, rtol=1e-6)
    assert torch.allclose(metrics["fusedice"], legacy_fusedice, atol=1e-6, rtol=1e-6)
    assert torch.allclose(metrics["sepcross"], legacy_sepcross, atol=1e-6, rtol=1e-6)
    assert torch.allclose(metrics["sepdice"], legacy_sepdice, atol=1e-6, rtol=1e-6)
    assert torch.allclose(metrics["prmcross"], legacy_prmcross, atol=1e-6, rtol=1e-6)
    assert torch.allclose(metrics["prmdice"], legacy_prmdice, atol=1e-6, rtol=1e-6)
    assert torch.allclose(metrics["loss"], legacy_total, atol=1e-6, rtol=1e-6)


def test_imfuse_training_loss_respects_region_fusion_flag() -> None:
    torch.manual_seed(11)
    num_classes = 4
    loss = IMFuseLoss(num_classes=num_classes)
    target = _random_target(batch=1, num_classes=num_classes, size=6)
    fuse_pred = _random_probabilities(batch=1, num_classes=num_classes, size=6)
    sep_preds = [_random_probabilities(batch=1, num_classes=num_classes, size=6)]
    prm_preds = [_random_probabilities(batch=1, num_classes=num_classes, size=6)]

    with_fuse = loss.training_loss(
        (fuse_pred, sep_preds, prm_preds),
        target,
        include_fuse=True,
    )
    without_fuse = loss.training_loss(
        (fuse_pred, sep_preds, prm_preds),
        target,
        include_fuse=False,
    )

    expected_delta = with_fuse["fusecross"] + with_fuse["fusedice"]
    assert torch.allclose(
        with_fuse["loss"] - without_fuse["loss"],
        expected_delta,
        atol=1e-6,
        rtol=1e-6,
    )
