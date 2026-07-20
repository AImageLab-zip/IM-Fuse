from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F

from mimose.losses.config import (
    KwargField,
    nonneg_float,
    positive_float,
    unit_interval_float,
)
from mimose.losses.imfuse import softmax_weighted_loss


def dice_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-7,
) -> torch.Tensor:
    if prediction.ndim != 5 or target.ndim != 5:
        raise ValueError("dice_loss expects 5D tensors shaped [B, C, H, W, D]")

    prediction = prediction.float()
    target = target.float()
    reduce_dims = (0, 2, 3, 4)
    intersection = torch.sum(prediction * target, dim=reduce_dims)
    denominator = torch.sum(prediction, dim=reduce_dims) + torch.sum(
        target,
        dim=reduce_dims,
    )
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def tversky_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Tversky loss (Salehi et al., 2017): generalizes Dice with independent
    false-positive (``alpha``) and false-negative (``beta``) weights. Setting
    ``beta > alpha`` penalizes missed (false-negative) tumor voxels more than
    spurious ones, which trades a bit of precision for recall on the small,
    severely imbalanced tumor classes."""
    if prediction.ndim != 5 or target.ndim != 5:
        raise ValueError("tversky_loss expects 5D tensors shaped [B, C, H, W, D]")

    prediction = prediction.float()
    target = target.float()
    reduce_dims = (0, 2, 3, 4)
    true_pos = torch.sum(prediction * target, dim=reduce_dims)
    false_neg = torch.sum(target * (1.0 - prediction), dim=reduce_dims)
    false_pos = torch.sum(prediction * (1.0 - target), dim=reduce_dims)
    tversky = (true_pos + eps) / (
        true_pos + alpha * false_pos + beta * false_neg + eps
    )
    return 1.0 - tversky.mean()


class TinyMimosaLoss:
    KWARG_SPEC: ClassVar[dict[str, KwargField]] = {
        "dice_weight": nonneg_float(),
        "ce_weight": nonneg_float(),
        "eps": positive_float(),
    }

    def __init__(
        self,
        *,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        eps: float = 1e-7,
        num_classes
    ) -> None:
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.eps = float(eps)

    def segmentation_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        metrics = self._compute_metrics(prediction, target)
        return metrics["loss"]

    def training_loss(
        self,
        outputs: tuple[torch.Tensor, object, object],
        target: torch.Tensor,
        *,
        include_fuse: bool,
    ) -> dict[str, torch.Tensor]:
        fuse_pred, _, _ = outputs
        metrics = self._compute_metrics(fuse_pred, target)
        zero = metrics["loss"].new_tensor(0.0)
        total = metrics["loss"] if include_fuse else zero
        return {
            "loss": total,
            "fusecross": metrics["cross_entropy"] if include_fuse else zero,
            "fusedice": metrics["dice"] if include_fuse else zero,
            "sepcross": zero,
            "sepdice": zero,
            "prmcross": zero,
            "prmdice": zero,
        }

    def _compute_metrics(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target_indices = target.argmax(dim=1)
        log_prediction = torch.log(prediction.clamp_min(self.eps))
        cross_entropy = F.nll_loss(log_prediction, target_indices)
        dice = dice_loss(prediction, target, eps=self.eps)
        total = (self.ce_weight * cross_entropy) + (self.dice_weight * dice)
        return {
            "loss": total,
            "cross_entropy": cross_entropy,
            "dice": dice,
        }


class TinyMimosaWeightedLoss(TinyMimosaLoss):
    """TinyMimosaLoss variant for severely imbalanced classes: swaps the plain
    NLL cross-entropy for IMFuse's dynamic per-sample per-class weighting
    (``softmax_weighted_loss``), and the plain Dice term for a Tversky term
    (``tversky_alpha``/``tversky_beta``) that can penalize false negatives on
    the small tumor classes more than false positives."""

    KWARG_SPEC: ClassVar[dict[str, KwargField]] = {
        **TinyMimosaLoss.KWARG_SPEC,
        "log_clamp_min": unit_interval_float(),
        "tversky_alpha": unit_interval_float(),
        "tversky_beta": unit_interval_float(),
    }

    def __init__(
        self,
        *,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        eps: float = 1e-7,
        log_clamp_min: float = 0.005,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        num_classes,
    ) -> None:
        super().__init__(
            dice_weight=dice_weight,
            ce_weight=ce_weight,
            eps=eps,
            num_classes=num_classes,
        )
        self.log_clamp_min = float(log_clamp_min)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.num_classes = int(num_classes)

    def _compute_metrics(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target = target.float()
        cross_entropy = softmax_weighted_loss(
            prediction,
            target,
            num_cls=self.num_classes,
            log_clamp_min=self.log_clamp_min,
            eps=self.eps,
        )
        dice = tversky_loss(
            prediction,
            target,
            alpha=self.tversky_alpha,
            beta=self.tversky_beta,
            eps=self.eps,
        )
        total = (self.ce_weight * cross_entropy) + (self.dice_weight * dice)
        return {
            "loss": total,
            "cross_entropy": cross_entropy,
            "dice": dice,
        }


__all__ = ["TinyMimosaLoss", "TinyMimosaWeightedLoss", "dice_loss"]
