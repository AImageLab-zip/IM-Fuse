from __future__ import annotations

import torch
import torch.nn.functional as F


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


class TinyMimosaLoss:
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


__all__ = ["TinyMimosaLoss", "dice_loss"]
