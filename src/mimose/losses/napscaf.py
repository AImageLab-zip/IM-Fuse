from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F

from mimose.losses.config import KwargField, nonneg_float, positive_int


def soft_dice_loss(probabilities: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Legacy ``SoftDiceLoss``: per-class Dice over the whole batch (background included), averaged over classes."""
    dims = (0, *range(2, probabilities.ndim))
    intersection = torch.sum(probabilities * target, dims)
    denominator = torch.sum(probabilities + target, dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


class NAPSCAFLoss:
    """Legacy NAP-SCAF segmentation loss: ``ce_weight * CE + dice_weight * soft Dice`` on softmax outputs.

    Legacy's reconstruction/KL terms only apply when the model returns a dict,
    which it never does, so they are not ported.
    """

    KWARG_SPEC: ClassVar[dict[str, KwargField]] = {
        "num_classes": positive_int(),
        "ce_weight": nonneg_float(),
        "dice_weight": nonneg_float(),
    }

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        num_cls: int | None = None,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        self.num_classes = (
            int(num_classes)
            if num_classes is not None
            else int(num_cls) if num_cls is not None else 4
        )
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)

    def segmentation_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Validation loss on probabilities (``predict`` output) and one-hot targets."""
        probabilities = output.float()
        target = target.float()
        ce = -(target * probabilities.clamp_min(1e-7).log()).sum(dim=1).mean()
        return self.ce_weight * ce + self.dice_weight * soft_dice_loss(probabilities, target)

    def training_loss(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """``logits`` [N, C, H, W] and ``target`` class labels [N, H, W] for a batch of 2D slices."""
        logits = logits.float()
        target = target.long()
        ce = F.cross_entropy(logits, target)
        one_hot = F.one_hot(target, self.num_classes).movedim(-1, 1).float()
        dice = soft_dice_loss(logits.softmax(dim=1), one_hot)
        return {
            "loss": self.ce_weight * ce + self.dice_weight * dice,
            "ce": ce,
            "dice": dice,
        }


__all__ = ["NAPSCAFLoss", "soft_dice_loss"]
