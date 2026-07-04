from __future__ import annotations

from typing import ClassVar

import torch

from mimose.losses.config import KwargField, positive_int, unit_interval_float
from mimose.losses.imfuse import dice_loss, softmax_weighted_loss


class M3AELoss:
    """Loss wrapper matching legacy M3AE's fine-tuning objective (``EDiceLoss``):
    a plain softmax cross-entropy + Dice segmentation loss, with no auxiliary
    terms. (Legacy also adds a cross-view consistency MSE term during
    fine-tuning; this port doesn't reproduce it, see ``models/m3ae.py``.)
    """

    KWARG_SPEC: ClassVar[dict[str, KwargField]] = {
        "num_classes": positive_int(),
        "log_clamp_min": unit_interval_float(),
    }

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        num_cls: int | None = None,
        log_clamp_min: float = 0.005,
    ) -> None:
        resolved_num_classes = (
            int(num_classes) if num_classes is not None else int(num_cls) if num_cls is not None else 4
        )
        self.num_classes = resolved_num_classes
        self.log_clamp_min = float(log_clamp_min)

    def segmentation_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.softmax_weighted_loss(output, target) + self.dice_loss(output, target)

    def dice_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss(output, target, num_cls=self.num_classes, eps=1e-7)

    def softmax_weighted_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return softmax_weighted_loss(output, target, num_cls=self.num_classes, log_clamp_min=self.log_clamp_min)

    def training_loss(self, output: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        cross = self.softmax_weighted_loss(output, target)
        dice = self.dice_loss(output, target)
        return {"loss": cross + dice, "cross": cross, "dice": dice}


__all__ = ["M3AELoss"]
