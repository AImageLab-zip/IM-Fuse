from __future__ import annotations

from typing import ClassVar

import torch

from mimose.losses.config import KwargField, nonneg_float, positive_float, positive_int, unit_interval_float
from mimose.losses.imfuse import dice_loss, softmax_weighted_loss


class ManyMimosasKDLoss:
    """Segmentation loss for ``ManyMimosasKD``: Dice+CE on the routed
    student prediction, plus (during the distillation phase) a weighted
    sum of the per-stage cosine hint loss and the logit-KD term that
    ``ManyMimosasKD.forward`` computes internally (it needs the model's own
    intermediate features, the same reason ``LCKDLoss`` leaves its ``kd``
    term to be computed inside ``LCKD.forward``).
    """

    KWARG_SPEC: ClassVar[dict[str, KwargField]] = {
        "num_classes": positive_int(),
        "hint_weight": nonneg_float(),
        "kd_weight": nonneg_float(),
        "eps": positive_float(),
        "log_clamp_min": unit_interval_float(),
    }

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        num_cls: int | None = None,
        hint_weight: float = 1.0,
        kd_weight: float = 0.3,
        eps: float = 1e-7,
        log_clamp_min: float = 0.005,
    ) -> None:
        resolved_num_classes = (
            int(num_classes)
            if num_classes is not None
            else int(num_cls) if num_cls is not None else 4
        )
        self.num_classes = resolved_num_classes
        self.hint_weight = float(hint_weight)
        self.kd_weight = float(kd_weight)
        self.eps = float(eps)
        self.log_clamp_min = float(log_clamp_min)

    def segmentation_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.softmax_weighted_loss(output, target) + self.dice_loss(output, target)

    def dice_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss(output, target, num_cls=self.num_classes, eps=self.eps)

    def softmax_weighted_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return softmax_weighted_loss(
            output, target, num_cls=self.num_classes, log_clamp_min=self.log_clamp_min, eps=self.eps
        )

    def teacher_pretrain_loss(
        self,
        teacher_out: tuple[torch.Tensor, tuple, tuple],
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fuse_pred, _, _ = teacher_out
        cross = self.softmax_weighted_loss(fuse_pred, target)
        dice = self.dice_loss(fuse_pred, target)
        return {"loss": cross + dice, "cross": cross, "dice": dice}

    def training_loss(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fuse_pred, hint_loss, kd_loss = outputs

        cross = self.softmax_weighted_loss(fuse_pred, target)
        dice = self.dice_loss(fuse_pred, target)
        seg = cross + dice

        zero = seg.new_tensor(0.0)
        hint = hint_loss if isinstance(hint_loss, torch.Tensor) else zero
        kd = kd_loss if isinstance(kd_loss, torch.Tensor) else zero
        total = seg + self.hint_weight * hint + self.kd_weight * kd

        return {
            "loss": total,
            "cross": cross,
            "dice": dice,
            "hint": hint,
            "kd": kd,
        }


__all__ = ["ManyMimosasKDLoss"]
