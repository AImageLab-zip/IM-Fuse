from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F

from mimose.losses.config import KwargField, positive_int, unit_interval_float
from mimose.losses.imfuse import dice_loss, softmax_weighted_loss


def m3ae_reconstruction_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    limage: torch.Tensor,
    *,
    reg_weight: float = 0.005,
) -> dict[str, torch.Tensor]:
    """Legacy M3AE pretraining objective: MSE reconstruction of the original,
    unmasked input plus a smoothness regularizer on the learned placeholder
    (``limage``) that keeps it from degenerating into per-voxel noise.

    Matches ``legacy/m3ae/pretrain.py``'s ``loss_ + loss2 * .005``, where
    ``loss_`` is the MSE term and ``loss2`` is
    ``torch.norm(limage - limage.mean((2,3,4), keepdim=True), 2)``.
    """
    recon = F.mse_loss(output, target)
    smoothness = torch.norm(limage - limage.mean(dim=(2, 3, 4), keepdim=True), p=2)
    return {"loss": recon + reg_weight * smoothness, "recon": recon, "smoothness": smoothness}


class M3AELoss:
    """Loss wrapper for legacy M3AE's fine-tuning objective: softmax
    cross-entropy + Dice. ``training_loss`` is the single-output form;
    ``deep_supervised_training_loss`` sums it over the deep-supervision heads.
    The cross-view consistency MSE term is added by ``M3AETrainer`` (which owns
    the two-view forward passes), not here.

    Used for the fine-tuning stage only; the pretraining stage uses
    ``m3ae_reconstruction_loss`` instead (see ``M3AETrainer``).
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

    def deep_supervised_training_loss(
        self, seg_outputs: list[torch.Tensor], target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Legacy M3AE deep-supervised objective: the softmax cross-entropy +
        Dice segmentation loss summed (unweighted) over every deep-supervision
        head -- matches ``for l in segs_S1: loss_ += criterion(l, labels)`` in
        ``legacy/m3ae/train.py:304-305``."""
        cross = target.new_tensor(0.0)
        dice = target.new_tensor(0.0)
        for output in seg_outputs:
            cross = cross + self.softmax_weighted_loss(output, target)
            dice = dice + self.dice_loss(output, target)
        return {"loss": cross + dice, "cross": cross, "dice": dice}


__all__ = ["M3AELoss", "m3ae_reconstruction_loss"]
