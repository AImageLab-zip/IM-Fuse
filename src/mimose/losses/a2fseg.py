from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from mimose.losses.imfuse import dice_loss, softmax_weighted_loss


def _deep_supervision_weights(num_scales: int) -> list[float]:
    """Exponentially-decaying per-scale weights (mirrors the legacy
    ``MultipleOutputLoss2`` weighting: highest resolution weighted most,
    normalized to sum to 1)."""
    weights = [1.0 / (2**i) for i in range(num_scales)]
    total = sum(weights)
    return [w / total for w in weights]


class A2FSegLoss:
    """Loss wrapper matching A2FSeg's multi-branch deep-supervision training:
    the fused prediction, each modality's independent decoder prediction, and
    the fusion decoder's own multi-scale predictions are all supervised with
    Dice + weighted cross-entropy."""

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        num_cls: int | None = None,
        fuse_weight: float = 1.0,
        sep_weight: float = 1.0,
        fusion_ds_weight: float = 1.0,
        eps: float = 1e-7,
        log_clamp_min: float = 0.005,
    ) -> None:
        resolved_num_classes = (
            int(num_classes)
            if num_classes is not None
            else int(num_cls) if num_cls is not None else 4
        )
        self.num_classes = resolved_num_classes
        self.fuse_weight = float(fuse_weight)
        self.sep_weight = float(sep_weight)
        self.fusion_ds_weight = float(fusion_ds_weight)
        self.eps = float(eps)
        self.log_clamp_min = float(log_clamp_min)

    def segmentation_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.softmax_weighted_loss(output, target) + self.dice_loss(output, target)

    def training_loss(
        self,
        outputs: tuple[
            torch.Tensor,
            Sequence[Sequence[torch.Tensor]],
            Sequence[torch.Tensor],
        ],
        target: torch.Tensor,
        include_fuse: bool = True,
        include_sep: bool = True,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        fused_pred, sep_preds, fusion_ds_preds = outputs
        zero = target.new_tensor(0.0)

        fuse = (
            self._branch_loss(F.softmax(fused_pred, dim=1), target)
            if include_fuse
            else {"cross": zero, "dice": zero, "total": zero}
        )
        sep = (
            self._sep_loss(sep_preds, target, mask)
            if include_sep
            else {"cross": zero, "dice": zero, "total": zero}
        )
        fusion_ds = self._multi_scale_loss(fusion_ds_preds, target)

        total = (
            (self.fuse_weight * fuse["total"] if include_fuse else zero)
            + (self.sep_weight * sep["total"] if include_sep else zero)
            + (self.fusion_ds_weight * fusion_ds["total"])
        )

        return {
            "loss": total,
            "fusecross": fuse["cross"],
            "fusedice": fuse["dice"],
            "sepcross": sep["cross"],
            "sepdice": sep["dice"],
            "fusiondscross": fusion_ds["cross"],
            "fusiondsdice": fusion_ds["dice"],
        }

    def dice_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss(output, target, num_cls=self.num_classes, eps=self.eps)

    def softmax_weighted_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return softmax_weighted_loss(
            output,
            target,
            num_cls=self.num_classes,
            log_clamp_min=self.log_clamp_min,
            eps=self.eps,
        )

    def _branch_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        cross = self.softmax_weighted_loss(prediction, target)
        dice = self.dice_loss(prediction, target)
        return {"cross": cross, "dice": dice, "total": cross + dice}

    def _multi_scale_loss(
        self,
        predictions: Sequence[torch.Tensor],
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not predictions:
            zero = target.new_tensor(0.0)
            return {"cross": zero, "dice": zero, "total": zero}

        weights = _deep_supervision_weights(len(predictions))
        cross = target.new_tensor(0.0)
        dice = target.new_tensor(0.0)
        for weight, prediction in zip(weights, predictions):
            branch = self._branch_loss(prediction, target)
            cross = cross + (weight * branch["cross"])
            dice = dice + (weight * branch["dice"])
        return {"cross": cross, "dice": dice, "total": cross + dice}

    def _sep_loss(
        self,
        sep_preds: Sequence[Sequence[torch.Tensor]],
        target: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        zero = target.new_tensor(0.0)
        cross = zero
        dice = zero
        num_supervised = 0
        for modal_index, predictions in enumerate(sep_preds):
            present = (
                mask[:, modal_index] if mask is not None else torch.ones(
                    target.size(0), dtype=torch.bool, device=target.device
                )
            )
            if not bool(present.any()):
                continue
            modality_target = target[present]
            modality_predictions = [prediction[present] for prediction in predictions]
            branch = self._multi_scale_loss(modality_predictions, modality_target)
            cross = cross + branch["cross"]
            dice = dice + branch["dice"]
            num_supervised += 1

        divisor = max(num_supervised, 1)
        cross = cross / divisor
        dice = dice / divisor
        return {"cross": cross, "dice": dice, "total": cross + dice}


__all__ = ["A2FSegLoss"]
