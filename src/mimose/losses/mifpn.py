from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import torch
import torch.nn.functional as F

from mimose.losses.config import (
    KwargField,
    nonneg_float,
    positive_float,
    positive_int,
    unit_interval_float,
)
from mimose.losses.imfuse import dice_loss, softmax_weighted_loss


def prompt_consistency_kl_loss(
    prompts: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Legacy `softmax_kl_loss`: KL(log_softmax(mi_prompt), softmax(modality_prompt))
    summed over the 4 per-modality prompts."""
    flair_prompt, t1ce_prompt, t1_prompt, t2_prompt, mi_prompt = prompts
    mi_log_softmax = F.log_softmax(mi_prompt, dim=1)

    loss = mi_prompt.new_tensor(0.0)
    for modality_prompt in (flair_prompt, t1ce_prompt, t1_prompt, t2_prompt):
        modality_softmax = F.softmax(modality_prompt, dim=1) + eps
        loss = loss + F.kl_div(mi_log_softmax, modality_softmax, reduction="mean")
    return loss


class MIFPNLoss:
    """Loss wrapper matching legacy MIFPN's fuse + sep + prm + KL prompt-consistency terms."""

    KWARG_SPEC: ClassVar[dict[str, KwargField]] = {
        "num_classes": positive_int(),
        "fuse_weight": nonneg_float(),
        "sep_weight": nonneg_float(),
        "prm_weight": nonneg_float(),
        "kl_weight": nonneg_float(),
        "eps": positive_float(),
        "log_clamp_min": unit_interval_float(),
    }

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        num_cls: int | None = None,
        fuse_weight: float = 1.0,
        sep_weight: float = 1.0,
        prm_weight: float = 1.0,
        kl_weight: float = 1.0,
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
        self.prm_weight = float(prm_weight)
        self.kl_weight = float(kl_weight)
        self.eps = float(eps)
        self.log_clamp_min = float(log_clamp_min)

    def segmentation_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.softmax_weighted_loss(output, target) + self.dice_loss(output, target)

    def training_loss(
        self,
        outputs: tuple[torch.Tensor, Sequence[torch.Tensor], Sequence[torch.Tensor], tuple],
        target: torch.Tensor,
        include_fuse: bool,
    ) -> dict[str, torch.Tensor]:
        fuse_pred, sep_preds, prm_preds, prompts = outputs

        zero = target.new_tensor(0.0)

        fuse = self._branch_loss(fuse_pred, target) if include_fuse else {"cross": zero, "dice": zero, "total": zero}
        sep = self._multi_branch_loss(sep_preds, target)
        prm = self._multi_branch_loss(prm_preds, target)
        kl = prompt_consistency_kl_loss(prompts)

        total = (self.sep_weight * sep["total"]) + (self.prm_weight * prm["total"]) + (self.kl_weight * kl)
        if include_fuse:
            total = total + (self.fuse_weight * fuse["total"])

        return {
            "loss": total,
            "fusecross": fuse["cross"],
            "fusedice": fuse["dice"],
            "sepcross": sep["cross"],
            "sepdice": sep["dice"],
            "prmcross": prm["cross"],
            "prmdice": prm["dice"],
            "kl": kl,
        }

    def dice_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss(output, target, num_cls=self.num_classes, eps=self.eps)

    def softmax_weighted_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return softmax_weighted_loss(
            output, target, num_cls=self.num_classes, log_clamp_min=self.log_clamp_min, eps=self.eps
        )

    def _branch_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        cross = self.softmax_weighted_loss(prediction, target)
        dice = self.dice_loss(prediction, target)
        return {"cross": cross, "dice": dice, "total": cross + dice}

    def _multi_branch_loss(
        self,
        predictions: Sequence[torch.Tensor],
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not predictions:
            zero = target.new_tensor(0.0)
            return {"cross": zero, "dice": zero, "total": zero}

        cross = target.new_tensor(0.0)
        dice = target.new_tensor(0.0)
        for prediction in predictions:
            branch_loss = self._branch_loss(prediction, target)
            cross = cross + branch_loss["cross"]
            dice = dice + branch_loss["dice"]
        return {"cross": cross, "dice": dice, "total": cross + dice}


__all__ = ["MIFPNLoss", "prompt_consistency_kl_loss"]
