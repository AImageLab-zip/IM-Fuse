from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F

from mimose.losses.config import KwargField, positive_int, unit_interval_float
from mimose.losses.imfuse import dice_loss, softmax_weighted_loss


def _pairwise_cosine_contrastive(prototypes: torch.Tensor, temperature: float) -> torch.Tensor:
    """Matches legacy MCPL's ``memory_Loss.forward_class_cos`` /
    ``forward_modal_cos`` (legacy/MCPL/loss/dice.py:157-223): an InfoNCE-style
    contrastive term over pairs of prototype vectors drawn from the model's
    own learned (modality x class) prototype table (``_cluster_centers``),
    independent of any batch data.

    ``prototypes``: ``[Q, num_groups]`` (e.g. ``cc_classes`` columns of
    ``Q``-dim vectors when pulling classes apart, or ``cc_modalities`` columns
    when pulling modalities apart). Each pair of columns is pushed apart
    relative to all other columns via a softmax-cosine-similarity contrast.
    """
    num_groups = prototypes.shape[1]
    loss = prototypes.new_tensor(0.0)
    pair_count = 0
    for i in range(num_groups):
        for j in range(i + 1, num_groups):
            positive = torch.exp(torch.cosine_similarity(prototypes[:, i], prototypes[:, j], dim=0) / temperature)
            negative = prototypes.new_tensor(0.0)
            for k in range(num_groups):
                negative = negative + torch.exp(
                    torch.cosine_similarity(prototypes[:, i], prototypes[:, k], dim=0) / temperature
                )
            loss = loss - torch.log(positive / negative)
            pair_count += 1
    return loss / max(pair_count, 1)


def class_contrastive_loss(
    cluster_centers_weight: torch.Tensor,
    *,
    cc_modalities: int,
    cc_classes: int,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Pulls the per-class prototype directions apart across modalities,
    matching legacy MCPL's ``forward_class_cos`` (used during fine-tuning)."""
    prototypes = cluster_centers_weight.view(-1, cc_modalities, cc_classes)
    loss = prototypes.new_tensor(0.0)
    for modal_index in range(cc_modalities):
        loss = loss + _pairwise_cosine_contrastive(prototypes[:, modal_index, :], temperature)
    return loss / cc_modalities


def modal_contrastive_loss(
    cluster_centers_weight: torch.Tensor,
    *,
    cc_modalities: int,
    cc_classes: int,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Pulls the per-modality prototype directions apart across classes,
    matching legacy MCPL's ``forward_modal_cos`` (used during pretraining)."""
    prototypes = cluster_centers_weight.view(-1, cc_modalities, cc_classes)
    loss = prototypes.new_tensor(0.0)
    for class_index in range(cc_classes):
        loss = loss + _pairwise_cosine_contrastive(prototypes[:, :, class_index], temperature)
    return loss / cc_classes


def mcpl_reconstruction_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    cluster_centers_weight: torch.Tensor,
    *,
    cc_modalities: int,
    cc_classes: int,
    modal_contrast_weight: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Legacy MCPL pretraining objective: MSE reconstruction of the original,
    unmasked input plus the modality-prototype contrastive loss (matches
    ``legacy/MCPL/pretrain.py``'s ``loss_ + modulContrast_loss * 0.01``)."""
    recon = F.mse_loss(output, target)
    modal_contrast = modal_contrastive_loss(cluster_centers_weight, cc_modalities=cc_modalities, cc_classes=cc_classes)
    return {
        "loss": recon + modal_contrast_weight * modal_contrast,
        "recon": recon,
        "modal_contrast": modal_contrast,
    }


class MCPLLoss:
    """Loss wrapper for legacy MCPL's fine-tuning objective: softmax
    cross-entropy + Dice segmentation loss (following the single-label
    4-class convention shared with ``M3AELoss``/``ShaSpecLoss``), plus the
    class-prototype contrastive term. The cross-view consistency MSE and the
    class-contrastive weighting are added by ``MCPLTrainer`` (which owns the
    two-view forward passes and has access to the model's prototype table),
    not here. The pretraining stage uses ``mcpl_reconstruction_loss`` instead.
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
        """Sums the softmax cross-entropy + Dice loss over every
        deep-supervision head, matching legacy MCPL's
        ``for l in segs_S1: loss_ += criterion(l, labels_S1)``
        (``legacy/MCPL/train.py:305-306``)."""
        cross = target.new_tensor(0.0)
        dice = target.new_tensor(0.0)
        for output in seg_outputs:
            cross = cross + self.softmax_weighted_loss(output, target)
            dice = dice + self.dice_loss(output, target)
        return {"loss": cross + dice, "cross": cross, "dice": dice}


__all__ = [
    "MCPLLoss",
    "class_contrastive_loss",
    "mcpl_reconstruction_loss",
    "modal_contrastive_loss",
]
