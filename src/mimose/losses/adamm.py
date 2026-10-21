from __future__ import annotations

import math
from typing import ClassVar

import torch
import torch.nn.functional as F

from mimose.losses.config import KwargField, nonneg_float, positive_float, positive_int
from mimose.losses.mstkdnet import style_loss


def _per_sample_dice_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Legacy AdaMM ``dice_loss(..., reduction='none')``: squared-denominator soft Dice, one value per sample."""
    prediction = prediction.contiguous().flatten(1)
    target = target.contiguous().flatten(1)
    intersection = (prediction * target).sum(dim=1)
    denom = prediction.pow(2).sum(dim=1) + target.pow(2).sum(dim=1) + eps
    return 1.0 - 2.0 * intersection / denom


def _depth_mix_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Legacy ``mix_matrix`` applied to one sample's [C, D, H, W] global feature.

    Legacy indexes the batched [B, C, D, H, W] tensor per sample and then
    unpacks the result as (B, C, H, W), so the "channels" it correlates are the
    depth slices, batched over the feature channels.
    """
    channels, depth, height, width = x.shape
    x = x.reshape(channels, depth, height * width)
    y = y.reshape(channels, depth, height * width)
    return torch.bmm(x, y.transpose(1, 2)) / (height * width)


def global_style_loss(
    global_full: torch.Tensor,
    global_missing: torch.Tensor,
    *,
    per_sample: bool = False,
) -> torch.Tensor:
    """Legacy AdaMM ``get_GS_loss`` over the concatenated [c3d, style, c4d] bottleneck feature.

    Legacy loops ``i`` over the batch and pairs sample ``i`` with ``i - 1``;
    at its batch size of 1 that pairs each sample with itself, which is what
    ``per_sample=True`` keeps for larger batches (averaged over the batch).
    """
    batch = global_full.size(0)
    depth = global_full.size(2)
    size = global_full.size(3) * global_full.size(4)
    norm = 4.0 * (depth**2) * (size**2)

    loss = global_full.new_tensor(0.0)
    for index in range(batch):
        partner = index if per_sample else index - 1
        mix_full = _depth_mix_matrix(global_full[index], global_full[partner])
        mix_missing = _depth_mix_matrix(global_missing[index], global_missing[partner])
        loss = loss + torch.sum(torch.square(mix_full - mix_missing)) / norm
    return loss / batch if per_sample else loss


def _sigmoid_rampup(current: float, rampup_length: float) -> float:
    if rampup_length == 0:
        return 1.0
    current = min(max(current, 0.0), rampup_length)
    phase = 1.0 - current / rampup_length
    return float(math.exp(-5.0 * phase * phase))


class AdaMMLoss:
    """Loss wrapper matching legacy AdaMM's co-training + LPGM + distillation terms."""

    KWARG_SPEC: ClassVar[dict[str, KwargField]] = {
        "num_classes": positive_int(),
        "weight_mispath": nonneg_float(),
        "weight_content": nonneg_float(),
        "weight_gsm": nonneg_float(),
        "weight_presence": nonneg_float(),
        "consistency_max_weight": nonneg_float(),
        "consistency_rampup_epochs": positive_float(),
        "style_size_scale": positive_float(),
    }

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        num_cls: int | None = None,
        weight_mispath: float = 0.6,
        weight_content: float = 0.2,
        weight_gsm: float = 100.0,
        weight_presence: float = 1.0,
        consistency_max_weight: float = 10.0,
        consistency_rampup_epochs: float = 20.0,
        per_sample_statistics: bool = True,
        style_size_scale: float = 1.0,
    ) -> None:
        resolved_num_classes = (
            int(num_classes)
            if num_classes is not None
            else int(num_cls) if num_cls is not None else 4
        )
        self.num_classes = resolved_num_classes
        self.weight_mispath = float(weight_mispath)
        self.weight_content = float(weight_content)
        self.weight_gsm = float(weight_gsm)
        self.weight_presence = float(weight_presence)
        self.consistency_max_weight = float(consistency_max_weight)
        self.consistency_rampup_epochs = float(consistency_rampup_epochs)
        # Legacy runs at batch 1: per-sample Gram statistics keep that meaning at batch > 1.
        self.per_sample_statistics = bool(per_sample_statistics)
        self.style_size_scale = float(style_size_scale)

    def segmentation_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = output.new_tensor(0.0)
        for class_index in range(self.num_classes):
            loss = loss + _per_sample_dice_loss(output[:, class_index], target[:, class_index]).mean()
        return loss

    def _consistency_weight(self, epoch: int) -> float:
        return self.consistency_max_weight * _sigmoid_rampup(float(epoch), self.consistency_rampup_epochs)

    def _presence_weighted_dice(
        self, prediction: torch.Tensor, target: torch.Tensor, presence_weight: torch.Tensor
    ) -> torch.Tensor:
        """Sum over classes of batch-mean Dice; tumor classes are weighted by LPGM's |p - y| + 1."""
        loss = prediction.new_tensor(0.0)
        for class_index in range(self.num_classes):
            dice = _per_sample_dice_loss(prediction[:, class_index], target[:, class_index])
            if class_index > 0:
                dice = presence_weight[:, class_index - 1] * dice
            loss = loss + dice.mean()
        return loss

    def training_loss(
        self,
        outputs: tuple[tuple, tuple],
        target: torch.Tensor,
        epoch: int,
    ) -> dict[str, torch.Tensor]:
        student_out, teacher_out = outputs
        uout_s, style_s, content_s, gs_s, _, cls_s = (tensor.float() for tensor in student_out)
        uout_t, style_t, content_t, gs_t, _, cls_t = (tensor.float() for tensor in teacher_out)
        target = target.float()

        # Per-sample presence of every tumor class in the patch (legacy's `cls`).
        presence = target[:, 1:].flatten(2).amax(dim=2)

        loss_dc = self._presence_weighted_dice(uout_t, target, (cls_t - presence).abs() + 1.0)
        loss_miss_dc = self._presence_weighted_dice(uout_s, target, (cls_s - presence).abs() + 1.0)

        consistency = uout_s.new_tensor(0.0)
        for class_index in range(self.num_classes):
            consistency = consistency + F.mse_loss(uout_t[:, class_index], uout_s[:, class_index], reduction="mean")

        content = F.mse_loss(content_t, content_s, reduction="mean")
        style = style_loss(
            style_t, style_s, per_sample=self.per_sample_statistics, size_scale=self.style_size_scale
        )
        gsm = global_style_loss(gs_t, gs_s, per_sample=self.per_sample_statistics)
        presence_full = F.binary_cross_entropy(cls_t, presence)
        presence_missing = F.binary_cross_entropy(cls_s, presence)

        weight_missing = self.weight_mispath
        weight_full = 1.0 - weight_missing
        total = (
            weight_full * loss_dc
            + weight_missing * loss_miss_dc
            + self._consistency_weight(epoch) * consistency
            + self.weight_content * content
            + style
            + self.weight_presence * (presence_full + presence_missing)
            + self.weight_gsm * gsm
        )

        return {
            "loss": total,
            "loss_dc": loss_dc,
            "loss_miss_dc": loss_miss_dc,
            "consistency": consistency,
            "content": content,
            "style": style,
            "gsm": gsm,
            "presence_full": presence_full,
            "presence_missing": presence_missing,
        }


__all__ = ["AdaMMLoss", "global_style_loss"]
