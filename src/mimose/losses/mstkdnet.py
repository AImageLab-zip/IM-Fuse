from __future__ import annotations

import math
from typing import ClassVar

import torch
import torch.nn.functional as F

from mimose.losses.config import KwargField, nonneg_float, positive_float, positive_int


def _dice_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Legacy MST-KDNet per-channel soft dice, flattened over the whole batch+volume."""
    prediction = prediction.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (prediction * target).sum()
    denom = prediction.pow(2).sum() + target.pow(2).sum() + eps
    return 1.0 - (2.0 * intersection / denom)


def _gram_matrix(x: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width, depth = x.size()
    features = x.view(batch * channels, height * width * depth)
    gram = torch.mm(features, features.t())
    return gram.div(batch * channels * height * width * depth)


def _mix_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width, depth = x.size()
    x = x.view(batch * channels, height * width * depth)
    y = y.view(batch * channels, height * width * depth)
    mix = torch.mm(x, y.t())
    return mix.div(batch * channels * height * width * depth)


def style_loss(style_full: torch.Tensor, style_missing: torch.Tensor) -> torch.Tensor:
    gram_full = _gram_matrix(style_full)
    gram_missing = _gram_matrix(style_missing)
    channels = style_full.size(1)
    size = style_full.size(2) * style_full.size(3)
    loss = torch.sum(torch.square(gram_full - gram_missing)) / (4.0 * (channels**2) * (size**2))
    return loss * 1e-4


def global_style_match_loss(
    global_style_full: list[torch.Tensor],
    global_style_missing: list[torch.Tensor],
) -> torch.Tensor:
    """Legacy's cyclic (i, i-1) pairwise cross-correlation match over the
    3-element [c3d, style, c4d] "Global_style" lists."""
    mix_full = [_mix_matrix(global_style_full[i], global_style_full[i - 1]) for i in range(len(global_style_full))]
    mix_missing = [
        _mix_matrix(global_style_missing[i], global_style_missing[i - 1]) for i in range(len(global_style_missing))
    ]

    channels = global_style_full[0].size(1)
    size = global_style_full[0].size(2) * global_style_full[0].size(3)

    loss = global_style_full[0].new_tensor(0.0)
    for mix_f, mix_m in zip(mix_full, mix_missing):
        loss = loss + torch.sum(torch.square(mix_f - mix_m)) / (4.0 * (channels**2) * (size**2))
    return loss * 1e-4


def unetr_feature_loss(teacher_features: list[torch.Tensor], student_features: list[torch.Tensor]) -> torch.Tensor:
    loss = teacher_features[0].new_tensor(0.0)
    for teacher_feature, student_feature in zip(teacher_features, student_features):
        loss = loss + F.mse_loss(student_feature, teacher_feature, reduction="mean")
    return loss


def _compute_extreme_maps(
    weights_list: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    stacked = torch.stack(weights_list, dim=0)
    max_map = stacked.max(dim=0).values
    min_map = stacked.min(dim=0).values
    mean_map = stacked.mean(dim=0)
    return (
        [weights * max_map for weights in weights_list],
        [weights * min_map for weights in weights_list],
        [weights * mean_map for weights in weights_list],
    )


def extreme_value_distillation_loss(
    teacher_weights: list[torch.Tensor],
    student_weights: list[torch.Tensor],
) -> torch.Tensor:
    teacher_max, teacher_min, teacher_mean = _compute_extreme_maps(teacher_weights)
    student_max, student_min, student_mean = _compute_extreme_maps(student_weights)

    loss = teacher_weights[0].new_tensor(0.0)
    for t_max, s_max in zip(teacher_max, student_max):
        loss = loss + F.mse_loss(t_max, s_max, reduction="mean")
    for t_min, s_min in zip(teacher_min, student_min):
        loss = loss + F.mse_loss(t_min, s_min, reduction="mean")
    for t_mean, s_mean in zip(teacher_mean, student_mean):
        loss = loss + F.mse_loss(t_mean, s_mean, reduction="mean")
    return loss


def _standardize_logit(logit: torch.Tensor) -> torch.Tensor:
    mean = logit.mean(dim=-1, keepdim=True)
    std = logit.std(dim=-1, keepdim=True)
    return (logit - mean) / (1e-7 + std)


def logit_stand_kd_loss(logit_a: torch.Tensor, logit_b: torch.Tensor, temperature: float) -> torch.Tensor:
    """Legacy `DistillKL_logit_stand`, called positionally as (logit_full, logit_missing)."""
    kd_loss = torch.nn.functional.kl_div(
        F.log_softmax(_standardize_logit(logit_a) / temperature, dim=1),
        F.softmax(_standardize_logit(logit_b) / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature**2)
    return kd_loss / logit_a.numel()


def _sigmoid_rampup(current: float, rampup_length: float) -> float:
    if rampup_length == 0:
        return 1.0
    current = min(max(current, 0.0), rampup_length)
    phase = 1.0 - current / rampup_length
    return float(math.exp(-5.0 * phase * phase))


class MSTKDNetLoss:
    """Loss wrapper matching legacy MST-KDNet's co-training + distillation terms."""

    KWARG_SPEC: ClassVar[dict[str, KwargField]] = {
        "num_classes": positive_int(),
        "weight_mispath": nonneg_float(),
        "weight_content": nonneg_float(),
        "weight_gsm": nonneg_float(),
        "unetr_weight": nonneg_float(),
        "evd_weight": nonneg_float(),
        "slkd_temperature": positive_float(),
        "consistency_max_weight": nonneg_float(),
        "consistency_rampup_epochs": positive_float(),
    }

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        num_cls: int | None = None,
        weight_mispath: float = 0.6,
        weight_content: float = 0.2,
        weight_gsm: float = 1e11,
        unetr_weight: float = 0.2,
        evd_weight: float = 1e8,
        slkd_temperature: float = 7.0,
        consistency_max_weight: float = 10.0,
        consistency_rampup_epochs: float = 20.0,
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
        self.unetr_weight = float(unetr_weight)
        self.evd_weight = float(evd_weight)
        self.slkd_temperature = float(slkd_temperature)
        self.consistency_max_weight = float(consistency_max_weight)
        self.consistency_rampup_epochs = float(consistency_rampup_epochs)

    def segmentation_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = output.new_tensor(0.0)
        for class_index in range(self.num_classes):
            loss = loss + _dice_loss(output[:, class_index], target[:, class_index])
        return loss

    def _consistency_weight(self, epoch: int) -> float:
        return self.consistency_max_weight * _sigmoid_rampup(float(epoch), self.consistency_rampup_epochs)

    def training_loss(
        self,
        outputs: tuple[tuple, tuple],
        target: torch.Tensor,
        epoch: int,
    ) -> dict[str, torch.Tensor]:
        student_out, teacher_out = outputs
        uout_s, style_s, content_s, unetr_fs_s, weights_s, gstyle_s, logit_s = student_out
        uout_t, style_t, content_t, unetr_fs_t, weights_t, gstyle_t, logit_t = teacher_out

        loss_dc = uout_t.new_tensor(0.0)
        loss_miss_dc = uout_s.new_tensor(0.0)
        consistency = uout_s.new_tensor(0.0)
        for class_index in range(self.num_classes):
            loss_dc = loss_dc + _dice_loss(uout_t[:, class_index], target[:, class_index])
            loss_miss_dc = loss_miss_dc + _dice_loss(uout_s[:, class_index], target[:, class_index])
            consistency = consistency + F.mse_loss(uout_t[:, class_index], uout_s[:, class_index], reduction="mean")

        content = F.mse_loss(content_t, content_s, reduction="mean")
        style = style_loss(style_t, style_s)
        gsm = global_style_match_loss(gstyle_t, gstyle_s)
        unetr = unetr_feature_loss(unetr_fs_t, unetr_fs_s)
        evd = extreme_value_distillation_loss(weights_t, weights_s)
        slkd = logit_stand_kd_loss(logit_t, logit_s, self.slkd_temperature)

        weight_missing = self.weight_mispath
        weight_full = 1.0 - weight_missing
        weight_consistency = self._consistency_weight(epoch)

        loss_co = (
            weight_full * loss_dc
            + weight_missing * loss_miss_dc
            + weight_consistency * consistency
            + self.weight_content * content
            + style
        )
        total = loss_co + self.unetr_weight * unetr + self.evd_weight * evd + slkd + self.weight_gsm * gsm

        return {
            "loss": total,
            "loss_dc": loss_dc,
            "loss_miss_dc": loss_miss_dc,
            "consistency": consistency,
            "content": content,
            "style": style,
            "gsm": gsm,
            "unetr": unetr,
            "evd": evd,
            "slkd": slkd,
        }


__all__ = [
    "MSTKDNetLoss",
    "extreme_value_distillation_loss",
    "global_style_match_loss",
    "logit_stand_kd_loss",
    "style_loss",
    "unetr_feature_loss",
]
