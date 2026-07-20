from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimose.losses.mstkdnet import logit_stand_kd_loss
from mimose.models.abstract_model import AbstractModel
from mimose.models.many_mimosas import ALL_MASK_PATTERNS, ManyMimosas
from mimose.models.tiny_mimosa import TinyMimosa


NUM_MODALITIES = 4
DEFAULT_NUM_CLASSES = 4
DEFAULT_INPUT_SHAPE = (182, 218, 182)
DEFAULT_STUDENT_FEATURES_PER_STAGE = (8, 16, 32, 64)
DEFAULT_TEACHER_FEATURES_PER_STAGE = (16, 32, 64, 128)


def _pattern_key(pattern: Sequence[bool]) -> str:
    return "".join("1" if bool(present) else "0" for present in pattern)


class _TinyMimosaHints(NamedTuple):
    pred: torch.Tensor
    logits: torch.Tensor
    encoder_feats: list[torch.Tensor]
    decoder_feats: list[torch.Tensor]


def _forward_with_hints(model: TinyMimosa, images: torch.Tensor, mask: torch.Tensor) -> _TinyMimosaHints:
    """Replicates ``TinyMimosa.forward`` + ``UNetDecoder.forward``'s stage
    loop (dynamic_network_architectures/building_blocks/unet_decoder.py),
    but additionally collects every encoder/decoder stage's intermediate
    feature map instead of only the final logits -- used by
    ``ManyMimosasKD`` for hint-based distillation. ``tiny_mimosa.py`` itself
    is left untouched; this is a read-only replay of its submodules.
    """
    gate = mask.to(images.dtype).view(images.size(0), images.size(1), 1, 1, 1)
    gated_images = images * gate

    skips = model.backbone.encoder(gated_images)
    encoder_feats = list(skips)

    scale, bias = model.mask_conditioning(mask)
    skips[-1] = skips[-1] * (1.0 + torch.tanh(scale)) + bias

    decoder = model.backbone.decoder
    lres_input = skips[-1]
    decoder_feats: list[torch.Tensor] = []
    for s in range(len(decoder.stages)):
        x = decoder.transpconvs[s](lres_input)
        x = torch.cat((x, skips[-(s + 2)]), 1)
        x = decoder.stages[s](x)
        decoder_feats.append(x)
        lres_input = x

    logits = decoder.seg_layers[-1](lres_input)
    pred = torch.softmax(logits, dim=1)
    return _TinyMimosaHints(pred=pred, logits=logits, encoder_feats=encoder_feats, decoder_feats=decoder_feats)


def _cosine_hint_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    similarity = F.cosine_similarity(student_feat, teacher_feat, dim=1)
    return (1.0 - similarity).mean()


class ManyMimosasKD(AbstractModel):
    """Distills a wider, same-depth TinyMimosa "teacher" (always full
    modalities) into the 15 missing-modality TinyMimosa "students" wrapped
    by ``ManyMimosas``, via deep-supervision feature hints at every
    encoder/decoder stage plus a logit-KD term, on top of each student's own
    Dice+CE segmentation loss.

    This codebase's trainer contract has no notion of loading a second,
    externally-pretrained checkpoint into a submodule (only the same live
    model's own checkpoints can be resumed/pretrained-from -- see
    ``BaseTrainer._load_pretrain``), so -- mirroring ``MambaVitAKD`` -- the
    teacher is embedded as a submodule and trained jointly with the
    students in a single run: ``ManyMimosasKDTrainer`` runs a
    ``teacher_pretrain_fraction``-scoped phase where only the teacher is
    trained (full modalities, direct segmentation loss), then freezes it
    and starts the distillation phase.

    Every one of the 15 students shares the same ``features_per_stage``
    (only their input-channel count differs, per ``ManyMimosas``), so their
    stage features are always identically shaped regardless of which
    modality pattern they handle -- this lets a single, shared set of 1x1x1
    hint adapters (linear channel projections, no nonlinearity) serve all
    15 students rather than needing one adapter set per pattern. Keeping
    the adapters linear-only and shared limits how much they can "absorb"
    a student/teacher representation mismatch instead of the student
    encoder actually learning teacher-like features.

    The deployable artifact of a ``ManyMimosasKD`` run is the 15 students
    alone -- architecturally a plain ``ManyMimosas`` -- since the teacher
    and hint adapters are training-only scaffolding. See
    ``model_for_export()``.
    """

    def __init__(
        self,
        *,
        num_cls: int = DEFAULT_NUM_CLASSES,
        num_modals: int = NUM_MODALITIES,
        input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
        student_features_per_stage: Sequence[int] = DEFAULT_STUDENT_FEATURES_PER_STAGE,
        teacher_features_per_stage: Sequence[int] = DEFAULT_TEACHER_FEATURES_PER_STAGE,
        n_conv_per_stage: int = 2,
        n_conv_per_stage_decoder: int = 2,
        overlap: float = 0.5,
        logit_kd_temperature: float = 4.0,
    ) -> None:
        super().__init__()
        if num_modals != NUM_MODALITIES:
            raise ValueError(f"ManyMimosasKD expects exactly {NUM_MODALITIES} modalities")
        if len(student_features_per_stage) != len(teacher_features_per_stage):
            raise ValueError(
                "ManyMimosasKD requires the student and teacher to have the same number of "
                f"stages, got {len(student_features_per_stage)} vs {len(teacher_features_per_stage)}"
            )

        self.num_cls = int(num_cls)
        self.num_modals = int(num_modals)
        self.input_shape = tuple(int(dim) for dim in input_shape)
        self.student_features_per_stage = tuple(int(ch) for ch in student_features_per_stage)
        self.teacher_features_per_stage = tuple(int(ch) for ch in teacher_features_per_stage)
        self.n_conv_per_stage = int(n_conv_per_stage)
        self.n_conv_per_stage_decoder = int(n_conv_per_stage_decoder)
        self.overlap = float(overlap)
        self.logit_kd_temperature = float(logit_kd_temperature)
        self.is_training = False

        self.submodels = nn.ModuleDict(
            {
                _pattern_key(pattern): TinyMimosa(
                    num_cls=num_cls,
                    num_modals=int(pattern.sum().item()),
                    input_shape=input_shape,
                    features_per_stage=self.student_features_per_stage,
                    n_conv_per_stage=n_conv_per_stage,
                    n_conv_per_stage_decoder=n_conv_per_stage_decoder,
                    overlap=overlap,
                )
                for pattern in ALL_MASK_PATTERNS
            }
        )
        self.teacher = TinyMimosa(
            num_cls=num_cls,
            num_modals=num_modals,
            input_shape=input_shape,
            features_per_stage=self.teacher_features_per_stage,
            n_conv_per_stage=n_conv_per_stage,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            overlap=overlap,
        )
        self.tile_shape = self.teacher.tile_shape

        self.hint_adapters_encoder = nn.ModuleList(
            nn.Conv3d(student_ch, teacher_ch, kernel_size=1)
            for student_ch, teacher_ch in zip(
                self.student_features_per_stage, self.teacher_features_per_stage
            )
        )
        self.hint_adapters_decoder = nn.ModuleList(
            nn.Conv3d(student_ch, teacher_ch, kernel_size=1)
            for student_ch, teacher_ch in zip(
                reversed(self.student_features_per_stage[:-1]),
                reversed(self.teacher_features_per_stage[:-1]),
            )
        )

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        *,
        mode: str = "auto",
        teacher_hints: _TinyMimosaHints | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | _TinyMimosaHints:
        self._validate_boundary_inputs(images, mask)

        if mode == "teacher_pretrain":
            full_mask = torch.ones_like(mask)
            self.teacher.is_training = self.is_training
            return self.teacher(images, full_mask)

        if mode == "teacher_forward":
            full_mask = torch.ones_like(mask)
            return _forward_with_hints(self.teacher, images, full_mask)

        for submodel in self.submodels.values():
            submodel.is_training = self.is_training

        if not self.is_training:
            return self._route_predict_batch(images, mask)

        mask_bool = mask.bool()
        pattern = mask_bool[0]
        student = self.submodels[_pattern_key(pattern.tolist())]
        channel_indices = pattern.nonzero(as_tuple=True)[0]
        sub_images = images.index_select(1, channel_indices)
        sub_mask = torch.ones(
            images.size(0), channel_indices.numel(), dtype=torch.bool, device=images.device
        )
        student_hints = _forward_with_hints(student, sub_images, sub_mask)

        if teacher_hints is None:
            return student_hints.pred, (), ()

        hint_loss = images.new_tensor(0.0)
        for adapter, student_feat, teacher_feat in zip(
            self.hint_adapters_encoder, student_hints.encoder_feats, teacher_hints.encoder_feats
        ):
            hint_loss = hint_loss + _cosine_hint_loss(adapter(student_feat), teacher_feat.detach())
        for adapter, student_feat, teacher_feat in zip(
            self.hint_adapters_decoder, student_hints.decoder_feats, teacher_hints.decoder_feats
        ):
            hint_loss = hint_loss + _cosine_hint_loss(adapter(student_feat), teacher_feat.detach())

        logit_kd_loss = logit_stand_kd_loss(
            teacher_hints.logits.detach(), student_hints.logits, self.logit_kd_temperature
        )

        return student_hints.pred, hint_loss, logit_kd_loss

    def _route_predict_batch(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_bool = mask.bool()
        batch_size = images.size(0)
        output_slots: list[torch.Tensor | None] = [None] * batch_size

        for pattern in torch.unique(mask_bool, dim=0):
            row_indices = (mask_bool == pattern).all(dim=1).nonzero(as_tuple=True)[0]
            channel_indices = pattern.nonzero(as_tuple=True)[0]
            sub_images = images.index_select(0, row_indices).index_select(1, channel_indices)
            sub_mask = torch.ones(
                row_indices.numel(), channel_indices.numel(), dtype=torch.bool, device=images.device
            )
            submodel = self.submodels[_pattern_key(pattern.tolist())]
            sub_pred = submodel(sub_images, sub_mask)
            for local_index, global_index in enumerate(row_indices.tolist()):
                output_slots[global_index] = sub_pred[local_index]

        return torch.stack(output_slots, dim=0)

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._validate_boundary_inputs(images, mask)
        original_shape = tuple(int(dim) for dim in images.shape[2:])

        target_shape = tuple(
            max(current, minimum) for current, minimum in zip(original_shape, self.tile_shape)
        )
        pad_before: list[int] = []
        pad_sizes: list[int] = []
        for current, target in zip(reversed(original_shape), reversed(target_shape)):
            total_pad = max(target - current, 0)
            before = total_pad // 2
            after = total_pad - before
            pad_before.append(before)
            pad_sizes.extend((before, after))
        pad_before.reverse()

        if any(pad_sizes):
            images = torch.nn.functional.pad(images, tuple(pad_sizes))

        padded_shape = tuple(int(dim) for dim in images.shape[2:])

        if padded_shape == self.tile_shape:
            prediction = self._route_predict_batch(images, mask)
            return TinyMimosa._crop_to_original_shape(prediction, pad_before, original_shape)

        prediction = torch.zeros(
            images.size(0), self.num_cls, *padded_shape, device=images.device, dtype=images.dtype
        )
        weight = torch.zeros(images.size(0), 1, *padded_shape, device=images.device, dtype=images.dtype)

        h_starts = TinyMimosa._window_starts(padded_shape[0], self.tile_shape[0], self.overlap)
        w_starts = TinyMimosa._window_starts(padded_shape[1], self.tile_shape[1], self.overlap)
        d_starts = TinyMimosa._window_starts(padded_shape[2], self.tile_shape[2], self.overlap)

        for h in h_starts:
            for w in w_starts:
                for d in d_starts:
                    patch = images[
                        :, :, h : h + self.tile_shape[0], w : w + self.tile_shape[1], d : d + self.tile_shape[2]
                    ]
                    patch_pred = self._route_predict_batch(patch, mask)
                    prediction[
                        :, :, h : h + self.tile_shape[0], w : w + self.tile_shape[1], d : d + self.tile_shape[2]
                    ] += patch_pred
                    weight[
                        :, :, h : h + self.tile_shape[0], w : w + self.tile_shape[1], d : d + self.tile_shape[2]
                    ] += 1

        prediction = prediction / weight.clamp_min(1)
        return TinyMimosa._crop_to_original_shape(prediction, pad_before, original_shape)

    def predict_teacher(self, images: torch.Tensor) -> torch.Tensor:
        full_mask = torch.ones(images.size(0), self.num_modals, dtype=torch.bool, device=images.device)
        return self.teacher.predict(images, full_mask)

    def model_for_export(self) -> ManyMimosas:
        export_model = ManyMimosas(
            num_cls=self.num_cls,
            num_modals=self.num_modals,
            input_shape=self.input_shape,
            features_per_stage=self.student_features_per_stage,
            n_conv_per_stage=self.n_conv_per_stage,
            n_conv_per_stage_decoder=self.n_conv_per_stage_decoder,
            overlap=self.overlap,
        )
        export_model.submodels.load_state_dict(self.submodels.state_dict())
        export_model._mimose_model_kwargs = {
            "num_cls": self.num_cls,
            "num_modals": self.num_modals,
            "input_shape": self.input_shape,
            "features_per_stage": self.student_features_per_stage,
            "n_conv_per_stage": self.n_conv_per_stage,
            "n_conv_per_stage_decoder": self.n_conv_per_stage_decoder,
            "overlap": self.overlap,
        }
        export_model._mimose_model_name = ManyMimosas.__name__
        return export_model

    def _validate_boundary_inputs(self, images: torch.Tensor, mask: torch.Tensor) -> None:
        if images.ndim != 5 or images.size(1) != self.num_modals:
            raise RuntimeError(
                f"ManyMimosasKD expects images with shape [B, {self.num_modals}, H, W, D], "
                f"got {tuple(images.shape)}"
            )
        if mask.ndim != 2 or mask.size(0) != images.size(0) or mask.size(1) != self.num_modals:
            raise RuntimeError(
                f"ManyMimosasKD expects mask shape [B, {self.num_modals}], got {tuple(mask.shape)}"
            )
        if not mask.bool().any(dim=1).all():
            raise RuntimeError("ManyMimosasKD requires at least one available modality per sample")


Model = ManyMimosasKD


__all__ = ["ManyMimosasKD", "Model", "DEFAULT_STUDENT_FEATURES_PER_STAGE", "DEFAULT_TEACHER_FEATURES_PER_STAGE"]
