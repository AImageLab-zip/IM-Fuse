from __future__ import annotations

import itertools
from collections.abc import Sequence

import torch
import torch.nn as nn

from mimose.models.abstract_model import AbstractModel
from mimose.models.tiny_mimosa import TinyMimosa


NUM_MODALITIES = 4
DEFAULT_NUM_CLASSES = 4
DEFAULT_INPUT_SHAPE = (182, 218, 182)


def _all_mask_patterns(num_modalities: int) -> torch.Tensor:
    patterns = []
    for size in range(1, num_modalities + 1):
        for combo in itertools.combinations(range(num_modalities), size):
            pattern = [False] * num_modalities
            for index in combo:
                pattern[index] = True
            patterns.append(pattern)
    return torch.tensor(patterns, dtype=torch.bool)


ALL_MASK_PATTERNS = _all_mask_patterns(NUM_MODALITIES)


def _pattern_key(pattern: Sequence[bool]) -> str:
    return "".join("1" if bool(present) else "0" for present in pattern)


class ManyMimosas(AbstractModel):
    """Wraps one TinyMimosa per non-empty modality-presence pattern.

    Every one of the 2**NUM_MODALITIES - 1 possible missing-modality
    configurations gets its own TinyMimosa, sized down to only the input
    channels that configuration actually has available (1 channel for
    single-modality patterns, up to NUM_MODALITIES for the full set).
    forward()/predict() route each sample to the TinyMimosa matching its
    mask, feeding it only the present channels.
    """

    def __init__(
        self,
        *,
        num_cls: int = DEFAULT_NUM_CLASSES,
        num_modals: int = NUM_MODALITIES,
        input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
        features_per_stage: Sequence[int] = (8, 16, 32, 64),
        n_conv_per_stage: int = 2,
        n_conv_per_stage_decoder: int = 2,
        overlap: float = 0.5,
    ) -> None:
        super().__init__()
        if num_modals != NUM_MODALITIES:
            raise ValueError(f"ManyMimosas expects exactly {NUM_MODALITIES} modalities")

        self.num_cls = int(num_cls)
        self.num_modals = int(num_modals)
        self.overlap = float(overlap)
        self.is_training = False

        self.submodels = nn.ModuleDict(
            {
                _pattern_key(pattern): TinyMimosa(
                    num_cls=num_cls,
                    num_modals=int(pattern.sum().item()),
                    input_shape=input_shape,
                    features_per_stage=features_per_stage,
                    n_conv_per_stage=n_conv_per_stage,
                    n_conv_per_stage_decoder=n_conv_per_stage_decoder,
                    overlap=overlap,
                )
                for pattern in ALL_MASK_PATTERNS
            }
        )
        self.tile_shape = next(iter(self.submodels.values())).tile_shape

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[()], tuple[()]]:
        self._validate_boundary_inputs(images, mask)
        for submodel in self.submodels.values():
            submodel.is_training = self.is_training

        mask_bool = mask.bool()
        batch_size = images.size(0)
        output_slots: list[torch.Tensor | None] = [None] * batch_size

        for pattern in torch.unique(mask_bool, dim=0):
            row_indices = (mask_bool == pattern).all(dim=1).nonzero(as_tuple=True)[0]
            channel_indices = pattern.nonzero(as_tuple=True)[0]
            sub_images = images.index_select(0, row_indices).index_select(1, channel_indices)
            sub_mask = torch.ones(
                row_indices.numel(),
                channel_indices.numel(),
                dtype=torch.bool,
                device=images.device,
            )
            submodel = self.submodels[_pattern_key(pattern.tolist())]
            sub_output = submodel(sub_images, sub_mask)
            sub_pred = sub_output[0] if self.is_training else sub_output
            for local_index, global_index in enumerate(row_indices.tolist()):
                output_slots[global_index] = sub_pred[local_index]

        fuse_pred = torch.stack(output_slots, dim=0)
        if self.is_training:
            return fuse_pred, (), ()
        return fuse_pred

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
            prediction = self(images, mask)
            return TinyMimosa._crop_to_original_shape(prediction, pad_before, original_shape)

        prediction = torch.zeros(
            images.size(0),
            self.num_cls,
            *padded_shape,
            device=images.device,
            dtype=images.dtype,
        )
        weight = torch.zeros(
            images.size(0),
            1,
            *padded_shape,
            device=images.device,
            dtype=images.dtype,
        )

        h_starts = TinyMimosa._window_starts(padded_shape[0], self.tile_shape[0], self.overlap)
        w_starts = TinyMimosa._window_starts(padded_shape[1], self.tile_shape[1], self.overlap)
        d_starts = TinyMimosa._window_starts(padded_shape[2], self.tile_shape[2], self.overlap)

        for h in h_starts:
            for w in w_starts:
                for d in d_starts:
                    patch = images[
                        :,
                        :,
                        h : h + self.tile_shape[0],
                        w : w + self.tile_shape[1],
                        d : d + self.tile_shape[2],
                    ]
                    patch_pred = self(patch, mask)
                    prediction[
                        :,
                        :,
                        h : h + self.tile_shape[0],
                        w : w + self.tile_shape[1],
                        d : d + self.tile_shape[2],
                    ] += patch_pred
                    weight[
                        :,
                        :,
                        h : h + self.tile_shape[0],
                        w : w + self.tile_shape[1],
                        d : d + self.tile_shape[2],
                    ] += 1

        prediction = prediction / weight.clamp_min(1)
        return TinyMimosa._crop_to_original_shape(prediction, pad_before, original_shape)

    def _validate_boundary_inputs(self, images: torch.Tensor, mask: torch.Tensor) -> None:
        if images.ndim != 5 or images.size(1) != self.num_modals:
            raise RuntimeError(
                f"ManyMimosas expects images with shape [B, {self.num_modals}, H, W, D], "
                f"got {tuple(images.shape)}"
            )
        if mask.ndim != 2 or mask.size(0) != images.size(0) or mask.size(1) != self.num_modals:
            raise RuntimeError(
                f"ManyMimosas expects mask shape [B, {self.num_modals}], got {tuple(mask.shape)}"
            )
        if not mask.bool().any(dim=1).all():
            raise RuntimeError("ManyMimosas requires at least one available modality per sample")


Model = ManyMimosas


__all__ = ["ManyMimosas", "Model", "ALL_MASK_PATTERNS"]
