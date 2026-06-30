from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from mimose.models.abstract_model import AbstractModel
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import (
    convert_dim_to_conv_op,
    get_matching_instancenorm,
)


NUM_MODALITIES = 4
DEFAULT_NUM_CLASSES = 4
DEFAULT_INPUT_SHAPE = (182, 218, 182)


class LatentMaskConditioning(nn.Module):
    def __init__(self, modality_count: int, latent_channels: int) -> None:
        super().__init__()
        hidden = max(latent_channels // 2, modality_count * 2)
        self.mlp = nn.Sequential(
            nn.Linear(modality_count, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_channels * 2),
        )

    def forward(self, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conditioning = self.mlp(mask.float())
        scale, bias = conditioning.chunk(2, dim=1)
        return scale[..., None, None, None], bias[..., None, None, None]


class TinyMimosa(AbstractModel):
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
        resolved_input_shape = tuple(int(dim) for dim in input_shape)
        if len(resolved_input_shape) != 3 or any(dim <= 0 for dim in resolved_input_shape):
            raise ValueError(
                "TinyMimosa input_shape must contain three positive spatial dimensions"
            )
        if num_modals != NUM_MODALITIES:
            raise ValueError(f"TinyMimosa expects exactly {NUM_MODALITIES} modalities")
        if not features_per_stage:
            raise ValueError("TinyMimosa features_per_stage cannot be empty")
        if len(features_per_stage) < 4:
            raise ValueError("TinyMimosa needs at least four feature stages for auxiliary heads")
        if not (0.0 <= overlap < 1.0):
            raise ValueError("TinyMimosa overlap must be in the range [0, 1)")

        self.num_cls = int(num_cls)
        self.num_modals = int(num_modals)
        self.input_shape = resolved_input_shape
        self.features_per_stage = tuple(int(ch) for ch in features_per_stage)
        self.overlap = float(overlap)
        self.is_training = False
        self.spatial_multiple = 2 ** (len(self.features_per_stage) - 1)
        self.tile_shape = tuple(
            self._round_up_to_multiple(dim, self.spatial_multiple)
            for dim in self.input_shape
        )

        conv_op = convert_dim_to_conv_op(3)
        norm_op = get_matching_instancenorm(conv_op)
        self.backbone = PlainConvUNet(
            input_channels=self.num_modals,
            n_stages=len(self.features_per_stage),
            features_per_stage=list(self.features_per_stage),
            conv_op=conv_op,
            kernel_sizes=[3] * len(self.features_per_stage),
            strides=[1] + [2] * (len(self.features_per_stage) - 1),
            n_conv_per_stage=n_conv_per_stage,
            num_classes=self.num_cls,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=True,
            norm_op=norm_op,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
            deep_supervision=False,
        )

        self.mask_conditioning = LatentMaskConditioning(
            self.num_modals,
            self.features_per_stage[-1],
        )

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[()], tuple[()]]:
        self._validate_boundary_inputs(images, mask)
        if tuple(int(dim) for dim in images.shape[2:]) != self.tile_shape:
            raise RuntimeError(
                "TinyMimosa forward expects spatial shape "
                f"{self.tile_shape}, got {tuple(images.shape[2:])}"
            )

        skips = self.backbone.encoder(images)
        latent = skips[-1]
        scale, bias = self.mask_conditioning(mask)
        skips[-1] = latent * (1.0 + torch.tanh(scale)) + bias

        fuse_logits = self.backbone.decoder(skips)
        fuse_pred = torch.softmax(fuse_logits, dim=1)
        if self.is_training:
            return fuse_pred, (), ()
        return fuse_pred

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._validate_boundary_inputs(images, mask)
        padded_shape = tuple(int(dim) for dim in images.shape[2:])

        if padded_shape == self.tile_shape:
            return self(images, mask)

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

        h_starts = self._window_starts(
            padded_shape[0],
            self.tile_shape[0],
            self.overlap,
        )
        w_starts = self._window_starts(
            padded_shape[1],
            self.tile_shape[1],
            self.overlap,
        )
        d_starts = self._window_starts(
            padded_shape[2],
            self.tile_shape[2],
            self.overlap,
        )

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

        return prediction / weight.clamp_min(1)

    def _validate_boundary_inputs(self, images: torch.Tensor, mask: torch.Tensor) -> None:
        if images.ndim != 5 or images.size(1) != self.num_modals:
            raise RuntimeError(
                f"TinyMimosa expects images with shape [B, {self.num_modals}, H, W, D], "
                f"got {tuple(images.shape)}"
            )
        if mask.ndim != 2 or mask.size(0) != images.size(0) or mask.size(1) != self.num_modals:
            raise RuntimeError(
                f"TinyMimosa expects mask shape [B, {self.num_modals}], got {tuple(mask.shape)}"
            )

    @staticmethod
    def _round_up_to_multiple(value: int, multiple: int) -> int:
        if multiple <= 0:
            raise ValueError("multiple must be positive")
        return ((value + multiple - 1) // multiple) * multiple

    @staticmethod
    def _window_starts(size: int, window: int, overlap: float) -> list[int]:
        if size < window:
            raise RuntimeError(
                "TinyMimosa sliding-window inference expects padded dimensions to be at least "
                f"{window}, got {size}"
            )
        if size == window:
            return [0]

        stride = max(int(window * (1.0 - overlap)), 1)
        starts = list(range(0, max(size - window, 0), stride))
        last_start = size - window
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        return starts

    @staticmethod
    def _remap_input_order(
        images: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images.size(1) != NUM_MODALITIES:
            raise RuntimeError(
                f"TinyMimosa expects {NUM_MODALITIES} input modalities, got {images.size(1)}"
            )
        if mask.ndim != 2 or mask.size(1) != NUM_MODALITIES:
            raise RuntimeError(
                f"TinyMimosa expects mask shape [B, {NUM_MODALITIES}], got {tuple(mask.shape)}"
            )

        return images, mask


Model = TinyMimosa


__all__ = ["TinyMimosa", "Model"]
