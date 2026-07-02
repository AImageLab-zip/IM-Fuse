from __future__ import annotations

import torch
import torch.nn as nn

from mimose.models.abstract_model import AbstractModel

basic_dims = 16
num_modals = 4
input_patch_size = 80
DATASET_MODALITY_ORDER = (2, 0, 1, 3)

A2FSegOutput = torch.Tensor | tuple[
    torch.Tensor,
    tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


def normalization(planes: int, norm: str = "in") -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm3d(planes)
    if norm == "gn":
        return nn.GroupNorm(4, planes)
    if norm == "in":
        return nn.InstanceNorm3d(planes)
    raise ValueError(f"normalization type {norm} is not supported")


class GeneralConv3d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pad_type: str = "reflect",
        norm: str = "in",
        act_type: str = "lrelu",
        relufactor: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=k_size,
            stride=stride,
            padding=padding,
            padding_mode=pad_type,
            bias=True,
        )
        self.norm = normalization(out_ch, norm=norm)
        if act_type == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif act_type == "lrelu":
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)
        else:
            raise ValueError(f"activation type {act_type} is not supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)


class ModalityEncoder(nn.Module):
    """Per-modality independent encoder (mirrors one instance of the legacy
    per-modality ``Generic_UNet`` encoder path)."""

    def __init__(self) -> None:
        super().__init__()
        self.e1_c1 = GeneralConv3d(1, basic_dims)
        self.e1_c2 = GeneralConv3d(basic_dims, basic_dims)
        self.e1_c3 = GeneralConv3d(basic_dims, basic_dims)

        self.e2_c1 = GeneralConv3d(basic_dims, basic_dims * 2, stride=2)
        self.e2_c2 = GeneralConv3d(basic_dims * 2, basic_dims * 2)
        self.e2_c3 = GeneralConv3d(basic_dims * 2, basic_dims * 2)

        self.e3_c1 = GeneralConv3d(basic_dims * 2, basic_dims * 4, stride=2)
        self.e3_c2 = GeneralConv3d(basic_dims * 4, basic_dims * 4)
        self.e3_c3 = GeneralConv3d(basic_dims * 4, basic_dims * 4)

        self.e4_c1 = GeneralConv3d(basic_dims * 4, basic_dims * 8, stride=2)
        self.e4_c2 = GeneralConv3d(basic_dims * 8, basic_dims * 8)
        self.e4_c3 = GeneralConv3d(basic_dims * 8, basic_dims * 8)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))
        return x1, x2, x3, x4


class _DecoderBase(nn.Module):
    """Shared decoder topology used both per-modality and for the fusion
    branch (mirrors the localization pathway of the legacy ``Generic_UNet``
    / ``Generic_UNet_Decoder``, deep-supervision at every resolution)."""

    def __init__(self, num_cls: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.d3_c1 = GeneralConv3d(basic_dims * 8, basic_dims * 4)
        self.d3_c2 = GeneralConv3d(basic_dims * 8, basic_dims * 4)
        self.d3_out = GeneralConv3d(basic_dims * 4, basic_dims * 4, k_size=1, padding=0)
        self.d3_seg = nn.Conv3d(basic_dims * 4, num_cls, kernel_size=1, bias=True)

        self.d2_c1 = GeneralConv3d(basic_dims * 4, basic_dims * 2)
        self.d2_c2 = GeneralConv3d(basic_dims * 4, basic_dims * 2)
        self.d2_out = GeneralConv3d(basic_dims * 2, basic_dims * 2, k_size=1, padding=0)
        self.d2_seg = nn.Conv3d(basic_dims * 2, num_cls, kernel_size=1, bias=True)

        self.d1_c1 = GeneralConv3d(basic_dims * 2, basic_dims)
        self.d1_c2 = GeneralConv3d(basic_dims * 2, basic_dims)
        self.d1_out = GeneralConv3d(basic_dims, basic_dims, k_size=1, padding=0)
        self.d1_seg = nn.Conv3d(basic_dims, num_cls, kernel_size=1, bias=True)

        self.softmax = nn.Softmax(dim=1)
        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        de_x4 = self.d3_c1(self.up(x4))
        de_x3 = self.d3_out(self.d3_c2(torch.cat((de_x4, x3), dim=1)))
        seg3 = self.softmax(self.d3_seg(de_x3))

        de_x3 = self.d2_c1(self.up(de_x3))
        de_x2 = self.d2_out(self.d2_c2(torch.cat((de_x3, x2), dim=1)))
        seg2 = self.softmax(self.d2_seg(de_x2))

        de_x2 = self.d1_c1(self.up(de_x2))
        de_x1 = self.d1_out(self.d1_c2(torch.cat((de_x2, x1), dim=1)))
        seg1 = self.softmax(self.d1_seg(de_x1))

        return de_x1, de_x1, (seg1, self.up2(seg2), self.up4(seg3))


class A2FSeg(AbstractModel):
    """Faithful port of A2FSeg (adaptive attention-gated multi-modal fusion):
    each modality is processed by an independent encoder-decoder, a shared
    fusion decoder decodes the (masked) average of per-modality features,
    and a modality-aware attention module adaptively re-weights the
    per-modality decoder features before the final segmentation head.

    Unlike the legacy implementation (which hard-drops whole modalities per
    *batch* by skipping their encoder/decoder entirely), this port always
    runs every modality's encoder/decoder and instead masks per-*sample*
    contributions before averaging/attention, so it is compatible with the
    framework's per-sample ``mask`` convention used by every other model.
    """

    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.num_cls = num_cls
        self._mimose_model_kwargs = {"num_cls": num_cls}

        self.encoders = nn.ModuleList([ModalityEncoder() for _ in range(num_modals)])
        self.modality_decoders = nn.ModuleList(
            [_DecoderBase(num_cls) for _ in range(num_modals)]
        )
        self.fusion_decoder = _DecoderBase(num_cls)

        self.modality_aware_modules = nn.ModuleList(
            [
                nn.Sequential(
                    GeneralConv3d(basic_dims * 2, basic_dims, k_size=3, padding=1),
                    nn.Conv3d(basic_dims, basic_dims, kernel_size=3, padding=1, bias=True),
                )
                for _ in range(num_modals)
            ]
        )
        self.final_conv = GeneralConv3d(basic_dims * 2, basic_dims, k_size=3, padding=1)
        self.output_conv = nn.Conv3d(basic_dims, num_cls, kernel_size=1, bias=False)

        self.is_training = False

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                torch.nn.init.kaiming_normal_(module.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> A2FSegOutput:
        x, mask = self._remap_input_order(x, mask)
        mask_f = mask.float()
        present_count = mask_f.sum(dim=1).clamp_min(1.0)

        def masked_mean(features: list[torch.Tensor]) -> torch.Tensor:
            stacked = torch.stack(features, dim=1)
            weight = mask_f.view(mask_f.size(0), num_modals, 1, 1, 1, 1)
            return (stacked * weight).sum(dim=1) / present_count.view(
                present_count.size(0), 1, 1, 1, 1
            )

        encoder_scales: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        modality_features: list[torch.Tensor] = []
        sep_preds: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for modal_index in range(num_modals):
            x1, x2, x3, x4 = self.encoders[modal_index](x[:, modal_index : modal_index + 1])
            encoder_scales.append((x1, x2, x3, x4))
            feature, _, ds_preds = self.modality_decoders[modal_index](x1, x2, x3, x4)
            modality_features.append(feature)
            sep_preds.append(ds_preds)

        fused_x1 = masked_mean([scales[0] for scales in encoder_scales])
        fused_x2 = masked_mean([scales[1] for scales in encoder_scales])
        fused_x3 = masked_mean([scales[2] for scales in encoder_scales])
        fused_x4 = masked_mean([scales[3] for scales in encoder_scales])
        fusion_feature, _, fusion_ds_preds = self.fusion_decoder(
            fused_x1, fused_x2, fused_x3, fused_x4
        )

        attn_logits = []
        for modal_index in range(num_modals):
            attn_logits.append(
                self.modality_aware_modules[modal_index](
                    torch.cat([modality_features[modal_index], fusion_feature], dim=1)
                )
            )
        attn_stack = torch.stack(attn_logits, dim=1)
        gate = mask.view(mask.size(0), num_modals, 1, 1, 1, 1)
        attn_stack = attn_stack.masked_fill(~gate, -1e4)
        attn = torch.softmax(attn_stack, dim=1)

        features_stack = torch.stack(modality_features, dim=1)
        weighted = (attn * features_stack).sum(dim=1)

        combined = self.final_conv(torch.cat([weighted, fusion_feature], dim=1))
        fused_output = self.output_conv(combined)

        if self.is_training:
            return fused_output, tuple(sep_preds), fusion_ds_preds
        return torch.softmax(fused_output, dim=1)

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, _, height, width, depth = images.shape
        if (height, width, depth) == (input_patch_size, input_patch_size, input_patch_size):
            return self(images, mask)

        h_starts = self._window_starts(height)
        w_starts = self._window_starts(width)
        d_starts = self._window_starts(depth)
        prediction = torch.zeros(
            images.size(0), self.num_cls, height, width, depth, device=images.device
        )
        weight = torch.zeros(images.size(0), 1, height, width, depth, device=images.device)

        for h in h_starts:
            for w in w_starts:
                for d in d_starts:
                    patch = images[
                        :, :, h : h + input_patch_size, w : w + input_patch_size, d : d + input_patch_size
                    ]
                    patch_pred = self(patch, mask)
                    prediction[
                        :, :, h : h + input_patch_size, w : w + input_patch_size, d : d + input_patch_size
                    ] += patch_pred
                    weight[
                        :, :, h : h + input_patch_size, w : w + input_patch_size, d : d + input_patch_size
                    ] += 1
        return prediction / weight

    @staticmethod
    def _window_starts(size: int) -> list[int]:
        if size < input_patch_size:
            raise RuntimeError(
                f"Model prediction expects spatial dimensions to be at least {input_patch_size}, got {size}"
            )
        if size == input_patch_size:
            return [0]

        stride = input_patch_size // 2
        starts = list(range(0, max(size - input_patch_size, 0), stride))
        last_start = size - input_patch_size
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        return starts

    @staticmethod
    def _remap_input_order(
        images: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images.size(1) != num_modals:
            raise RuntimeError(f"A2FSeg expects {num_modals} input modalities, got {images.size(1)}")
        if mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(f"A2FSeg expects mask shape [B, {num_modals}], got {tuple(mask.shape)}")

        return images[:, DATASET_MODALITY_ORDER, ...], mask[:, DATASET_MODALITY_ORDER]


Model = A2FSeg


__all__ = ["Model", "A2FSeg"]
