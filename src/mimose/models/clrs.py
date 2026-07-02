from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimose.models.abstract_model import AbstractModel

n_filters = 16
num_modals = 4
input_patch_size = 80
DATASET_MODALITY_ORDER = (2, 0, 1, 3)
bottleneck_dims = n_filters * 16

CLRSOutput = torch.Tensor | tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
]


def normalization(planes: int, norm: str = "bn") -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm3d(planes)
    if norm == "gn":
        return nn.GroupNorm(4, planes)
    if norm == "in":
        return nn.InstanceNorm3d(planes)
    raise ValueError(f"normalization type {norm} is not supported")


def split_into_list_general(total_size: int, num_parts: int) -> list[tuple[int, int]]:
    part_size = total_size // num_parts
    remainder = total_size % num_parts
    result = []
    start = 0
    for i in range(num_parts):
        end = start + part_size
        if i < remainder:
            end += 1
        result.append((start, end))
        start = end
    return result


class ConvBlock(nn.Module):
    """VNet-style stacked conv block (mirrors legacy ``ConvBlock``)."""

    def __init__(self, n_stages: int, n_filters_in: int, n_filters_out: int) -> None:
        super().__init__()
        ops = []
        for i in range(n_stages):
            input_channel = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            ops.append(normalization(n_filters_out))
            ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in: int, n_filters_out: int, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride),
            normalization(n_filters_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsamplingFunction(nn.Module):
    def __init__(self, n_filters_in: int, n_filters_out: int, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True),
            nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1),
            normalization(n_filters_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Encoder(nn.Module):
    """Shared VNet-style 5-level encoder over the channel-stacked modalities."""

    def __init__(self, n_channels: int = num_modals) -> None:
        super().__init__()
        self.pre_process = nn.Sequential(
            nn.Conv3d(n_channels, n_filters, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm3d(n_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
        )

        self.block_one = ConvBlock(1, n_filters, n_filters)
        self.block_one_dw = DownsamplingConvBlock(n_filters, n_filters * 2)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.pre_process(x)

        x1 = self.block_one(x)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        return [x1, x2, x3, x4, x5]


class Decoder(nn.Module):
    """Shared VNet-style decoder with additive skip connections."""

    def __init__(self) -> None:
        super().__init__()
        self.block_five_up = UpsamplingFunction(n_filters * 16, n_filters * 8)
        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8)
        self.block_six_up = UpsamplingFunction(n_filters * 8, n_filters * 4)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4)
        self.block_seven_up = UpsamplingFunction(n_filters * 4, n_filters * 2)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2)
        self.block_eight_up = UpsamplingFunction(n_filters * 2, n_filters)

        self.block_nine = ConvBlock(1, n_filters, n_filters)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x1, x2, x3, x4, x5 = features

        x5_up = self.block_five_up(x5) + x4
        x6 = self.block_six(x5_up)

        x6_up = self.block_six_up(x6) + x3
        x7 = self.block_seven(x6_up)

        x7_up = self.block_seven_up(x7) + x2
        x8 = self.block_eight(x7_up)

        x8_up = self.block_eight_up(x8) + x1
        return self.block_nine(x8_up)


class StarReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.relu(x) ** 2 + self.bias


class LayerNormChannelsLast(nn.Module):
    """Mirrors legacy ``LayerNormGeneral`` (bias-free, normalized over the
    channel-last dimension)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(-1, keepdim=True)
        variance = centered.pow(2).mean(-1, keepdim=True)
        return (centered / torch.sqrt(variance + self.eps)) * self.weight


class SepConv(nn.Module):
    """Depthwise-separable channel-mixing token mixer (mirrors legacy
    ``SepConv``); operates on channels-last ``[B, H, W, D, C]`` tensors."""

    def __init__(self, dim: int, expansion_ratio: float = 2.0, kernel_size: int = 3) -> None:
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=False)
        self.act1 = StarReLU()
        self.dwconv = nn.Conv3d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=med_channels, bias=False,
        )
        self.pwconv2 = nn.Linear(med_channels, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.pwconv1(x))
        x = x.permute(0, 4, 1, 2, 3)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)
        return self.pwconv2(x)


class EfficientAttention(nn.Module):
    """Linear-complexity attention token mixer (mirrors legacy
    ``EfficientAttention``); operates on channels-last tensors."""

    def __init__(self, dim: int, head_count: int = 1) -> None:
        super().__init__()
        self.head_count = head_count
        self.key_channels = dim
        self.value_channels = dim
        self.keys = nn.Conv3d(dim, dim, 1)
        self.queries = nn.Conv3d(dim, dim, 1)
        self.values = nn.Conv3d(dim, dim, 1)
        self.reprojection = nn.Conv3d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 4, 1, 2, 3)
        n, _, h, w, d = x.shape
        keys = self.keys(x).reshape(n, self.key_channels, h * w * d)
        queries = self.queries(x).reshape(n, self.key_channels, h * w * d)
        values = self.values(x).reshape(n, self.value_channels, h * w * d)

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w, d)
            attended_values.append(attended_value)

        aggregated = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated)
        return attention.permute(0, 2, 3, 4, 1)


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.act = StarReLU()
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """MetaFormer-style block: ``x + token_mixer(norm(x))``, then
    ``x + mlp(norm(x))``, operating on channels-last tensors."""

    def __init__(self, dim: int, token_mixer: nn.Module) -> None:
        super().__init__()
        self.norm1 = LayerNormChannelsLast(dim)
        self.token_mixer = token_mixer
        self.norm2 = LayerNormChannelsLast(dim)
        self.mlp = Mlp(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def build_spec_transform(dim: int, mixer_type: str, depths: int = 2) -> nn.Sequential:
    if mixer_type == "sepconv":
        blocks = [Block(dim, SepConv(dim)) for _ in range(depths)]
    elif mixer_type == "efficient_attention":
        blocks = [Block(dim, EfficientAttention(dim)) for _ in range(depths)]
    else:
        raise ValueError(f"unsupported spec transform type: {mixer_type}")
    return nn.Sequential(*blocks)


class FusionTransform(nn.Module):
    """Mirrors legacy ``get_fusion_transform``: conv-project the concatenated
    bottleneck + spec-fusion features back down to the bottleneck width."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = LayerNormChannelsLast(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 4, 1, 2, 3)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 4, 1)
        return self.norm(x)


class SpecificClassification(nn.Module):
    """Per-modality gated-attention module (mirrors legacy
    ``SpecificClassification``). The legacy ``fc`` layer size
    (``n_channels * batch_size``) and unbatched ``flatten()`` only work for
    ``batch_size == 1``; here the pooled feature is flattened per-sample
    (``flatten(1)``) so the module supports arbitrary batch sizes."""

    def __init__(self, n_channels: int, n_classes: int, norm: str = "bn") -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv3d(n_channels, n_channels, kernel_size=1),
            normalization(n_channels, norm),
            nn.AdaptiveMaxPool3d(1),
        )
        self.branch2 = nn.Sequential(
            nn.Conv3d(n_channels, n_channels, kernel_size=1),
            normalization(n_channels, norm),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_channels, n_channels, kernel_size=1),
            normalization(n_channels, norm),
        )
        self.branch3 = nn.Sequential(
            nn.Conv3d(n_channels, n_channels, kernel_size=1),
            normalization(n_channels, norm),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_channels, n_channels, kernel_size=1),
            normalization(n_channels, norm),
        )
        self.branch4 = nn.Sequential(
            nn.Conv3d(n_channels, n_channels, kernel_size=1),
            normalization(n_channels, norm),
            nn.AdaptiveAvgPool3d(1),
        )
        self.maxpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(n_channels, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
        )
        self.cls_head = nn.Linear(512, n_classes)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ft1 = self.branch1(features)
        ft2 = self.branch2(features)
        ft3 = self.branch3(features)
        ft4 = self.branch4(features)

        max_fuse_fm = ft2 * ft1
        avg_fuse_fm = ft3 * ft4
        add_fuse_fm = max_fuse_fm + avg_fuse_fm

        pooled = self.maxpool(add_fuse_fm).flatten(1)
        proj_feature = self.fc(pooled)
        cls_feature = self.cls_head(proj_feature)
        return cls_feature, add_fuse_fm, proj_feature


class CLRS(AbstractModel):
    """Port of CLRS (Cyclic Contrastive Representation Learning): a shared
    VNet-style encoder-decoder over channel-stacked modalities, with a
    transformer-based bottleneck decomposition into per-modality
    representations (spec_transform), per-modality gated-attention
    classification/contrastive heads, and a residual fusion back into the
    bottleneck before decoding.

    Unlike the legacy trainer (which zeroes modalities uniformly across the
    batch via an epoch-scheduled curriculum), this port zeroes modality
    channels per-*sample* from the framework's ``mask`` tensor, matching the
    convention used by every other model in this codebase.
    """

    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.num_cls = num_cls
        self._mimose_model_kwargs = {"num_cls": num_cls}

        self.encoder = Encoder(n_channels=num_modals)
        self.decoder = Decoder()
        self.output_conv = nn.Conv3d(n_filters, num_cls, kernel_size=1)

        self.modality_vec_list = split_into_list_general(bottleneck_dims, num_modals)
        self.spec_transform = build_spec_transform(bottleneck_dims, "sepconv")
        self.spec_transform02 = build_spec_transform(bottleneck_dims, "efficient_attention")
        self.fusion_transform = FusionTransform(bottleneck_dims * 2, bottleneck_dims)

        self.spec_module_list = nn.ModuleList(
            [
                SpecificClassification(n_channels=end - start, n_classes=num_modals)
                for start, end in self.modality_vec_list
            ]
        )

        self.is_training = False

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                torch.nn.init.kaiming_normal_(module.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> CLRSOutput:
        x, mask = self._remap_input_order(x, mask)
        x = x * mask.view(mask.size(0), num_modals, 1, 1, 1).float()

        features = self.encoder(x)
        bottleneck = features[-1]

        spec_map = self.spec_transform(bottleneck.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

        logit_list = []
        proj_feature_list = []
        fuse_fm_list = []
        generated_map = []
        for modal_index, (start, end) in enumerate(self.modality_vec_list):
            single_modal = spec_map[:, start:end]
            generated_map.append(single_modal)
            logit, add_fuse_fm, proj_feature = self.spec_module_list[modal_index](single_modal)
            logit_list.append(logit)
            proj_feature_list.append(proj_feature)
            fuse_fm_list.append(add_fuse_fm)

        spec_fm_cat = torch.cat(fuse_fm_list, dim=1)
        spec_fm_cat = self.spec_transform02(spec_fm_cat.permute(0, 2, 3, 4, 1))

        fuse_map = self.fusion_transform(
            torch.cat([bottleneck.permute(0, 2, 3, 4, 1), spec_fm_cat], dim=-1)
        )
        features[-1] = bottleneck + fuse_map.permute(0, 4, 1, 2, 3)

        decoded = self.decoder(features)
        fused_logits = self.output_conv(decoded)

        spec_logits_cat = torch.stack(logit_list, dim=0)
        spec_info_vector = torch.stack(proj_feature_list, dim=0)

        if self.is_training:
            return fused_logits, spec_logits_cat, spec_info_vector, generated_map
        return torch.softmax(fused_logits, dim=1)

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
            raise RuntimeError(f"CLRS expects {num_modals} input modalities, got {images.size(1)}")
        if mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(f"CLRS expects mask shape [B, {num_modals}], got {tuple(mask.shape)}")

        return images[:, DATASET_MODALITY_ORDER, ...], mask[:, DATASET_MODALITY_ORDER]


Model = CLRS


__all__ = ["Model", "CLRS"]
