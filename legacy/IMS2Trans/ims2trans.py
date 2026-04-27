import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import torch
import math
from layers import general_conv3d_prenorm, fusion_prenorm
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import numpy as np
import itertools
from monai.networks.blocks import UnetrBasicBlock
from torch.utils.checkpoint import checkpoint


basic_dims = 8
transformer_basic_dims = 192
mlp_dim = 4096
num_heads = 8
depth = 1
num_modals = 4
patch_size = 8


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size=2,
        in_chans=1,
        embed_dim=48,
        norm_layer=nn.LayerNorm,
        spatial_dims=3,
    ) -> None:
        """
        Args:
            patch_size: dimension of patch size.
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            norm_layer: normalization layer.
            spatial_dims: spatial dimension.
        """

        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            _, _, d, h, w = x_shape
            if w % self.patch_size[2] != 0:
                x = F.pad(x, (0, self.patch_size[2] - w % self.patch_size[2]))
            if h % self.patch_size[1] != 0:
                x = F.pad(x, (0, 0, 0, self.patch_size[1] - h % self.patch_size[1]))
            if d % self.patch_size[0] != 0:
                x = F.pad(
                    x, (0, 0, 0, 0, 0, self.patch_size[0] - d % self.patch_size[0])
                )

        elif len(x_shape) == 4:
            _, _, h, w = x_shape
            if w % self.patch_size[1] != 0:
                x = F.pad(x, (0, self.patch_size[1] - w % self.patch_size[1]))
            if h % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, self.patch_size[0] - h % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            x_shape = x.size()
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            if len(x_shape) == 5:
                d, wh, ww = x_shape[2], x_shape[3], x_shape[4]
                x = x.transpose(1, 2).view(-1, self.embed_dim, d, wh, ww)
            elif len(x_shape) == 4:
                wh, ww = x_shape[2], x_shape[3]
                x = x.transpose(1, 2).view(-1, self.embed_dim, wh, ww)
        return x


class PatchMergingV2(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, spatial_dims=3):
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):

        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x = torch.cat(
                [
                    x[:, i::2, j::2, k::2, :]
                    for i, j, k in itertools.product(range(2), range(2), range(2))
                ],
                -1,
            )

        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x = torch.cat(
                [x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))],
                -1,
            )

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchMerging(PatchMergingV2):
    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 4:
            return super().forward(x)
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        b, d, h, w, c = x_shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


MERGING_MODE = {
    "merging": PatchMerging,
    "mergingv2": PatchMergingV2,
}  # , "expanding": PatchExpanding}


def compute_mask(dims, window_size, shift_size, device):
    cnt = 0
    if len(dims) == 3:
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for d in (
            slice(-window_size[0]),
            slice(-window_size[0], -shift_size[0]),
            slice(-shift_size[0], None),
        ):
            for h in (
                slice(-window_size[1]),
                slice(-window_size[1], -shift_size[1]),
                slice(-shift_size[1], None),
            ):
                for w in (
                    slice(-window_size[2]),
                    slice(-window_size[2], -shift_size[2]),
                    slice(-shift_size[2], None),
                ):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

    elif len(dims) == 2:
        h, w = dims
        img_mask = torch.zeros((1, h, w, 1), device=device)
        for h in (
            slice(-window_size[0]),
            slice(-window_size[0], -shift_size[0]),
            slice(-shift_size[0], None),
        ):
            for w in (
                slice(-window_size[1]),
                slice(-window_size[1], -shift_size[1]),
                slice(-shift_size[1], None),
            ):
                img_mask[:, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )

    return attn_mask


class Mlp(nn.Module):
    def __init__(
        self, hidden_size, mlp_dim, dropout_rate=0.0, act="GELU", dropout_mode="swin"
    ):
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer. If 0, `hidden_size` will be used.
            dropout_rate: faction of the input units to drop.
            act: activation type and arguments. Defaults to GELU.
            dropout_mode: dropout mode, can be "vit" or "swin".
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        mlp_dim = mlp_dim or hidden_size
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.fn = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        dropout_opt = dropout_mode
        if dropout_opt == "vit":
            self.drop2 = nn.Dropout(dropout_rate)
        elif dropout_opt == "swin":
            self.drop2 = self.drop1
        else:
            raise ValueError(f"dropout_mode should be one of SUPPORTED_DROPOUT_MODE")

    def forward(self, x):
        x = self.fn(self.linear1(x))
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x


def window_partition(x, window_size):
    x_shape = x.size()
    if len(x_shape) == 5:
        # print('window_partition:', x.shape)
        b, d, h, w, c = x_shape
        x = x.view(
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 5, 2, 4, 6, 7)
            .contiguous()
            .view(-1, window_size[0] * window_size[1] * window_size[2], c)
        )
    elif len(x_shape) == 4:
        b, h, w, c = x.shape
        x = x.view(
            b,
            h // window_size[0],
            window_size[0],
            w // window_size[1],
            window_size[1],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, window_size[0] * window_size[1], c)
        )
    return windows


def window_reverse(windows, window_size, dims):
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.view(
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    elif len(dims) == 3:
        b, h, w = dims
        x = windows.view(
            b,
            h // window_size[0],
            w // window_size[1],
            window_size[0],
            window_size[1],
            -1,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1)
                    * (2 * self.window_size[1] - 1)
                    * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(
                    torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij")
                )
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (
                2 * self.window_size[2] - 1
            )
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        elif len(self.window_size) == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads
                )
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        # print(x.shape) #torch.Size([1000, 343, 25])
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        # print(q.shape)
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)  # type: ignore
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        shift_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer="GELU",
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.m_token = None
        self.norm1 = norm_layer(self.dim)
        self.normtoken = norm_layer(self.dim)
        self.attn = WindowAttention(
            self.dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(self.dim)
        mlp_hidden_dim = int(self.dim * mlp_ratio)
        self.mlp = Mlp(
            hidden_size=self.dim,
            mlp_dim=mlp_hidden_dim,
            act=act_layer,
            dropout_rate=drop,
            dropout_mode="swin",
        )

    def forward_part1(self, x, mask_matrix, m_token=None):
        x_shape = x.size()
        if m_token is None:
            x = self.norm1(x)
        else:
            x = self.norm1(x) + self.normtoken(m_token)
        if len(x_shape) == 5:
            b, d, h, w, c = x.shape
            window_size, shift_size = get_window_size(
                (d, h, w), self.window_size, self.shift_size
            )
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, dp, hp, wp, _ = x.shape
            dims = [b, dp, hp, wp]

        elif len(x_shape) == 4:
            b, h, w, c = x.shape
            window_size, shift_size = get_window_size(
                (h, w), self.window_size, self.shift_size
            )
            pad_l = pad_t = 0
            pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
            pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, hp, wp, _ = x.shape
            dims = [b, hp, wp]

        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                shifted_x = torch.roll(
                    x,
                    shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                    dims=(1, 2, 3),
                )
            elif len(x_shape) == 4:
                shifted_x = torch.roll(
                    x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2)
                )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        # print('x_windows:',x_windows.shape)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                x = torch.roll(
                    shifted_x,
                    shifts=(shift_size[0], shift_size[1], shift_size[2]),
                    dims=(1, 2, 3),
                )
            elif len(x_shape) == 4:
                x = torch.roll(
                    shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2)
                )
        else:
            x = shifted_x

        if len(x_shape) == 5:
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :d, :h, :w, :].contiguous()
        elif len(x_shape) == 4:
            if pad_r > 0 or pad_b > 0:
                x = x[:, :h, :w, :].contiguous()

        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix, m_token=None):  # todo
        shortcut = x
        if m_token is None:
            x = self.forward_part1(x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix, m_token)
        x = shortcut + self.drop_path(x)
        x = x + self.forward_part2(x)
        return x


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        drop_path,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(
                dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size)
            )

    def forward(self, x, m_token=None):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, c, d, h, w = x_shape
            window_size, shift_size = get_window_size(
                (d, h, w), self.window_size, self.shift_size
            )
            x = rearrange(x, "b c d h w -> b d h w c")
            dp = int(np.ceil(d / window_size[0])) * window_size[0]
            hp = int(np.ceil(h / window_size[1])) * window_size[1]
            wp = int(np.ceil(w / window_size[2])) * window_size[2]
            attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
            if m_token is not None:
                if b > 1 or c > 1:
                    m_token = m_token.expand(b, -1, -1, -1, c)

            for blk in self.blocks:
                if self.use_checkpoint and self.training:
                    if m_token is None:
                        x = checkpoint(lambda _x, _mask, _blk=blk: _blk(_x, _mask, None),
                                    x, attn_mask, use_reentrant=False)
                    else:
                        x = checkpoint(lambda _x, _mask, _mt, _blk=blk: _blk(_x, _mask, _mt),
                                    x, attn_mask, m_token, use_reentrant=False)

                else:
                    x = blk(x, attn_mask, m_token)
                    
            x = x.view(b, d, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, "b d h w c -> b c d h w")

        elif len(x_shape) == 4:
            b, c, h, w = x_shape
            window_size, shift_size = get_window_size(
                (h, w), self.window_size, self.shift_size
            )
            x = rearrange(x, "b c h w -> b h w c")
            hp = int(np.ceil(h / window_size[0])) * window_size[0]
            wp = int(np.ceil(w / window_size[1])) * window_size[1]
            attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:  # todo
                x = blk(x, attn_mask)
            x = x.view(b, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, "b h w c -> b c h w")
        return x


class SwinTransformer(nn.Module):
    def __init__(
        self,
        in_chans,
        embed_dim,
        window_size,
        patch_size,
        depths,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        patch_norm=False,
        use_checkpoint=False,
        norm_name="instance",
        spatial_dims=3,
        downsample="merging",
    ):
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            patch_norm: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: spatial dimension.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"`
        """

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        # self.layers4 = nn.ModuleList()
        down_sample_mod = (
            MERGING_MODE[downsample] if isinstance(downsample, str) else downsample
        )
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=down_sample_mod,
                use_checkpoint=use_checkpoint,
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_chans,
            out_channels=8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        mf_token1 = torch.zeros(
            [1, 64, 64, 64, 1], dtype=torch.float, requires_grad=True
        )
        self.mf_token1 = torch.nn.Parameter(mf_token1)

        mf_token2 = torch.zeros(
            [1, 32, 32, 32, 1], dtype=torch.float, requires_grad=True
        )
        self.mf_token2 = torch.nn.Parameter(mf_token2)

        mf_token3 = torch.zeros(
            [1, 16, 16, 16, 1], dtype=torch.float, requires_grad=True
        )
        self.mf_token3 = torch.nn.Parameter(mf_token3)

        mt1c_token1 = torch.zeros(
            [1, 64, 64, 64, 1], dtype=torch.float, requires_grad=True
        )
        self.mt1c_token1 = torch.nn.Parameter(mt1c_token1)

        mt1c_token2 = torch.zeros(
            [1, 32, 32, 32, 1], dtype=torch.float, requires_grad=True
        )
        self.mt1c_token2 = torch.nn.Parameter(mt1c_token2)

        mt1c_token3 = torch.zeros(
            [1, 16, 16, 16, 1], dtype=torch.float, requires_grad=True
        )
        self.mt1c_token3 = torch.nn.Parameter(mt1c_token3)

        mt1_token1 = torch.zeros(
            [1, 64, 64, 64, 1], dtype=torch.float, requires_grad=True
        )
        self.mt1_token1 = torch.nn.Parameter(mt1_token1)

        mt1_token2 = torch.zeros(
            [1, 32, 32, 32, 1], dtype=torch.float, requires_grad=True
        )
        self.mt1_token2 = torch.nn.Parameter(mt1_token2)

        mt1_token3 = torch.zeros(
            [1, 16, 16, 16, 1], dtype=torch.float, requires_grad=True
        )
        self.mt1_token3 = torch.nn.Parameter(mt1_token3)

        mt2_token1 = torch.zeros(
            [1, 64, 64, 64, 1], dtype=torch.float, requires_grad=True
        )
        self.mt2_token1 = torch.nn.Parameter(mt2_token1)

        mt2_token2 = torch.zeros(
            [1, 32, 32, 32, 1], dtype=torch.float, requires_grad=True
        )
        self.mt2_token2 = torch.nn.Parameter(mt2_token2)

        mt2_token3 = torch.zeros(
            [1, 16, 16, 16, 1], dtype=torch.float, requires_grad=True
        )
        self.mt2_token3 = torch.nn.Parameter(mt2_token3)

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def forward(self, x, normalize=True, m_label="all"):
        enc0 = self.encoder1(x)
        x0 = self.patch_embed(x)
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)
        if m_label == "flair":
            x1 = self.layers1[0](x0.contiguous(), self.mf_token1)
        elif m_label == "t1ce":
            x1 = self.layers1[0](x0.contiguous(), self.mt1c_token1)
        elif m_label == "t1":
            x1 = self.layers1[0](x0.contiguous(), self.mt1_token1)
        elif m_label == "t2":
            x1 = self.layers1[0](x0.contiguous(), self.mt2_token1)
        else:
            x1 = self.layers1[0](x0.contiguous())

        x1_out = self.proj_out(x1, normalize)

        if m_label == "flair":
            x2 = self.layers2[0](x1.contiguous(), self.mf_token2)
        elif m_label == "t1ce":
            x2 = self.layers2[0](x1.contiguous(), self.mt1c_token2)
        elif m_label == "t1":
            x2 = self.layers2[0](x1.contiguous(), self.mt1_token2)
        elif m_label == "t2":
            x2 = self.layers2[0](x1.contiguous(), self.mt2_token2)
        else:
            x2 = self.layers2[0](x1.contiguous())

        x2_out = self.proj_out(x2, normalize)

        if m_label == "flair":
            x3 = self.layers3[0](x2.contiguous(), self.mf_token3)
        elif m_label == "t1ce":
            x3 = self.layers3[0](x2.contiguous(), self.mt1c_token3)
        elif m_label == "t1":
            x3 = self.layers3[0](x2.contiguous(), self.mt1_token3)
        elif m_label == "t2":
            x3 = self.layers3[0](x2.contiguous(), self.mt2_token3)
        else:
            x3 = self.layers3[0](x2.contiguous())

        x3_out = self.proj_out(x3, normalize)

        return [enc0, x0_out, x1_out, x2_out, x3_out]


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class shiftmlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        shift_size=5,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [
            torch.roll(x_c, shift, 2)
            for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))
        ]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)

        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)

        x = self.fc1(x_shift_r)

        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [
            torch.roll(x_c, shift, 3)
            for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))
        ]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x


class shiftedBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        # self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_fuse, self).__init__()

        self.d4_c1 = general_conv3d_prenorm(
            basic_dims * 24, basic_dims * 12, pad_type="reflect"
        )
        self.d4_c2 = general_conv3d_prenorm(
            basic_dims * 24, basic_dims * 12, pad_type="reflect"
        )
        self.d4_out = general_conv3d_prenorm(
            basic_dims * 12, basic_dims * 12, k_size=1, padding=0, pad_type="reflect"
        )

        self.d3_c1 = general_conv3d_prenorm(
            basic_dims * 12, basic_dims * 6, pad_type="reflect"
        )
        self.d3_c2 = general_conv3d_prenorm(
            basic_dims * 12, basic_dims * 6, pad_type="reflect"
        )
        self.d3_out = general_conv3d_prenorm(
            basic_dims * 6, basic_dims * 6, k_size=1, padding=0, pad_type="reflect"
        )

        self.d2_c1 = general_conv3d_prenorm(
            basic_dims * 6, basic_dims * 3, pad_type="reflect"
        )
        self.d2_c2 = general_conv3d_prenorm(
            basic_dims * 6, basic_dims * 3, pad_type="reflect"
        )
        self.d2_out = general_conv3d_prenorm(
            basic_dims * 3, basic_dims * 3, k_size=1, padding=0, pad_type="reflect"
        )

        self.d1_c1 = general_conv3d_prenorm(
            basic_dims * 3, basic_dims, pad_type="reflect"
        )
        self.d1_c2 = general_conv3d_prenorm(
            basic_dims * 2, basic_dims, pad_type="reflect"
        )
        self.d1_out = general_conv3d_prenorm(
            basic_dims, basic_dims, k_size=1, padding=0, pad_type="reflect"
        )

        self.seg_d4 = nn.Conv3d(
            in_channels=basic_dims * 24,
            out_channels=num_cls,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.seg_d3 = nn.Conv3d(
            in_channels=basic_dims * 12,
            out_channels=num_cls,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.seg_d2 = nn.Conv3d(
            in_channels=basic_dims * 6,
            out_channels=num_cls,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.seg_d1 = nn.Conv3d(
            in_channels=basic_dims * 3,
            out_channels=num_cls,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.seg_layer = nn.Conv3d(
            in_channels=basic_dims,
            out_channels=num_cls,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode="trilinear", align_corners=True)

        self.RFM5 = fusion_prenorm(in_channel=basic_dims * 24, num_cls=num_cls)
        self.RFM4 = fusion_prenorm(in_channel=basic_dims * 12, num_cls=num_cls)
        self.RFM3 = fusion_prenorm(in_channel=basic_dims * 6, num_cls=num_cls)
        self.RFM2 = fusion_prenorm(in_channel=basic_dims * 3, num_cls=num_cls)
        self.RFM1 = fusion_prenorm(in_channel=basic_dims * 1, num_cls=num_cls)

    def forward(self, x1, x2, x3, x4, x5):

        de_x5 = self.RFM5(x5)
        pred4 = self.softmax(self.seg_d4(de_x5))
        de_x5 = self.d4_c1(self.up2(de_x5))

        de_x4 = self.RFM4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        pred3 = self.softmax(self.seg_d3(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))

        de_x3 = self.RFM3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        pred2 = self.softmax(self.seg_d2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        de_x2 = self.RFM2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        pred1 = self.softmax(self.seg_d1(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        de_x1 = self.RFM1(x1)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred, (
            self.up2(pred1),
            self.up4(pred2),
            self.up8(pred3),
            self.up16(pred4),
        )


class MaskModal(nn.Module):
    def __init__(self):
        super(MaskModal, self).__init__()

    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        x = y.view(B, -1, H, W, Z)
        return x


class Model(nn.Module):
    def __init__(
        self,
        num_cls=4,
        img_size=128,
        in_channels=1,
        out_channels=3,
        depths=(2, 2, 2),  # , 2),
        model_num_heads=(3, 6, 12, 24),
        feature_size=24,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3,
    ):
        super(Model, self).__init__()
        img_size = (img_size,) * spatial_dims
        patch_size = (2,) * spatial_dims
        window_size = (7,) * spatial_dims

        import numpy as np

        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError(
                        "input image size (img_size) should be divisible by stage-wise image resolution."
                    )

        self.normalize = normalize

        self.swinEncoder = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=model_num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )

        embed_dims = [1, 4, 16]
        num_heads1 = [1, 2, 4, 8]
        qkv_bias = False
        qk_scale = None
        drop_rate = 0.0
        attn_drop_rate = 0.0
        drop_path_rate = 0.0
        norm_layer = nn.LayerNorm
        depths = [1, 1, 1]
        sr_ratios = [8, 4, 2, 1]
        mlp_img_size = 224

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.patch_embed3 = OverlapPatchEmbed(
            img_size=mlp_img_size // 8,
            patch_size=7,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=mlp_img_size // 8,
            patch_size=7,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
        )
        self.block2 = nn.ModuleList(
            [
                shiftedBlock(
                    dim=embed_dims[2],
                    num_heads=num_heads1[1],
                    mlp_ratio=1,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[1],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
            ]
        )
        self.block3 = nn.ModuleList(
            [
                shiftedBlock(
                    dim=embed_dims[1],
                    num_heads=num_heads1[1],
                    mlp_ratio=1,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[1],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
            ]
        )

        self.norm3 = nn.LayerNorm(embed_dims[1])
        self.norm4 = nn.LayerNorm(embed_dims[2])

        self.masker = MaskModal()

        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x, mask):
        B = x.shape[0]

        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.swinEncoder(
            x[:, 0:1, :, :, :], self.normalize, m_label="flair"
        ) # x1: (B, 8, 128, 128, 128), x2: (B, 24, 64, 64, 64), x3: (B, 48, 32, 32, 32), x4: (B, 96, 16, 16, 16), x5: (B, 192, 8, 8, 8)
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.swinEncoder(
            x[:, 1:2, :, :, :], self.normalize, m_label="t1ce"
        )
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.swinEncoder(
            x[:, 2:3, :, :, :], self.normalize, m_label="t1"
        )
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.swinEncoder(
            x[:, 3:4, :, :, :], self.normalize, m_label="t2"
        )

        flair_token_x5 = (
            flair_x5.permute(0, 2, 3, 4, 1)
            .contiguous()
            .view(x.size(0), -1, transformer_basic_dims)
        )   # (B, 512, 192)
        t1ce_token_x5 = (
            t1ce_x5.permute(0, 2, 3, 4, 1)
            .contiguous()
            .view(x.size(0), -1, transformer_basic_dims)
        )
        t1_token_x5 = (
            t1_x5.permute(0, 2, 3, 4, 1)
            .contiguous()
            .view(x.size(0), -1, transformer_basic_dims)
        )
        t2_token_x5 = (
            t2_x5.permute(0, 2, 3, 4, 1)
            .contiguous()
            .view(x.size(0), -1, transformer_basic_dims)
        )

        flair_token_x5 = flair_token_x5.unsqueeze(1).contiguous()   # (B, 1, 1, 512, 192)
        t1ce_token_x5 = t1ce_token_x5.unsqueeze(1).contiguous()
        t1_token_x5 = t1_token_x5.unsqueeze(1).contiguous()
        t2_token_x5 = t2_token_x5.unsqueeze(1).contiguous()

        out, H, W = self.patch_embed3(flair_token_x5)   # out: (B, 24576, 4), H=256, W=4
        for i, blk in enumerate(self.block3):   # 1 block of shiftedBlock
            out = blk(out, H, W)                        # out: (B, 24576, 4)
        flair_intra_token_x5 = self.norm3(out)          # flair_intra_token_x5: (B, 24576, 4)
        flair_intra_token_x5 = flair_intra_token_x5.reshape(
            B, 2 * H, 2 * W
        ).contiguous()                                  # (B, 512, 192)

        out, H, W = self.patch_embed3(t1ce_token_x5)
        for i, blk in enumerate(self.block3):
            out = blk(out, H, W)
        t1ce_intra_token_x5 = self.norm3(out)
        t1ce_intra_token_x5 = t1ce_intra_token_x5.reshape(B, 2 * H, 2 * W).contiguous()

        out, H, W = self.patch_embed3(t1_token_x5)
        for i, blk in enumerate(self.block3):
            out = blk(out, H, W)
        t1_intra_token_x5 = self.norm3(out)
        t1_intra_token_x5 = t1_intra_token_x5.reshape(B, 2 * H, 2 * W).contiguous()

        out, H, W = self.patch_embed3(t2_token_x5)
        for i, blk in enumerate(self.block3):
            out = blk(out, H, W)
        t2_intra_token_x5 = self.norm3(out)
        t2_intra_token_x5 = t2_intra_token_x5.reshape(B, 2 * H, 2 * W).contiguous()

        flair_intra_x5 = (
            flair_intra_token_x5.view(
                x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims
            )
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )                                               # (B, 192, 8, 8, 8)
        t1ce_intra_x5 = (
            t1ce_intra_token_x5.view(
                x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims
            )
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )
        t1_intra_x5 = (
            t1_intra_token_x5.view(
                x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims
            )
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )
        t2_intra_x5 = (
            t2_intra_token_x5.view(
                x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims
            )
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )

        x1 = self.masker(
            torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1), mask
        )  # Bx4xCxHxWxZ
        x2 = self.masker(torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1), mask)
        x3 = self.masker(torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1), mask)
        x4 = self.masker(torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1), mask)
        x5_intra = self.masker(
            torch.stack(
                (flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5), dim=1
            ),
            mask,
        )

        flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5 = torch.chunk(
            x5_intra, num_modals, dim=1
        )
        average_x5 = flair_intra_x5 + t1ce_intra_x5 + t1_intra_x5 + t2_intra_x5
        average_x5 = torch.div(average_x5, 4)

        multimodal_token_x5 = torch.cat(
            (
                flair_intra_x5.permute(0, 2, 3, 4, 1)
                .contiguous()
                .view(x.size(0), 1, -1, transformer_basic_dims),        #(B, 1, 512, 192)
                t1ce_intra_x5.permute(0, 2, 3, 4, 1)
                .contiguous()
                .view(x.size(0), 1, -1, transformer_basic_dims),
                t1_intra_x5.permute(0, 2, 3, 4, 1)
                .contiguous()
                .view(x.size(0), 1, -1, transformer_basic_dims),
                t2_intra_x5.permute(0, 2, 3, 4, 1)
                .contiguous()
                .view(x.size(0), 1, -1, transformer_basic_dims),
            ),
            dim=1,
        )

        out, H, W = self.patch_embed4(multimodal_token_x5)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, 2 * H, 2 * W, -1).permute(0, 3, 1, 2).contiguous()
        x5_inter = (
            out.view(
                B,
                patch_size,
                patch_size,
                patch_size,
                transformer_basic_dims * num_modals,
            )
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )

        if self.training:
            fuse_pred, preds = checkpoint(
                lambda a,b,c,d,e, _dec=self.decoder_fuse: _dec(a,b,c,d,e),
                x1, x2, x3, x4, x5_inter,
                use_reentrant=False
            )
        else:
            fuse_pred, preds = self.decoder_fuse(x1, x2, x3, x4, x5_inter)


        if self.is_training:
            flair_intra_x5 = (
                flair_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(B, -1)
            )
            t1ce_intra_x5 = (
                t1ce_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(B, -1)
            )
            t1_intra_x5 = t1_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(B, -1)
            t2_intra_x5 = t2_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(B, -1)
            average_x5 = average_x5.permute(0, 2, 3, 4, 1).contiguous().view(B, -1)
            return (
                fuse_pred,
                (flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5, average_x5),
                preds,
            )
        return fuse_pred
