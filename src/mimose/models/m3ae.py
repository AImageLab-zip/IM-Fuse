from __future__ import annotations

import torch
import torch.nn as nn

from mimose.models.abstract_model import AbstractModel

num_modals = 4
init_channels = 16
input_patch_size = 128


class BasicBlock(nn.Module):
    """Pre-activation GroupNorm residual block, matching legacy M3AE's ``BasicBlock``."""

    def __init__(self, in_channels: int, out_channels: int, n_groups: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(self.act1(self.norm1(x)))
        out = self.conv2(self.act2(self.norm2(out)))
        return out + residual


class UNet3D(nn.Module):
    """Standard 3D U-Net trunk with additive (not concatenated) skip
    connections, matching legacy M3AE's ``UNet3D_g``.

    Returns the three decoder feature volumes ``(u2, u3, u4)`` (with
    ``channels``, ``channels*2`` and ``channels*4`` output channels
    respectively) rather than a specific task head, since ``M3AE`` attaches
    multiple heads (segmentation, deep-supervision, reconstruction) to the same
    trunk for its two-stage training. ``u2`` is the full-resolution trunk
    output; ``u3``/``u4`` feed the deep-supervision heads.
    """

    def __init__(self, in_channels: int, channels: int = init_channels) -> None:
        super().__init__()
        self.conv1a = nn.Conv3d(in_channels, channels, kernel_size=3, padding=1)
        self.conv1b = BasicBlock(channels, channels)
        self.down1 = nn.Conv3d(channels, channels * 2, kernel_size=3, stride=2, padding=1)

        self.conv2a = BasicBlock(channels * 2, channels * 2)
        self.conv2b = BasicBlock(channels * 2, channels * 2)
        self.down2 = nn.Conv3d(channels * 2, channels * 4, kernel_size=3, stride=2, padding=1)

        self.conv3a = BasicBlock(channels * 4, channels * 4)
        self.conv3b = BasicBlock(channels * 4, channels * 4)
        self.down3 = nn.Conv3d(channels * 4, channels * 8, kernel_size=3, stride=2, padding=1)

        self.conv4a = BasicBlock(channels * 8, channels * 8)
        self.conv4b = BasicBlock(channels * 8, channels * 8)
        self.conv4c = BasicBlock(channels * 8, channels * 8)
        self.conv4d = BasicBlock(channels * 8, channels * 8)

        self.up4conv = nn.Conv3d(channels * 8, channels * 4, kernel_size=1)
        self.up4 = nn.Upsample(scale_factor=2)
        self.up4block = BasicBlock(channels * 4, channels * 4)

        self.up3conv = nn.Conv3d(channels * 4, channels * 2, kernel_size=1)
        self.up3 = nn.Upsample(scale_factor=2)
        self.up3block = BasicBlock(channels * 2, channels * 2)

        self.up2conv = nn.Conv3d(channels * 2, channels, kernel_size=1)
        self.up2 = nn.Upsample(scale_factor=2)
        self.up2block = BasicBlock(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.conv1b(self.conv1a(x))
        c1d = self.down1(c1)

        c2 = self.conv2b(self.conv2a(c1d))
        c2d = self.down2(c2)

        c3 = self.conv3b(self.conv3a(c2d))
        c3d = self.down3(c3)

        c4 = self.conv4b(self.conv4a(c3d))
        c4 = self.conv4d(self.conv4c(c4))

        u4 = self.up4block(self.up4(self.up4conv(c4)) + c3)
        u3 = self.up3block(self.up3(self.up3conv(u4)) + c2)
        u2 = self.up2block(self.up2(self.up2conv(u3)) + c1)

        return u2, u3, u4


class M3AE(AbstractModel):
    """Implementation of M3AE introduced in [1].

    [1] Liu, et al. "M3AE: Multimodal Representation Learning for Brain
        Tumor Segmentation with Missing Modalities". AAAI 2023.

    Missing modalities are substituted with a single learned per-voxel
    embedding (rather than zero-filled, unlike every other model here), then
    all 4 channels are jointly encoded by one shared U-Net (no per-modality
    encoders).

    Legacy M3AE is a two-stage pipeline: a masked-autoencoder-style
    pretraining stage (reconstructing the original input from a
    missing-modality view, filled in via the "latent image" placeholder)
    followed by supervised fine-tuning that adds deep supervision and a
    cross-view consistency loss (MSE between the coarse ``out4`` head applied
    to two independently-resampled missing-modality views of the same crop).
    Both are reproduced here: ``mode="segment_train"`` exposes the
    deep-supervision heads and the ``out4`` logits, and ``M3AETrainer`` runs a
    second masked view per step to form the consistency term (see
    ``custom_trainer_kwargs.deep_supervised`` / ``consistency_weight``). The
    MAE-style pretraining stage is also reproduced (see ``pretrain_fraction``).

    Modes: ``mode="segment"`` (default, used by ``predict``/inference) returns
    a single softmax volume. ``mode="segment_train"`` returns
    ``([uout, out3, out4], out4_logits)`` -- the three softmax
    (deep-supervision) heads plus the pre-softmax ``out4`` logits used for the
    cross-view consistency MSE. ``mode="reconstruct"`` is the pretraining
    path. All modes share the masked-input construction and U-Net trunk,
    differing only in which head is applied.
    """

    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.num_cls = num_cls
        self.limage = nn.Parameter(
            torch.randn(1, num_modals, input_patch_size, input_patch_size, input_patch_size) * 0.02
        )
        self.unet = UNet3D(in_channels=num_modals)
        self.seg_layer = nn.Conv3d(init_channels, num_cls, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.recon_layer = nn.Conv3d(init_channels, num_modals, kernel_size=1)

        # Deep-supervision heads on the two coarser decoder levels, matching
        # legacy ``Unet_missing`` (ds_out convs at init_channels*4 / *2 then
        # trilinear upsampling back to full resolution). ``out4`` (the ×4
        # head) also supplies the pre-softmax logits compared across views by
        # the cross-view consistency loss (legacy feature_level=2).
        self.ds_out4 = nn.Conv3d(init_channels * 4, num_cls, kernel_size=1)
        self.ds_up4 = nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)
        self.ds_out3 = nn.Conv3d(init_channels * 2, num_cls, kernel_size=1)
        self.ds_up3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)

    def _forward_trunk(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.size(1) != num_modals:
            raise RuntimeError(f"M3AE expects {num_modals} input modalities, got {x.size(1)}")
        if mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(f"M3AE expects mask shape [B, {num_modals}], got {tuple(mask.shape)}")
        if x.shape[-3:] != self.limage.shape[-3:]:
            raise RuntimeError(
                f"M3AE requires spatial dimensions of exactly {tuple(self.limage.shape[-3:])}, got {tuple(x.shape[-3:])}"
            )

        gate = mask.view(-1, num_modals, 1, 1, 1).to(x.dtype)
        x = x * gate + self.limage * (1 - gate)
        return self.unet(x)  # (u2, u3, u4)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, mode: str = "segment"
    ) -> torch.Tensor | tuple[list[torch.Tensor], torch.Tensor]:
        """``mode="segment"`` (default) is the fine-tuning/inference path,
        used by ``predict``. ``mode="reconstruct"`` is the pretraining-stage
        path. Both go through this single ``forward`` (rather than a
        separate public method doing the real work) so that trainer code can
        always call ``self.model(x, mask, mode=...)`` and pick up
        ``DistributedDataParallel``'s gradient-sync hooks, which only fire on
        the wrapped model's ``forward`` — calling a distinct method on the
        DDP-unwrapped module directly would silently skip gradient
        all-reduce under multi-GPU training.
        """
        u2, u3, u4 = self._forward_trunk(x, mask)
        if mode == "reconstruct":
            return self.recon_layer(u2)
        if mode == "segment_train":
            out4_logits = self.ds_up4(self.ds_out4(u4))
            out3_logits = self.ds_up3(self.ds_out3(u3))
            uout = self.softmax(self.seg_layer(u2))
            seg_outputs = [uout, self.softmax(out3_logits), self.softmax(out4_logits)]
            return seg_outputs, out4_logits
        return self.softmax(self.seg_layer(u2))

    def reconstruct(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Non-distributed convenience wrapper around
        ``forward(x, mask, mode="reconstruct")``, reconstructing the
        original, unmasked ``x`` from the masked input (matching legacy
        M3AE's reconstruction objective). Trainer code should call
        ``self.model(x, mask, mode="reconstruct")`` directly instead, to stay
        DDP-safe (see ``forward``'s docstring)."""
        return self(x, mask, mode="reconstruct")

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, _, height, width, depth = images.shape
        if (height, width, depth) == (input_patch_size, input_patch_size, input_patch_size):
            return self(images, mask)

        h_starts = self._window_starts(height)
        w_starts = self._window_starts(width)
        d_starts = self._window_starts(depth)
        prediction = torch.zeros(images.size(0), self.num_cls, height, width, depth, device=images.device)
        weight = torch.zeros(images.size(0), 1, height, width, depth, device=images.device)

        for h in h_starts:
            for w in w_starts:
                for d in d_starts:
                    patch = images[
                        :,
                        :,
                        h : h + input_patch_size,
                        w : w + input_patch_size,
                        d : d + input_patch_size,
                    ]
                    patch_pred = self(patch, mask)
                    prediction[
                        :,
                        :,
                        h : h + input_patch_size,
                        w : w + input_patch_size,
                        d : d + input_patch_size,
                    ] += patch_pred
                    weight[
                        :,
                        :,
                        h : h + input_patch_size,
                        w : w + input_patch_size,
                        d : d + input_patch_size,
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

        stride = int(input_patch_size * 0.5)
        starts = list(range(0, max(size - input_patch_size, 0), stride))
        last_start = size - input_patch_size
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts


Model = M3AE


__all__ = ["M3AE", "Model"]
