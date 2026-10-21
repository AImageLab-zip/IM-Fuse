from __future__ import annotations

from itertools import product
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimose.models.abstract_model import AbstractModel

num_modals = 4
input_patch_size = 128
init_channels = 16
adapter_bottleneck = 64
num_graph_nodes = 8

# External pipeline uses [t1c, t1n, t2f, t2w]; AdaMM internals expect
# [flair, t1, t1ce, t2] (same convention as legacy MST-KDNet, which AdaMM
# is built on).
DATASET_MODALITY_ORDER = (2, 1, 0, 3)
FULL_MODALITY_KEY = "1" * num_modals


def mask_key(mask: torch.Tensor) -> str:
    """Legacy adapter-bank key: one bit per modality in legacy order, e.g. ``"1010"``."""
    return "".join("1" if bool(present) else "0" for present in mask.tolist())


class BasicBlock(nn.Module):
    """Myronenko-style pre-activation residual block (GroupNorm->ReLU->Conv3d x2)."""

    def __init__(self, in_channels: int, out_channels: int, n_groups: int = 8) -> None:
        super().__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(n_groups, in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)
        x = self.relu2(self.gn2(x))
        x = self.conv2(x)
        return x + residual


class Adapter(nn.Module):
    """Residual 1x1x1 bottleneck adapter from legacy's per-mask "Adapter Bank"."""

    def __init__(self, channels: int, bottleneck: int = adapter_bottleneck, p: float = 0.1) -> None:
        super().__init__()
        self.down = nn.Conv3d(channels, bottleneck, kernel_size=1)
        self.act = nn.GELU()
        self.up = nn.Conv3d(bottleneck, channels, kernel_size=1)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.up(self.act(self.down(x))))


class BasicConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.stdv = 1.0 / math.sqrt(in_channels)

    def reset_params(self) -> None:
        self.conv.weight.data.uniform_(-self.stdv, self.stdv)
        self.bn.weight.data.uniform_()
        self.bn.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class GraphNet3D(nn.Module):
    """Soft-assigns every voxel feature to ``node_num`` learnable anchors (NetVLAD-style)."""

    def __init__(self, node_num: int, dim: int) -> None:
        super().__init__()
        self.node_num = node_num
        self.dim = dim
        self.anchor = nn.Parameter(torch.rand(node_num, dim))
        self.sigma = nn.Parameter(torch.rand(node_num, dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels = x.shape[:2]
        sigma = torch.sigmoid(self.sigma)
        flattened = x.reshape(batch, channels, -1).transpose(1, 2)  # [B, V, C]
        # [B, V, N, C]; vectorised form of legacy's per-node loop.
        residual = (flattened.unsqueeze(2) - self.anchor).div(sigma)
        soft_assign = F.softmax(-residual.pow(2).sum(dim=-1) / 2, dim=2).transpose(1, 2)  # [B, N, V]

        weight = soft_assign.unsqueeze(-1)  # [B, N, V, 1]
        nodes = (residual.transpose(1, 2) * weight).sum(dim=2) / (soft_assign.sum(dim=2, keepdim=True) + 1e-9)
        nodes = F.normalize(nodes, p=2, dim=2)
        nodes = F.normalize(nodes.reshape(batch, -1), p=2, dim=1)
        return nodes.reshape(batch, channels, self.node_num), soft_assign


class GraphConvNet(nn.Module):
    """Simple GCN layer (Kipf & Welling) with a softmax-normalised adjacency."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        stdv = 1.0 / math.sqrt(out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.matmul(x.transpose(1, 2), self.weight)  # [B, N, C]
        adj = torch.softmax(adj, dim=2)
        return torch.matmul(adj, support).transpose(1, 2)  # [B, C, N]


class GATLayer(nn.Module):
    """Legacy's single-head dot-product attention over graph nodes.

    Legacy accepts an adjacency matrix but never reads it (the attention is
    dense over all nodes), and divides the scores by sqrt(#nodes), not
    sqrt(dim) -- both kept as-is.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qk = self.attn(x.transpose(1, 2))  # [B, N, C]
        scores = torch.bmm(qk, qk.transpose(1, 2)) / math.sqrt(qk.size(1))
        out = torch.bmm(F.softmax(scores, dim=-1), qk)
        return self.output_proj(out).transpose(1, 2)


class AdaptiveGraphEnhanceModule(nn.Module):
    """Legacy ``AdaptiveGraphEnhanceModule`` (the paper's GARM).

    Fuses the adapted (missing-modality) bottleneck feature (``contour_x``)
    with the shared one (``region_x``) through two node graphs, a GCN, and a
    unified cross-graph attention pass, then re-projects onto the voxels.
    """

    def __init__(self, dim: int, out_dim: int, num_clusters: int = num_graph_nodes) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.contour_proj0 = GraphNet3D(node_num=num_clusters, dim=dim)
        self.region_proj0 = GraphNet3D(node_num=num_clusters, dim=dim)
        self.contour_conv = BasicConv2d(dim, dim, kernel_size=1, padding=0)
        self.contour_conv.reset_params()
        self.gcn = GraphConvNet(dim, dim)
        self.gat = GATLayer(dim)
        self.conv0 = BasicConv3d(dim, dim, kernel_size=1, padding=0)
        self.conv1 = BasicConv3d(dim, dim, kernel_size=1, padding=0)
        self.pred = nn.Conv3d(dim, out_dim, kernel_size=1)

    def forward(self, contour_x: torch.Tensor, region_x: torch.Tensor) -> torch.Tensor:
        region_graph, region_assign = self.region_proj0(region_x)
        contour_graph, contour_assign = self.contour_proj0(contour_x)
        contour_graph = self.contour_conv(contour_graph.unsqueeze(3)).squeeze(3)

        adjacency = region_graph.transpose(1, 2).bmm(contour_graph)
        region_graph = self.gcn(region_graph, adjacency)
        contour_graph = self.gcn(contour_graph, adjacency)

        contour_graph = self.gat(contour_graph)
        region_graph = self.gat(region_graph)
        unified_graph = self.gat(torch.cat([contour_graph, region_graph], dim=-1))
        region_part = unified_graph[:, :, self.num_clusters :]

        n = self.num_clusters
        weights = torch.softmax(torch.cat((contour_assign, region_assign), dim=1), dim=1)
        fused_assign = weights[:, :n] * contour_assign + weights[:, n:] * region_assign

        n_region_x = self.conv0(region_part.bmm(fused_assign).view(region_x.size()))
        region_x = self.conv1(region_x + n_region_x + contour_x)
        return self.pred(region_x)


class _AdaMMBackbone(nn.Module):
    """Myronenko 3D U-Net with legacy AdaMM's adapter bank, GARM, and lesion-presence head.

    Returns the legacy 6-tuple ``(uout, style, content, global_gs, logit,
    cls_probs)``. ``with_missing_modules=False`` builds the teacher, which
    only ever runs on the full-modality key and so never touches the adapter
    bank or GARM.
    """

    def __init__(
        self,
        in_channels: int = num_modals,
        out_channels: int = 4,
        base_channels: int = init_channels,
        num_presence_classes: int = 3,
        p: float = 0.2,
        with_missing_modules: bool = True,
    ) -> None:
        super().__init__()
        c = base_channels
        self.conv1a = nn.Conv3d(in_channels, c, kernel_size=3, padding=1)
        self.conv1b = BasicBlock(c, c)
        self.ds1 = nn.Conv3d(c, c * 2, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicBlock(c * 2, c * 2)
        self.conv2b = BasicBlock(c * 2, c * 2)
        self.ds2 = nn.Conv3d(c * 2, c * 4, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicBlock(c * 4, c * 4)
        self.conv3b = BasicBlock(c * 4, c * 4)
        self.ds3 = nn.Conv3d(c * 4, c * 8, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicBlock(c * 8, c * 8)
        self.conv4b = BasicBlock(c * 8, c * 8)
        self.conv4c = BasicBlock(c * 8, c * 8)
        self.conv4d = BasicBlock(c * 8, c * 8)

        self.up4conva = nn.Conv3d(c * 8, c * 4, kernel_size=1)
        self.up4 = nn.Upsample(scale_factor=2)
        self.up4convb = BasicBlock(c * 4, c * 4)
        self.up3conva = nn.Conv3d(c * 4, c * 2, kernel_size=1)
        self.up3 = nn.Upsample(scale_factor=2)
        self.up3convb = BasicBlock(c * 2, c * 2)
        self.up2conva = nn.Conv3d(c * 2, c, kernel_size=1)
        self.up2 = nn.Upsample(scale_factor=2)
        self.up2convb = BasicBlock(c, c)
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.convc = nn.Conv3d(c * 20, c * 8, kernel_size=1)
        self.convco = nn.Conv3d(c * 16, c * 8, kernel_size=1)
        self.up1conv = nn.Conv3d(c, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(p=p)

        # Lesion-presence (LPGM) classification head on the last decoder feature.
        self.cls_conv1 = nn.Conv3d(c, c, kernel_size=3, padding=1)
        self.cls_bn1 = nn.BatchNorm3d(c)
        self.cls_conv2 = nn.Conv3d(c, num_presence_classes, kernel_size=1)
        self.cls_pool = nn.AdaptiveAvgPool3d(1)

        self.with_missing_modules = with_missing_modules
        if with_missing_modules:
            self.enc_adapters = nn.ModuleDict()
            self.dec_adapters = nn.ModuleDict()
            self.convc_ = nn.ModuleDict()
            self.convco_ = nn.ModuleDict()
            for combo in product((0, 1), repeat=in_channels):
                # The full key never uses adapters; the empty key never occurs.
                if all(combo) or not any(combo):
                    continue
                key = "".join(str(bit) for bit in combo)
                self.enc_adapters[key] = nn.ModuleList(
                    [Adapter(c * 2, p=p), Adapter(c * 4, p=p), Adapter(c * 8, p=p), Adapter(c * 8, p=p)]
                )
                self.dec_adapters[key] = nn.ModuleList([Adapter(c * 4, p=p), Adapter(c * 2, p=p), Adapter(c, p=p)])
                self.convc_[key] = nn.Conv3d(c * 20, c * 8, kernel_size=1)
                self.convco_[key] = nn.Conv3d(c * 16, c * 8, kernel_size=1)
            self.adagem = AdaptiveGraphEnhanceModule(dim=c * 8, out_dim=c * 8)

    def forward(
        self, x: torch.Tensor, key: str = FULL_MODALITY_KEY
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        adapted = key != FULL_MODALITY_KEY
        if adapted and not self.with_missing_modules:
            raise RuntimeError("the AdaMM teacher branch only supports the full-modality input")
        enc = self.enc_adapters[key] if adapted else None
        dec = self.dec_adapters[key] if adapted else None

        c1 = self.conv1b(self.conv1a(x))
        c1d = self.ds1(c1)
        c2 = self.conv2b(self.conv2a(enc[0](c1d) if adapted else c1d))
        c2d = self.ds2(c2)
        c2d_p = self.pool(c2d)
        if adapted:
            c2d_ = enc[1](c2d)
            c2d_p_ = self.pool(c2d_)
            c3 = self.conv3a(c2d_)
        else:
            c3 = self.conv3a(c2d)
        c3 = self.conv3b(c3)
        c3d = self.ds3(c3)
        c3d_ = enc[2](c3d) if adapted else c3d
        c4 = self.conv4c(self.conv4b(self.conv4a(c3d_)))
        c4d = self.conv4d(c4)

        style = self.convc(torch.cat([c2d_p, c3d, c4d], dim=1))
        global_gs = torch.cat([c3d, style, c4d], dim=1)
        content = c4d
        c4d = self.dropout(self.convco(torch.cat([style, content], dim=1)))
        if adapted:
            # Legacy adapts conv4c's output (pre-conv4d) for the missing-modality path.
            c4d_ = enc[3](c4)
            style_ = self.convc_[key](torch.cat([c2d_p_, c3d_, c4d_], dim=1))
            c4d_ = self.dropout(self.convco_[key](torch.cat([style_, c4d_], dim=1)))
            c4d = self.adagem(c4d_, c4d)

        u4 = self.up4convb(self.up4(self.up4conva(c4d)) + c3)
        if adapted:
            u4 = dec[0](u4)
        u3 = self.up3convb(self.up3(self.up3conva(u4)) + c2)
        if adapted:
            u3 = dec[1](u3)
        u2 = self.up2(self.up2conva(u3))
        if adapted:
            u2 = dec[2](u2)
        u2 = u2 + c1

        cls_feat = F.relu(self.cls_bn1(self.cls_conv1(u2)))
        cls_probs = torch.sigmoid(self.cls_pool(self.cls_conv2(cls_feat)).flatten(1))

        logit = self.up1conv(self.up2convb(u2))
        uout = torch.sigmoid(logit)
        return uout, style, content, global_gs, logit, cls_probs


class AdaMM(AbstractModel):
    """Implementation of AdaMM ("No Modality Left Behind") introduced in [1].

    Like MST-KDNet (whose legacy code it extends), AdaMM co-trains a "full"
    teacher branch on the true 4-modality volume and a "missing" student
    branch on the zero-filled volume, and only the student is used at
    inference. On top of the shared Myronenko 3D U-Net, the student adds:

    - an Adapter Bank: a dedicated set of encoder/decoder bottleneck adapters
      and bottleneck fusion convs for every non-full modality combination,
      selected by the sample's modality mask;
    - GARM (``AdaptiveGraphEnhanceModule``): graph-based fusion of the
      adapted and shared bottleneck features, missing-modality path only;
    - LPGM: a lesion-presence head predicting, per tumor class, whether it
      appears in the patch. It reweights the Dice terms during training and
      gates the student's tumor-class outputs at inference.

    Port deviations (see docs/training.md for the rationale):

    - Legacy targets are overlapping WT/TC/ET sigmoid regions; this port uses
      MiMoSe's one-hot class targets (like the MST-KDNet port), so the
      presence head predicts each tumor *class* (NCR, ED, ET), and gating
      zeroes absent tumor-class channels while leaving background untouched.
    - Legacy's style discriminator (2e-4 weight, Adam rebuilt every
      iteration) is dropped, as in the MST-KDNet port.
    - Dead legacy code is not ported: the teacher's never-used adapters and
      GARM, GARM's unused ``MutualModule1``/``region_conv*``, and the
      adjacency matrices its attention layer never reads. The adapter for the
      all-missing mask (which never occurs) is also omitted.
    - Legacy runs one mask per batch (batch size 1); this port routes each
      sample through the adapters of its own mask.

    [1] AdaMM: https://github.com/Quanato607/AdaMM
    """

    def __init__(self, num_cls: int = 4, lpgm_gating: bool = True) -> None:
        super().__init__()
        self.num_cls = num_cls
        self.lpgm_gating = lpgm_gating
        num_presence = num_cls - 1
        self.student = _AdaMMBackbone(out_channels=num_cls, num_presence_classes=num_presence)
        self.teacher = _AdaMMBackbone(
            out_channels=num_cls, num_presence_classes=num_presence, with_missing_modules=False
        )
        self.is_training = False

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | tuple[tuple, tuple]:
        x, mask = self._remap_input_order(x, mask)
        masked = x * mask.view(mask.size(0), num_modals, 1, 1, 1).to(x.dtype)

        student_out = self._student_forward(masked, mask)
        if not self.is_training:
            uout, cls_probs = student_out[0], student_out[5]
            if not self.lpgm_gating:
                return uout
            gate = (cls_probs > 0.5).to(uout.dtype).view(uout.size(0), -1, 1, 1, 1)
            return torch.cat([uout[:, :1], uout[:, 1:] * gate], dim=1)

        teacher_out = self.teacher(x)
        return student_out, teacher_out

    def _student_forward(self, masked: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        keys = [mask_key(sample_mask) for sample_mask in mask]
        if len(set(keys)) == 1:
            return self.student(masked, keys[0])

        # Each mask has its own adapters: run every mask group, then restore batch order.
        groups: dict[str, list[int]] = {}
        for index, key in enumerate(keys):
            groups.setdefault(key, []).append(index)
        order: list[int] = []
        group_outputs = []
        for key, indices in groups.items():
            order.extend(indices)
            group_outputs.append(self.student(masked[indices], key))
        inverse = torch.argsort(torch.tensor(order, device=masked.device))
        return tuple(torch.cat(parts, dim=0)[inverse] for parts in zip(*group_outputs))

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
                    window = (
                        slice(None),
                        slice(None),
                        slice(h, h + input_patch_size),
                        slice(w, w + input_patch_size),
                        slice(d, d + input_patch_size),
                    )
                    prediction[window] += self(images[window], mask)
                    weight[window] += 1
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
        images: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images.size(1) != num_modals:
            raise RuntimeError(f"AdaMM expects {num_modals} input modalities, got {images.size(1)}")
        if mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(f"AdaMM expects mask shape [B, {num_modals}], got {tuple(mask.shape)}")

        return images[:, DATASET_MODALITY_ORDER, ...], mask[:, DATASET_MODALITY_ORDER]


Model = AdaMM


__all__ = ["AdaMM", "Model"]
