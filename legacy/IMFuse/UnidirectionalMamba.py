import numpy as np
import torch
from torch import nn
from mamba_ssm import Mamba
from torch.cuda.amp import autocast


class MambaTrans(nn.Module):
    def __init__(self, channels):
        super(MambaTrans, self).__init__()
        self.mamba = Mamba(
            d_model=channels,
            d_state=min(channels, 256),
            d_conv=4,
            expand=2,
        )
        self.norm1 = nn.LayerNorm(channels,)
        self.norm2 = nn.LayerNorm(channels,)
        self.head = nn.Linear(channels, channels)

    def forward(self, x):
        x = self.mamba(self.norm1(x)) + x
        x = self.head(self.norm2(x)) + x
        return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = MambaTrans(dim)
    
    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        return out