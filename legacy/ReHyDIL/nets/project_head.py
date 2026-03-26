import torch
import torch.nn as nn
import numpy as np
# from scipy.signal import gaussian
from scipy.signal.windows import gaussian


class ProjHead(nn.Module):
    def __init__(self):
        super(ProjHead, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.proj_head_Linear = nn.Sequential(
            nn.Linear(1 * 144 * 144, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        in_size = x.size(0)
        x_conv = self.conv1(x)
        out_linear = x_conv.view(in_size, -1)
        out = self.proj_head_Linear(out_linear)
        return out

