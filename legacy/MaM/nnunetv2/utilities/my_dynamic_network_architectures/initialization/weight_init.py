from torch import nn
import sys
import os

# 假设FolderA和FolderB都在同一级目录下
parent_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件的绝对路径，然后获取其父目录
folder_b_path = os.path.join(parent_dir, '../building_blocks')  # 构建FolderB的路径
from residual import BasicBlockD, BottleneckD


class InitWeights_He(object):
    def __init__(self, neg_slope: float = 1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class InitWeights_XavierUniform(object):
    def __init__(self, gain: int = 1):
        self.gain = gain

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.xavier_uniform_(module.weight, self.gain)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


def init_last_bn_before_add_to_0(module):
    if isinstance(module, BasicBlockD):
        module.conv2.norm.weight = nn.init.constant_(module.conv2.norm.weight, 0)
        module.conv2.norm.bias = nn.init.constant_(module.conv2.norm.bias, 0)
    if isinstance(module, BottleneckD):
        module.conv3.norm.weight = nn.init.constant_(module.conv3.norm.weight, 0)
        module.conv3.norm.bias = nn.init.constant_(module.conv3.norm.bias, 0)
