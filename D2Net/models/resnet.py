# NOTE! OS: output stride, the ratio of input image resolution to final output resolution
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from models.sync_batchnorm import SynchronizedBatchNorm3d

# ----------------------------------------------- 3D Resnet -----------------------------------------------
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = SynchronizedBatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = SynchronizedBatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = SynchronizedBatchNorm3d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = SynchronizedBatchNorm3d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = SynchronizedBatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, input_channels, block, layers, num_classes=1, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = SynchronizedBatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                SynchronizedBatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(input_channels, **kwargs):
    model = ResNet(input_channels, BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(input_channels, **kwargs):
    model = ResNet(input_channels, BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(input_channels, **kwargs):
    model = ResNet(input_channels, Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(input_channels, **kwargs):
    model = ResNet(input_channels, Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(input_channels, **kwargs):
    model = ResNet(input_channels, Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


# -------------------------------------- Resnet for Deeplab --------------------------------------
def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1]*(num_blocks - 1)

    blocks = []
    for stride in strides:
        blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
        in_channels = block.expansion*channels

    layer = nn.Sequential(*blocks) # (*blocks: call with unpacked list entires as arguments)

    return layer

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        out_channels = self.expansion*channels
        
        if type(dilation) != type(1):
            dilation = 1
            
        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = SynchronizedBatchNorm3d(channels)

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = SynchronizedBatchNorm3d(channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = SynchronizedBatchNorm3d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + self.downsample(x)

        out = F.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(Bottleneck, self).__init__()

        out_channels = self.expansion*channels

        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm3d(channels)

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = SynchronizedBatchNorm3d(channels)

        self.conv3 = nn.Conv3d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = SynchronizedBatchNorm3d(out_channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = SynchronizedBatchNorm3d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = out + self.downsample(x)

        out = F.relu(out)

        return out

class ResNet_Bottleneck_OS16(nn.Module):
    def __init__(self, num_layers, input_channels):
        super(ResNet_Bottleneck_OS16, self).__init__()

        if num_layers == 50:
            resnet = resnet50(input_channels)
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])
        elif num_layers == 101:
            resnet = resnet101(input_channels)
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])
        elif num_layers == 152:
            resnet = resnet152(input_channels)
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])
        else:
            raise Exception("num_layers must be in {50, 101, 152}!")

        self.layer5 = make_layer(Bottleneck, in_channels=4*256, channels=512, num_blocks=3, stride=1, dilation=2)

    def forward(self, x):
        c4 = self.resnet(x)

        output = self.layer5(c4)

        return output

class ResNet_BasicBlock_OS16(nn.Module):
    def __init__(self, num_layers, input_channels):
        super(ResNet_BasicBlock_OS16, self).__init__()

        if num_layers == 18:
            resnet = resnet18(input_channels)
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])

            num_blocks = 2

        elif num_layers == 34:
            resnet = resnet34(input_channels)
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])

            num_blocks = 3
        else:
            raise Exception("num_layers must be in {18, 34}!")
    
        self.layer5 = make_layer(BasicBlock, in_channels=256, channels=512, num_blocks=num_blocks, stride=1, dilation=2)

    def forward(self, x):
        c4 = self.resnet(x)

        output = self.layer5(c4)

        return output

class ResNet_BasicBlock_OS8(nn.Module):
    def __init__(self, num_layers, input_channels):
        super(ResNet_BasicBlock_OS8, self).__init__()

        if num_layers == 18:
            resnet = resnet18(input_channels)
            
            self.resnet = nn.Sequential(*list(resnet.children())[:-4])

            num_blocks_layer_4 = 2
            num_blocks_layer_5 = 2

        elif num_layers == 34:
            resnet = resnet34(input_channels)
            
            self.resnet = nn.Sequential(*list(resnet.children())[:-4])

            num_blocks_layer_4 = 6
            num_blocks_layer_5 = 3
        else:
            raise Exception("num_layers must be in {18, 34}!")

        self.layer4 = make_layer(BasicBlock, in_channels=128, channels=256, num_blocks=num_blocks_layer_4, stride=1, dilation=2)

        self.layer5 = make_layer(BasicBlock, in_channels=256, channels=512, num_blocks=num_blocks_layer_5, stride=1, dilation=4)

    def forward(self, x):
        c3 = self.resnet(x)

        output = self.layer4(c3)
        output = self.layer5(output)

        return output

def ResNet18_OS16(input_channels):
    return ResNet_BasicBlock_OS16(num_layers=18, input_channels=input_channels)

def ResNet50_OS16(input_channels):
    return ResNet_Bottleneck_OS16(num_layers=50, input_channels=input_channels)

def ResNet101_OS16(input_channels):
    return ResNet_Bottleneck_OS16(num_layers=101, input_channels=input_channels)

def ResNet152_OS16(input_channels):
    return ResNet_Bottleneck_OS16(num_layers=152, input_channels=input_channels)

def ResNet34_OS16(input_channels):
    return ResNet_BasicBlock_OS16(num_layers=34, input_channels=input_channels)

def ResNet18_OS8(input_channels):
    return ResNet_BasicBlock_OS8(num_layers=18, input_channels=input_channels)

def ResNet34_OS8(input_channels):
    return ResNet_BasicBlock_OS8(num_layers=34, input_channels=input_channels)