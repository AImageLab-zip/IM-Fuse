import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .unetr import UNETR

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=8):
        super(BasicBlock, self).__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.gn2 = nn.GroupNorm(n_groups, in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        residul = x
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)

        x = self.relu2(self.gn2(x))
        x = self.conv2(x)
        x = x + residul

        return x

class MSTKDNet(nn.Module):
    """3d unet
    Ref:
        3D MRI brain tumor segmentation using autoencoder regularization. Andriy Myronenko
    Args:
        input_shape: tuple, (height, width, depth)
    """

    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=32, p=0.2, unetr=True):
        super(MSTKDNet, self).__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        self.unetr = unetr
        self.make_encoder()
        self.make_decoder()
        self.dropout = nn.Dropout(p=p)

    def make_encoder(self):
        init_channels = self.init_channels
        self.conv1a = nn.Conv3d(self.in_channels, init_channels, (3, 3, 3), padding=(1, 1, 1))
        self.conv1b = BasicBlock(init_channels, init_channels)  # 32

        self.ds1 = nn.Conv3d(init_channels, init_channels * 2, (3, 3, 3), stride=(2, 2, 2),
                             padding=(1, 1, 1))  # down sampling and add channels

        self.conv2a = BasicBlock(init_channels * 2, init_channels * 2)
        self.conv2b = BasicBlock(init_channels * 2, init_channels * 2)

        self.ds2 = nn.Conv3d(init_channels * 2, init_channels * 4, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv3a = BasicBlock(init_channels * 4, init_channels * 4)
        self.conv3b = BasicBlock(init_channels * 4, init_channels * 4)

        self.ds3 = nn.Conv3d(init_channels * 4, init_channels * 8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv4a = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4b = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4c = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4d = BasicBlock(init_channels * 8, init_channels * 8)

        if self.unetr:
            self.Unetr = UNETR(img_shape=(20, 24, 16), input_dim=128, output_dim=128, patch_size=4)

    def make_decoder(self):
        init_channels = self.init_channels
        self.up4conva = nn.Conv3d(init_channels * 8, init_channels * 4, (1, 1, 1))
        self.up4 = nn.Upsample(scale_factor=2)  # mode='bilinear'
        self.up4convb = BasicBlock(init_channels * 4, init_channels * 4)

        self.up3conva = nn.Conv3d(init_channels * 4, init_channels * 2, (1, 1, 1))
        self.up3 = nn.Upsample(scale_factor=2)
        self.up3convb = BasicBlock(init_channels * 2, init_channels * 2)

        self.up2conva = nn.Conv3d(init_channels * 2, init_channels, (1, 1, 1))
        self.up2 = nn.Upsample(scale_factor=2)
        self.up2convb = BasicBlock(init_channels, init_channels)
        
        self.pool     = nn.MaxPool3d(kernel_size = 2)
        self.convc    = nn.Conv3d(init_channels * 20, init_channels * 8, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.convco   = nn.Conv3d(init_channels * 16, init_channels * 8, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.up1conv  = nn.Conv3d(init_channels, self.out_channels, (1, 1, 1))

    def forward(self, x):
        c1 = self.conv1a(x) #  [1, 16, 160, 192, 128]
        c1 = self.conv1b(c1) # [1, 16, 160, 192, 128]
        c1d = self.ds1(c1)  # [1, 32, 80, 96, 64]
        #print("c1d shape:", c1d.shape)
        
        c2 = self.conv2a(c1d)
        c2 = self.conv2b(c2) # [1, 32, 80, 96, 64]
        c2d = self.ds2(c2) #  [1, 64, 40, 48, 32]
        c2d_p = self.pool(c2d) # [1, 64, 20, 24, 16]
#         print("c2d shape:", c2d_p.shape)
        
        c3 = self.conv3a(c2d)
        c3 = self.conv3b(c3) # [1, 64, 40, 48, 32]
        c3d = self.ds3(c3) # [1, 128, 20, 24, 16]
#         print("c3d shape:", c3d.shape)

        output, unetr_fs, extract_weights = self.Unetr(c3d)
        output = output + c3d
            
        c4 = self.conv4a(output)
        c4 = self.conv4b(c4)
        c4 = self.conv4c(c4)
        c4d = self.conv4d(c4) # [1, 128, 20, 24, 16]
#       print("c4d shape:", c4d.shape)

        style = self.convc(torch.cat([c2d_p, c3d, output], dim = 1))
        content = c4d 
        Global_style = [c3d, style, c4d]
        
        c4d = self.convco(torch.cat([style, content], dim = 1))
        
        c4d = self.dropout(c4d)

        u4 = self.up4conva(c4d)
        u4 = self.up4(u4)
        u4 = u4 + c3
        u4 = self.up4convb(u4)

        u3 = self.up3conva(u4)
        u3 = self.up3(u3)
        u3 = u3 + c2
        u3 = self.up3convb(u3)

        u2 = self.up2conva(u3)
        u2 = self.up2(u2)
        u2 = u2 + c1
        u2 = self.up2convb(u2)

        logit = self.up1conv(u2)
        uout = F.sigmoid(logit)

        return uout, style, content, unetr_fs, extract_weights, Global_style, logit,

class Unet_module(nn.Module):
    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=32, p=0.2):
        super(Unet_module, self).__init__()
        self.unet = MSTKDNet(input_shape, in_channels, out_channels, init_channels, p)
 
    def forward(self, x):
        uout, style, content = self.unet(x)
        return uout, style, content


if __name__ == "__main__":
    input1 = torch.randn(2, 128, 20, 24, 16)  # (N=2, C=4, D=128, H=128, W=128)
    model = MSTKDNet(img_shape=(20, 24, 16), input_dim=128, output_dim=128, patch_size=4)
    output, z3, z6, z9, z12 = model(input1)
    print(output.shape)
