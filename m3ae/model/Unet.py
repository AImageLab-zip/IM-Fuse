#!/usr/bin/env python3
# encoding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random 
from model.models import Model

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

class encoder(nn.Module):
    def __init__(self, init_channels=8):
        super(encoder, self).__init__()
        self.init_channels = init_channels
        self.conv1a = nn.Conv3d(1, init_channels, (3, 3, 3), padding=(1, 1, 1))
        self.conv1b = BasicBlock(init_channels, init_channels*1)  # 32

        self.ds1 = torch.nn.MaxPool3d(2)
                    #nn.Conv3d(init_channels, init_channels * 2, (3, 3, 3), stride=(2, 2, 2),
                            # padding=(1, 1, 1))  # down sampling and add channels

        #self.conv2a = BasicBlock(init_channels * 2, init_channels * 2)
        self.conv2b = BasicBlock(init_channels * 1, init_channels * 2)

        self.ds2 = torch.nn.MaxPool3d(2)
        #nn.Conv3d(init_channels * 2, init_channels * 4, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        #self.conv3a = BasicBlock(init_channels * 4, init_channels * 4)
        self.conv3b = BasicBlock(init_channels * 2, init_channels * 4)

        self.ds3 = torch.nn.MaxPool3d(2)
        #nn.Conv3d(init_channels * 4, init_channels * 8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        #self.conv4a = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4b = BasicBlock(init_channels * 4, init_channels * 8)
        
        #self.conv4c = BasicBlock(init_channels * 8, init_channels * 8)
        #self.conv4d = BasicBlock(init_channels * 8, init_channels * 8)

    def forward(self, x):
        
        c1 = self.conv1a(x)
        c1 = self.conv1b(c1)
        c1d = self.ds1(c1)
        
        #c2 = self.conv2a(c1d)
        c2 = self.conv2b(c1d)
        c2d = self.ds2(c2)
        
        #c3 = self.conv3a(c2d)
        c3 = self.conv3b(c2d)
        c3d = self.ds3(c3)

        #c4 = self.conv4a(c3d)
        c4 = self.conv4b(c3d)
        #c4 = self.conv4c(c4)
        #c4d = self.conv4d(c4)

        return [c1, c2, c3, c4]

class decoder(nn.Module):
    def __init__(self, init_channels=8):
        super(decoder, self).__init__()
        self.up4conva = nn.Conv3d(init_channels * 8, init_channels * 4, (1, 1, 1))
        self.up4 = nn.Upsample(scale_factor=2)  # mode='bilinear'
        self.up4convb = BasicBlock(init_channels * 8, init_channels * 4)

        self.up3conva = nn.Conv3d(init_channels * 4, init_channels * 2, (1, 1, 1))
        self.up3 = nn.Upsample(scale_factor=2)
        self.up3convb = BasicBlock(init_channels * 4, init_channels * 2)

        self.up2conva = nn.Conv3d(init_channels * 2, init_channels, (1, 1, 1))
        self.up2 = nn.Upsample(scale_factor=2)
        self.up2convb = BasicBlock(init_channels*2, init_channels)
        
        self.pool     = nn.MaxPool3d(kernel_size = 2)
        #self.convc    = nn.Conv3d(init_channels * 20, init_channels * 8, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        #self.convco   = nn.Conv3d(init_channels * 16, init_channels * 8, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.up1conv  = nn.Conv3d(init_channels, 3, (1, 1, 1))

    def forward(self, x):
        for l in x:
            print(l.shape)
        u4 = self.up4conva(x[3])
        u4 = self.up4(u4)
        u4 = torch.cat([u4 , x[2]], 1)

        u4 = self.up4convb(u4)
        u3 = self.up3conva(u4)
        u3 = self.up3(u3)
        u3 = torch.cat([u3 , x[1]], 1)

        u3 = self.up3convb(u3)

        u2 = self.up2conva(u3)
        u2 = self.up2(u2)
        u2 = torch.cat([u2 , x[0]], 1)
        u2 = self.up2convb(u2)

        uout = self.up1conv(u2)

        return uout

class UNet3D_hved(nn.Module):
    """
    A normal 3D - Unet, different from the original model architecture in Ref, which use the content and style to reconstruc the high level feature.
    
    3d unet
    Ref:
        3D MRI brain tumor segmentation using autoencoder regularization. Andriy Myronenko
    Args:
        input_shape: tuple, (height, width, depth)
    """

    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=16, p=0.2, deep_supervised = False):
        super(UNet3D_hved, self).__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        
        self.ds = deep_supervised
        
        self.encoders = nn.ModuleList()
        for i in range(4):
            self.encoders.append(encoder(init_channels))
        
        self.decoder =decoder(init_channels)
        self.dropout = nn.Dropout(p=p)
        

    def forward(self, x):
        
        encoder_out = []

        for l in range(4):
            encoder_out.append(self.encoders[l](x[:,l:l+1]))
        
        fs = []
        for l2 in range(4):
            cfeature = torch.zeros_like(encoder_out[0][l2])
            for l in encoder_out:
                cfeature += l[l2]
            cfeature /= 4
            fs.append(cfeature)

        uout = self.decoder(fs)

        
        return uout

class UNet3D_hved2(nn.Module):
    """
    A normal 3D - Unet, different from the original model architecture in Ref, which use the content and style to reconstruc the high level feature.
    
    3d unet
    Ref:
        3D MRI brain tumor segmentation using autoencoder regularization. Andriy Myronenko
    Args:
        input_shape: tuple, (height, width, depth)
    """

    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=32, p=0.2, deep_supervised = False):
        super(UNet3D_hved2, self).__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        
        self.ds = deep_supervised
        
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

        #--------------------------------------------------------------#

        self.conv1a2 = nn.Conv3d(self.in_channels, init_channels, (3, 3, 3), padding=(1, 1, 1))
        self.conv1b2 = BasicBlock(init_channels, init_channels)  # 32

        self.ds12 = nn.Conv3d(init_channels, init_channels * 2, (3, 3, 3), stride=(2, 2, 2),
                             padding=(1, 1, 1))  # down sampling and add channels

        self.conv2a2 = BasicBlock(init_channels * 2, init_channels * 2)
        self.conv2b2 = BasicBlock(init_channels * 2, init_channels * 2)

        self.ds22 = nn.Conv3d(init_channels * 2, init_channels * 4, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv3a2 = BasicBlock(init_channels * 4, init_channels * 4)
        self.conv3b2 = BasicBlock(init_channels * 4, init_channels * 4)

        self.ds32 = nn.Conv3d(init_channels * 4, init_channels * 8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv4a2 = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4b2 = BasicBlock(init_channels * 8, init_channels * 8)
        
        self.conv4c2 = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4d2 = BasicBlock(init_channels * 8, init_channels * 8)

        #--------------------------------------------------------------#

        self.conv1a3 = nn.Conv3d(self.in_channels, init_channels, (3, 3, 3), padding=(1, 1, 1))
        self.conv1b3 = BasicBlock(init_channels, init_channels)  # 32

        self.ds13 = nn.Conv3d(init_channels, init_channels * 2, (3, 3, 3), stride=(2, 2, 2),
                             padding=(1, 1, 1))  # down sampling and add channels

        self.conv2a3 = BasicBlock(init_channels * 2, init_channels * 2)
        self.conv2b3 = BasicBlock(init_channels * 2, init_channels * 2)

        self.ds23 = nn.Conv3d(init_channels * 2, init_channels * 4, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv3a3 = BasicBlock(init_channels * 4, init_channels * 4)
        self.conv3b3 = BasicBlock(init_channels * 4, init_channels * 4)

        self.ds33 = nn.Conv3d(init_channels * 4, init_channels * 8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv4a3= BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4b3 = BasicBlock(init_channels * 8, init_channels * 8)
        
        self.conv4c3 = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4d3 = BasicBlock(init_channels * 8, init_channels * 8)

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
        #self.convc    = nn.Conv3d(init_channels * 20, init_channels * 8, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        #self.convco   = nn.Conv3d(init_channels * 16, init_channels * 8, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.up1conv  = nn.Conv3d(init_channels, self.out_channels, (1, 1, 1))
        
        self.ds_out = []
        self.up_out = []
        if self.ds:
            self.ds_out.append(nn.Conv3d(init_channels*4, self.out_channels, (1, 1, 1)))
            self.up_out.append(nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True))
            
            self.ds_out.append(nn.Conv3d(init_channels*2, self.out_channels, (1, 1, 1)))
            self.up_out.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
            
            self.ds_out = nn.ModuleList(self.ds_out)

    def forward(self, x):
        c1 = self.conv1a(x)
        c1 = self.conv1b(c1)
        c1d = self.ds1(c1)
        
        c2 = self.conv2a(c1d)
        c2 = self.conv2b(c2)
        c2d = self.ds2(c2)
        c2d_p = self.pool(c2d)
        
        c3 = self.conv3a(c2d)
        c3 = self.conv3b(c3)
        c3d = self.ds3(c3)

        c4 = self.conv4a(c3d)
        c4 = self.conv4b(c4)
        c4 = self.conv4c(c4)
        c4d = self.conv4d(c4)

        style = [c2d, c3d, c4d]
        content = c4d 
        
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

        uout = self.up1conv(u2)
        
        
        if self.ds and self.training:
        
            out4 = self.up_out[0](self.ds_out[0](u4))
            out3 = self.up_out[1](self.ds_out[1](u3))
            uout = [out4, out3, uout]
        
        return uout, style, content

class UNet3D_g(nn.Module):
    """
    A normal 3D - Unet, different from the original model architecture in Ref, which use the content and style to reconstruc the high level feature.
    
    3d unet
    Ref:
        3D MRI brain tumor segmentation using autoencoder regularization. Andriy Myronenko
    Args:
        input_shape: tuple, (height, width, depth)
    """

    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=32, p=0.2, deep_supervised = False):
        super(UNet3D_g, self).__init__()
        self.input_shape = input_shape #(128, 128, 128)
        self.in_channels = in_channels #4
        self.out_channels = out_channels #4 in pre-training and 3 in training
        self.init_channels = init_channels #16
        
        self.ds = deep_supervised #False in pre-training, True in training
        
        self.make_encoder()
        self.make_decoder()
        self.dropout = nn.Dropout(p=p) #0.2
        

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
        #self.convc    = nn.Conv3d(init_channels * 20, init_channels * 8, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        #self.convco   = nn.Conv3d(init_channels * 16, init_channels * 8, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.up1conv  = nn.Conv3d(init_channels, self.out_channels, (1, 1, 1))
        
        self.ds_out = []
        self.up_out = []
        if self.ds:
            self.ds_out.append(nn.Conv3d(init_channels*4, self.out_channels, (1, 1, 1)))
            self.up_out.append(nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True))
            
            self.ds_out.append(nn.Conv3d(init_channels*2, self.out_channels, (1, 1, 1)))
            self.up_out.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
            
            self.ds_out = nn.ModuleList(self.ds_out)

    def forward(self, x):
        c1 = self.conv1a(x)
        c1 = self.conv1b(c1)
        c1d = self.ds1(c1)
        
        c2 = self.conv2a(c1d)
        c2 = self.conv2b(c2)
        c2d = self.ds2(c2)
        c2d_p = self.pool(c2d)
        
        c3 = self.conv3a(c2d)
        c3 = self.conv3b(c3)
        c3d = self.ds3(c3)

        c4 = self.conv4a(c3d)
        c4 = self.conv4b(c4)
        c4 = self.conv4c(c4)
        c4d = self.conv4d(c4)

        style = [c2d, c3d, c4d]
        content = c4d 
        
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

        uout = self.up1conv(u2)
        
        
        if self.ds and self.training:
            out4 = self.up_out[0](self.ds_out[0](u4))
            out3 = self.up_out[1](self.ds_out[1](u3))
            uout = [out4, out3, uout]
        
        return uout, style, content
        
class UNet3D_t(nn.Module):
    """3d unet
    Ref:
        3D MRI brain tumor segmentation using autoencoder regularization. Andriy Myronenko
    Args:
        input_shape: tuple, (height, width, depth)
    """

    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=32, p=0.2, deep_supervised = False):
        super(UNet3D_t, self).__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        self.make_encoder()
        self.make_decoder()
        self.dropout = nn.Dropout(p=p)
        self.ds = deep_supervised

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
        
        self.deep_out = []
        if self.ds:
            pass

    def forward(self, x):
        c1 = self.conv1a(x)
        c1 = self.conv1b(c1)
        c1d = self.ds1(c1)
        #print("c1d shape:", c1d.shape)
        
        c2 = self.conv2a(c1d)
        c2 = self.conv2b(c2)
        c2d = self.ds2(c2)
        c2d_p = self.pool(c2d)
#         print("c2d shape:", c2d_p.shape)
        
        c3 = self.conv3a(c2d)
        c3 = self.conv3b(c3)
        c3d = self.ds3(c3)
#         print("c3d shape:", c3d.shape)

        c4 = self.conv4a(c3d)
        c4 = self.conv4b(c4)
        c4 = self.conv4c(c4)
        c4d = self.conv4d(c4) #[1, 128, 20, 24, 16]
#         print("c4d shape:", c4d.shape)

        style = self.convc(torch.cat([c2d_p, c3d, c4d], dim = 1))
        content = c4d 
        
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

        uout = self.up1conv(u2)
        
        return uout, style, content

class UNet3D(nn.Module):
    """3d unet
    Ref:
        3D MRI brain tumor segmentation using autoencoder regularization. Andriy Myronenko
    Args:
        input_shape: tuple, (height, width, depth)
    """

    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=32, p=0.2):
        super(UNet3D, self).__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
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
        c1 = self.conv1a(x)
        c1 = self.conv1b(c1)
        c1d = self.ds1(c1)
        #print("c1d shape:", c1d.shape)
        
        c2 = self.conv2a(c1d)
        c2 = self.conv2b(c2)
        c2d = self.ds2(c2)
        c2d_p = self.pool(c2d)
#         print("c2d shape:", c2d_p.shape)
        
        c3 = self.conv3a(c2d)
        c3 = self.conv3b(c3)
        c3d = self.ds3(c3)
#         print("c3d shape:", c3d.shape)

        c4 = self.conv4a(c3d)
        c4 = self.conv4b(c4)
        c4 = self.conv4c(c4)
        c4d = self.conv4d(c4) #[1, 128, 20, 24, 16]
#         print("c4d shape:", c4d.shape)

        style = self.convc(torch.cat([c2d_p, c3d, c4d], dim = 1))
        content = c4d 
        
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

        uout = self.up1conv(u2)
        
        
        
        return uout, style, content

def ShuffleIndex_with_MDP(index: list, sample_ratio: float, mdp = 0, mask = True):
    temp_index = index.copy()
    
    ### Random selection of the modalities to mask
    mdp_list = [] # which modalities to mask
    mdp = np.random.randint(0, mdp+1) #how many modalities to mask  
    if mdp > 3:
        mdp = 3  
    for l in range(mdp):
        cindex = np.random.randint(0,4)
        while cindex in mdp_list:
            cindex = np.random.randint(0,4)
        mdp_list.append(cindex)
    
    ### Modality masking
    # Remove indices belonging to the selected modalities from temp_index 
    if len(mdp_list) != 0:
        for l in mdp_list:
            for ls in range(l*512, (l+1)*512):
                temp_index.remove(ls)

    ### Patch Masking
    sample_list = []
    if len(index) < 4:
        raise ValueError("inputs must be more than 4")
    elif mask:
        sample_length = int((1-sample_ratio) * len(index)) #how many indices to include in the sample_list based on sample_ratio
        while len(sample_list) < sample_length: #Chooses indices from temp_index until the target length is reached.
            sample = random.choice(temp_index)
            sample_list.append(sample)
            temp_index.remove(sample)
        
        mask_list = [x for x in index if x not in sample_list]  # get the remain index not in cls token and not in sample_index
    else:
        # only with MDP
        sample_list = temp_index
        mask_list = [x for x in index if x not in sample_list]
    
    return sample_list, mask_list 
    
def ShuffleIndex_with_mask_modal(index: list, mask_modal = [], patch_shape = 128):
    interal = int(np.power(patch_shape / 16, 3))
    temp_index = index.copy()
    mdp_list = mask_modal
        
    if len(mdp_list) != 0:
        for l in mdp_list:
            for ls in range(l*interal, (l+1)*interal):
                temp_index.remove(ls)
    sample_list = []
    sample_list = temp_index
    mask_list = [x for x in index if x not in sample_list]
    
    return sample_list, mask_list 
    
def proj(image, patch_size = 16):

    B, C, D, H, W = image.shape
    image_ = image.reshape(B, C, D // patch_size, patch_size, H // patch_size, patch_size, W // patch_size, patch_size)
    
    image_ = image_.permute(0, 1, 2, 4, 6, 3, 5, 7).reshape(B, C * D // patch_size * H // patch_size * W // patch_size,patch_size, patch_size, patch_size)
    
    return image_

def MaskEmbeeding2(B, mask_ratio=0.75, raw_input = None, patch_size = 16, mdp = 0, mask = True, mask_modal = [], patch_shape = 128):
    """get the mask embeeding after patch_emb + pos_emb
    outputs a mask where the unmasked patches correspond to ones, while the masked patches are replaced with zeros
    """
    
    D, H, W = patch_shape, patch_shape, patch_shape
    #int(np.power(patch_shape / 16, 3)) * 4 = total number of tokens
    token_index = [x for x in range(0, int(np.power(patch_shape / 16, 3)) * 4)] #A list of indices for all tokens
    
    #sample_index: Indices of unmasked tokens
    #mask_index: Indices of masked tokens
    if len(mask_modal) == 0:  # do not mask specific modal during pre-training, but random modalities
        sample_index, mask_index = ShuffleIndex_with_MDP(token_index, mask_ratio, mdp = mdp, mask = mask)
    else:
        if -1 in mask_modal:
            sample_index, mask_index = ShuffleIndex_with_mask_modal(token_index, mask_modal = [], patch_shape = patch_shape)
        else: #mask specific modalities during training
            sample_index, mask_index = ShuffleIndex_with_mask_modal(token_index, mask_modal = mask_modal, patch_shape = patch_shape)
    
    
    decoder_embeeding = torch.zeros((B, raw_input.shape[1], patch_size, patch_size, patch_size)).to(raw_input.device)
    
    decoder_embeeding[:,  sample_index, :, :, :] = raw_input[:,  sample_index, :, :, :]
    
    #(B, 2048, 16, 16, 16) -> (B, 4, 8, 16, 8, 16, 8, 16)
    decoder_embeeding = decoder_embeeding.reshape(B, 4, D // patch_size, H // patch_size, W // patch_size, patch_size, patch_size, patch_size).permute(0, 1, 2, 5, 3, 6, 4, 7)
    
    decoder_embeeding = decoder_embeeding.reshape(B, 4, D, H, W) #(B, 4, 128, 128, 128)
    
    return decoder_embeeding
        
class Unet_missing(nn.Module):

    def __init__(self, input_shape, in_channels=4, out_channels=4, init_channels=16, p=0.2, pre_train = False, deep_supervised = False, mdp = 0, mask_modal = [], patch_shape = 128,  mask_ratio = 0.875, augment = False):
        super(Unet_missing, self).__init__()
        self.unet = UNet3D_g(input_shape, in_channels, out_channels, init_channels, p, deep_supervised = deep_supervised)
        
        self.limage = nn.Parameter(torch.randn((1, 4, 155, 240, 240))) #A learnable latent image initialized randomly
        self.patch_shape = patch_shape
        self.raw_input = proj(torch.ones((1, 4, patch_shape, patch_shape, patch_shape))) #(B, 2048, 16, 16, 16)
        self.token_index = [x for x in range(0, self.raw_input.shape[1])]
        self.pre_train = pre_train
        self.mask_ratio = mask_ratio
        self.mdp = mdp
        self.mask_modal = mask_modal
        self.augment = augment
        
    def forward(self, x, location = None, fmdp = None, aug_choises = None):
        #location = bounding box
        ###Pretraining Phase (self.pre_train=True)
        if self.pre_train and location == None: # never into this branch
        
            mask = MaskEmbeeding2(x.shape[0], mask_ratio = self.mask_ratio, raw_input = self.raw_input.to(x.device), mdp = self.mdp, mask = True)
            x = x * mask + self.limage[:, :, :128, :128, :128] * (1-mask)
            
        elif self.pre_train and self.training:
            # print(location, x.shape)
            # Fills the masked regions using self.limage based on the location parameter, which provides the spatial coordinates for replacement.
            x_new = []
            for l in range(x.shape[0]):
                mask = MaskEmbeeding2(1, mask_ratio = self.mask_ratio, raw_input = self.raw_input.to(x.device), mdp = self.mdp, mask = True)
                print(f"input shape: {x[l].shape}, locations: {(location[0][0][l].item(), location[0][1][l].item(), location[1][0][l].item(), location[1][1][l].item(), location[2][0][l].item(), location[2][1][l].item())}, limage shape: {self.limage[:, :, location[0][0][l]: location[0][1][l], location[1][0][l]: location[1][1][l], location[2][0][l]: location[2][1][l]].shape}")
                x_new.append(x[l] * mask + self.limage[:, :, location[0][0][l]: location[0][1][l], location[1][0][l]: location[1][1][l], location[2][0][l]: location[2][1][l]] * (1-mask) )# not detach here
            x = torch.cat(x_new)

        ###2. Training Phase (self.training=True)  (no masking (already done)) 
        elif self.mdp != 0 and self.training: 
            # input x is already augmented and masked
            if fmdp == None:              # in finetune, the x inputed is already do the data augmentation and masking. so no need mask again
                mask = torch.ones_like(x)
                pass
            else:
                # Dynamically selects a subset of modalities (cmask_modal) to mask
                if fmdp == 0:
                    cmask_modal = [-1]
                else:
                    cmask_modal = [] #indices of the modalities that will be masked
                    cmdp = fmdp.cpu()[0] #number of modalities to mask
                    for l in range(cmdp):
                        cindex = np.random.randint(0,4) #index of the modality to be masked
                        while cindex in cmask_modal:
                            cindex = np.random.randint(0,4)
                        cmask_modal.append(cindex)
                mask = MaskEmbeeding2(x.shape[0], raw_input = self.raw_input.to(x.device), mdp = self.mdp, mask = False, mask_modal = cmask_modal)
                x = x * mask + self.limage[:,:,location[0][0]: location[0][1],location[1][0]: location[1][1],location[2][0]: location[2][1]] * (1-mask) # detach here
        else:
            mask = torch.ones_like(x)
        
        ###### Inference Phase (self.training=False)
        if len(self.mask_modal) != 0 and not self.training:  # in inference mask=False not to do patch masking, but only modal masking
            mask = MaskEmbeeding2(x.shape[0], mask_ratio = self.mask_ratio, raw_input = self.raw_input.to(x.device), mdp = self.mdp, mask = False, mask_modal = self.mask_modal, patch_shape = self.patch_shape)
            x = x * mask + self.limage[:,:,location[0][0]: location[0][1],location[1][0]: location[1][1],location[2][0]: location[2][1]].detach() * (1-mask)                
        
        #uout = (B, out-channels, 128, 128, 128) in pre-training and out=[out4, out3, uout] during training, style = [c2d, c3d, c4d], content = c4d
        uout, style, content = self.unet(x)
        
        if self.training:
            return uout, mask.sum((2,3,4)), style, content
        else:
            return uout


class Unet_module(nn.Module):

    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=32, p=0.2):
        super(Unet_module, self).__init__()
        self.unet = UNet3D(input_shape, in_channels, out_channels, init_channels, p)
 
    def forward(self, x):
        uout, style, content = self.unet(x)
        return uout, style, content
