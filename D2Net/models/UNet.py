import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision

from models.sync_batchnorm import SynchronizedBatchNorm3d
import sys


def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    elif norm == 'sync_bn':
        m = SynchronizedBatchNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


class SEBlock(nn.Module):
    def __init__(self,in_channels,out_channels,net_mode='3d'):
        super(SEBlock,self).__init__()
        if net_mode == '2d':
            self.gap=nn.AdaptiveAvgPool2d(1)
            conv=nn.Conv2d
        elif net_mode == '3d':
            self.gap=nn.AdaptiveAvgPool3d(1)
            conv=nn.Conv3d
        else:
            self.gap=None
            conv=None
        self.conv1=conv(in_channels,out_channels,1)
        self.conv2=conv(in_channels,out_channels,1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        inpu=x
        x=self.gap(x)
        x=self.conv1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.sigmoid(x)
        return inpu*x


class DenseBlock(nn.Module):
    def __init__(self,channels,conv_num,net_mode='3d'):
        super(DenseBlock,self).__init__()
        self.conv_num=conv_num
        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None
        self.relu=nn.ReLU()
        self.conv_list=[]
        self.bottle_conv_list=[]
        for i in conv_num:
            self.bottle_conv_list.append(conv(channels*(i+1),channels*4,1))
            self.conv_list.append(conv(channels*4,channels,3,padding=1))

    def forward(self,x):
        res_x=[]
        res_x.append(x)
        for i in self.conv_num:
            inputs=torch.cat(res_x,dim=1)
            x=self.bottle_conv_list[i](inputs)
            x=self.relu(x)
            x=self.conv_list[i](x)
            x=self.relu(x)
            res_x.append(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, net_mode='3d'):
        super(ResBlock, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        if net_mode == '2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            bn = SynchronizedBatchNorm3d #nn.BatchNorm3d
        else:
            conv = None
            bn = None
        self.conv1 = conv(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = bn(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(out_channels, out_channels, 3, stride=stride, padding=1)
        self.bn2 = bn(out_channels)
        if in_channels!=out_channels:
            self.res_conv=conv(in_channels,out_channels,1,stride=stride)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            res=self.res_conv(x)
        else:
            res=x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        out = x + res
        out = self.relu(out)
        return out


class Up(nn.Module):
    def __init__(self, down_in_channels, in_channels, out_channels, conv_block, interpolation=True, net_mode='3d'):
        super(Up, self).__init__()
        if net_mode == '2d':
            inter_mode = 'bilinear'
            trans_conv = nn.ConvTranspose2d
        elif net_mode == '3d':
            inter_mode = 'trilinear'
            trans_conv = nn.ConvTranspose3d
        else:
            inter_mode = None
            trans_conv = None
        if interpolation == True:
            self.up = nn.Upsample(scale_factor=2, mode=inter_mode, align_corners=True) 
        else:
            self.up = trans_conv(down_in_channels, down_in_channels, 2, stride=2)
        self.conv = RecombinationBlock(in_channels + down_in_channels, out_channels, net_mode=net_mode)

    def forward(self, down_x, x):
        up_x = self.up(down_x)
        x = torch.cat((up_x, x), dim=1)
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block, net_mode='3d'):
        super(Down, self).__init__()
        if net_mode == '2d':
            maxpool = nn.MaxPool2d
        elif net_mode == '3d':
            maxpool = nn.MaxPool3d
        else:
            maxpool = None
        self.conv = RecombinationBlock(in_channels, out_channels, net_mode=net_mode)
        self.down = maxpool(2, stride=2)   

    def forward(self, x):
        x = self.conv(x)
        out = self.down(x)
        return x, out


class MultiBranch(nn.Module):
    def __init__(self, modality_num, in_channels, out_channels, conv_block, net_mode='3d'):  # (8,16)
        super(MultiBranch, self).__init__()
        self.multi_branch_conv1 = nn.Conv3d(1*modality_num, modality_num*in_channels, 3, groups=modality_num, padding=1)
        self.conv = RecombinationBlock(in_channels, out_channels, net_mode=net_mode)
        self.down = nn.MaxPool3d(2, stride=2)   
        self.conv1x1 = nn.Conv3d(out_channels, int(out_channels/modality_num), 1) # (16,4,1)
        self.in_channels = in_channels
        self.conv1x1_output = nn.Conv3d(4*8, 8, 1)
    def forward(self, x):
        c = self.in_channels
        x = self.multi_branch_conv1(x)  # [bs,4*8,128,128,128]
       
        x1 = (self.conv(x[:,0:c,:,:,:]))
        x2 = (self.conv(x[:,c:2*c,:,:,:]))
        x3 = (self.conv(x[:,2*c:3*c,:,:,:]))
        x4 = (self.conv(x[:,3*c:4*c,:,:,:]))
        x = torch.cat([x1,x2,x3,x4],axis=1)  # [bs,4*4,128,128,128] / [bs,4*8,128,128,128]
        del x1,x2,x3,x4
        torch.cuda.empty_cache() 
        out = self.down(x)
        x = self.conv1x1_output(x)
        return x, out


class CropLayer(nn.Module):
    def __init__(self):
        super(CropLayer, self).__init__()
    def forward(self, input_data, offset):
        """
        Currently, only for specific axis, the same offset. Assume for h, w dim.
        """
        cropped_data = input_data[:, :, offset:-offset, offset:-offset, offset:-offset]
        return cropped_data


class SegSEBlock(nn.Module):
    def __init__(self, in_channels, rate=2, net_mode='3d'):
        super(SegSEBlock, self).__init__()
        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None
        self.in_channels = in_channels
        self.rate = rate
        self.dila_conv = conv(self.in_channels, self.in_channels // self.rate, 3, padding=2, dilation=self.rate)
        self.conv1 = conv(self.in_channels // self.rate, self.in_channels, 1)

    def forward(self, input):
        x = self.dila_conv(input)
        x = self.conv1(x)
        x = nn.Sigmoid()(x)
        return x


class RecombinationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normalization=True, kernel_size=3, net_mode='3d'):
        super(RecombinationBlock, self).__init__()
        if net_mode == '2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            bn = SynchronizedBatchNorm3d #nn.BatchNorm3d
        else:
            conv = None
            bn = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bach_normalization = batch_normalization
        self.kerenl_size = kernel_size
        self.rate = 2
        self.expan_channels = self.out_channels * self.rate

        self.expansion_conv = conv(self.in_channels, self.expan_channels, 1)
        self.skip_conv = conv(self.in_channels, self.out_channels, 1)
        self.zoom_conv = conv(self.out_channels * self.rate, self.out_channels, 1)

        self.bn = bn(self.expan_channels)
        self.norm_conv = conv(self.expan_channels, self.expan_channels, self.kerenl_size, padding=1)

        self.segse_block = SegSEBlock(self.expan_channels, net_mode=net_mode)

    def forward(self, input):
        x = self.expansion_conv(input)
        for i in range(1):
            if self.bach_normalization:
                x = self.bn(x)
            x = nn.ReLU6()(x)
            x = self.norm_conv(x)
        se_x = self.segse_block(x)
        x = x * se_x
        x = self.zoom_conv(x)
        skip_x = self.skip_conv(input)
        out = x + skip_x
        return out


class Unet(nn.Module):
    def __init__(self, in_channels=4, filter_num_list=[8, 16, 32, 48, 64], class_num=4, conv_block=RecombinationBlock, net_mode='3d', edge_num=1, label_downsample=False, multi_branch=False, casenet=False):
        super(Unet, self).__init__()
        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None
        self.inc = conv(in_channels, 16, 1)
        self.filter_num_list = filter_num_list
        self.class_num = class_num
        self.edge_num = edge_num
        self.label_downsample = label_downsample
        self.multi_branch = multi_branch
        self.casenet = casenet
        self.crop_layer = CropLayer()

        # Multi-branch input:
        self.multi_branch1 = MultiBranch(modality_num=4, in_channels=4, out_channels=filter_num_list[0], conv_block=conv_block, net_mode=net_mode)

        # down
        self.down1 = Down(16, filter_num_list[0], conv_block=conv_block, net_mode=net_mode)
        self.down2 = Down(filter_num_list[0], filter_num_list[1], conv_block=conv_block, net_mode=net_mode)
        self.down3 = Down(filter_num_list[1], filter_num_list[2], conv_block=conv_block, net_mode=net_mode)
        self.down4 = Down(filter_num_list[2], filter_num_list[3], conv_block=conv_block, net_mode=net_mode)

        self.bridge = conv_block(filter_num_list[3], filter_num_list[4], net_mode=net_mode)

        # up
        self.up1 = Up(filter_num_list[4], filter_num_list[3], filter_num_list[3], conv_block=conv_block, net_mode=net_mode)
        self.up2 = Up(filter_num_list[3], filter_num_list[2], filter_num_list[2], conv_block=conv_block, net_mode=net_mode)      
        self.up3 = Up(filter_num_list[2], filter_num_list[1], filter_num_list[1], conv_block=conv_block, net_mode=net_mode)
        self.up4 = Up(filter_num_list[1], filter_num_list[0], filter_num_list[0], conv_block=conv_block, net_mode=net_mode)
                      
        self.branch_fusion = conv_block(8+filter_num_list[0], filter_num_list[0], net_mode=net_mode) # binary_edge
        self.branch_semantic_fusion = conv_block(4+3, class_num, net_mode=net_mode) # semantic_edge fusion
        self.class_conv = conv(filter_num_list[0], class_num, 1)

        # calculate the edge at each layer as side-outputs:
        self.softmax = nn.Softmax(dim=1)
        
        if self.edge_num == 1: # edge as fore/back-ground
            self.side_out_d1 = conv(filter_num_list[0], 1, 1)
            self.side_out_d1_multibranch = conv(filter_num_list[0], 4, 1, groups=4)
            self.side_out_d2 = conv(filter_num_list[1], 1, 1)
            self.side_out_d3 = conv(filter_num_list[2], 1, 1)
            self.side_out_d4 = conv(filter_num_list[3], 1, 1)
            self.upsample_d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.upsample_d3 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
            self.upsample_d4 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)

            self.side_out_u1 = conv(filter_num_list[3], 1, 1)
            self.side_out_u2 = conv(filter_num_list[2], 1, 1)
            self.side_out_u3 = conv(filter_num_list[1], 1, 1)
            self.side_out_u4 = conv(filter_num_list[0], 1, 1)
            self.upsample_u1 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
            self.upsample_u2 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
            self.upsample_u3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

            self.score_final = conv(8, 1, 1)
        else:  # semantic edge detection
            if self.casenet:
                self.side_out_d1 = conv(filter_num_list[0], 1, 1)
                self.side_out_d1_multibranch = conv(4*filter_num_list[0], 4*1, 1, groups=4)
                self.side_out_d2 = conv(filter_num_list[1], 1, 1)
                self.side_out_d3 = conv(filter_num_list[2], 1, 1)
                self.side_out_d4 = conv(filter_num_list[3], edge_num, 1)
                self.upsample_d2 = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False) # 4
                self.upsample_d3 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=4, bias=False) # 8
                self.upsample_d4 = nn.ConvTranspose2d(edge_num, edge_num, kernel_size=8, stride=8, groups=edge_num, bias=False) # 16

                self.side_out_embedding = conv(filter_num_list[4], edge_num, 1)
                self.upsample_embedding = nn.ConvTranspose2d(edge_num, edge_num, kernel_size=4, stride=16, groups=edge_num, bias=False)

                self.side_out_u1 = conv(filter_num_list[3], edge_num, 1)
                self.side_out_u2 = conv(filter_num_list[2], 1, 1)
                self.side_out_u3 = conv(filter_num_list[1], 1, 1)
                self.side_out_u4 = conv(filter_num_list[0], 1, 1)
                self.upsample_u1 = nn.ConvTranspose2d(edge_num, edge_num, kernel_size=8, stride=8, groups=edge_num, bias=False)
                self.upsample_u2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=4, bias=False)
                self.upsample_u3 = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
            else:
                self.side_out_d1 = conv(filter_num_list[0], edge_num, 1)
                self.side_out_d1_multibranch = conv(filter_num_list[0], 4*edge_num, 1, groups=4)
                self.side_out_d2 = conv(filter_num_list[1], edge_num, 1)
                self.side_out_d3 = conv(filter_num_list[2], edge_num, 1)
                self.side_out_d4 = conv(filter_num_list[3], edge_num, 1)
                self.upsample_d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
                self.upsample_d3 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
                self.upsample_d4 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)

                self.side_out_u1 = conv(filter_num_list[3], edge_num, 1)
                self.side_out_u2 = conv(filter_num_list[2], edge_num, 1)
                self.side_out_u3 = conv(filter_num_list[1], edge_num, 1)
                self.side_out_u4 = conv(filter_num_list[0], edge_num, 1)
                self.upsample_u1 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
                self.upsample_u2 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
                self.upsample_u3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

            self.score_final = conv(8*self.edge_num, self.edge_num, 1, groups=self.edge_num)

    def forward(self, input):
        x = input
        if self.multi_branch:
            conv1, x = self.multi_branch1(x) # [bs,4*8,128,128,128]
            conv1_tmp = self.side_out_d1_multibranch(conv1)
            
            if self.edge_num != 1:  # Semantic edge modeling:
                b,c,h,w,d = x.shape
                _conv1 = torch.zeros([b,self.edge_num*1,h,w,d]).float()
                for i in range(self.edge_num):  # modality_num=4
                    _conv1[:,i,:,:,:] = conv1_tmp[0+i] + conv1_tmp[3+i] + conv1_tmp[6+i] + conv1_tmp[9+i]
                do1 = _conv1
            elif self.edge_num == 1:
                do1 = torch.sum(conv1_tmp, axis=1)
                do1 = torch.unsqueeze(do1, axis=1)
        else:
            x = self.inc(x)
            conv1, x = self.down1(x)
            do1 = self.side_out_d1(conv1)

        conv2, x = self.down2(x) 
        do2_out = self.side_out_d2(conv2)
        do2 = self.upsample_d2(do2_out)

        conv3, x = self.down3(x)
        do3_out = self.side_out_d3(conv3)
        do3 = self.upsample_d3(do3_out)

        conv4, x = self.down4(x)
        do4_out = self.side_out_d4(conv4)
        do4 = self.upsample_d4(do4_out)
        
        x = self.bridge(x)

        x = self.up1(x, conv4)
        uo1_out = self.side_out_u1(x)
        uo1 = self.upsample_u1(uo1_out)

        x = self.up2(x, conv3)
        uo2_out = self.side_out_u2(x)
        uo2 = self.upsample_u2(uo2_out)

        x = self.up3(x, conv2)
        uo3_out = self.side_out_u3(x)
        uo3 = self.upsample_u3(uo3_out)

        x = self.up4(x, conv1)
        uo4 = self.side_out_u4(x)

        # two-branch fusion:
        _x = x

        if self.edge_num == 1:
            fusecat = torch.cat((do1, do2, do3, do4, uo1, uo2, uo3, uo4), dim=1) # Try to use dynamic/attention guided moduel to replace 'concat'
            fuse = self.score_final(fusecat)
        else:
            b,c,h,w,d = input.shape
            fusecat = torch.zeros([b,self.edge_num*8,h,w,d]).float()
            for i in range(self.edge_num):
                fusecat[:,8*i:(8*i+8),:,:,:] = torch.cat((torch.unsqueeze(do1[:,i,:,:,:], 1), torch.unsqueeze(do2[:,i,:,:,:], 1), torch.unsqueeze(do3[:,i,:,:,:], 1), torch.unsqueeze(do4[:,i,:,:,:], 1), torch.unsqueeze(uo1[:,i,:,:,:], 1), torch.unsqueeze(uo2[:,i,:,:,:], 1), torch.unsqueeze(uo3[:,i,:,:,:], 1), torch.unsqueeze(uo4[:,i,:,:,:], 1)), dim=1)
            fuse = self.score_final(fusecat.cuda())
        
        if self.label_downsample:
            results = [do1, do2_out, do3_out, do4_out, uo1_out, uo2_out, uo3_out, uo4, fuse]
        else:
            results = [do1, do2, do3, do4, uo1, uo2, uo3, uo4, fuse]
        results = [torch.sigmoid(r) for r in results]
    
        # --------two-branch fusion:------------
        if self.edge_num == 1:
            fusion = torch.cat([_x, do1, do2, do3, do4, uo1, uo2, uo3, fuse], axis=1)  # binary_edge
            fused_x = self.branch_fusion(fusion)
            fused_x = self.class_conv(fused_x)
            x = nn.Softmax(1)(fused_x)
        if self.edge_num != 1:
            fused_x = self.class_conv(_x)
            fusion = torch.cat([torch.unsqueeze(fused_x[:,0,:,:,:],1), fused_x[:,1:4,:,:,:]*fuse, fuse], axis=1) # concat[edge_out, seg_out*edge_out]
            fusion = self.branch_semantic_fusion(fusion)
            x = nn.Softmax(1)(fusion)
        
        return x, results
