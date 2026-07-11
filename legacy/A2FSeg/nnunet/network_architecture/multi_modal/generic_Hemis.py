import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Function
import sys
sys.path.append('/data/zirui/lab2/nnUNet/nnunet/network_architecture/multi_modal')
from buildingblocks import *
import numpy as np


from itertools import chain, combinations
sys.path.append('/data/zirui/lab2/nnUNet')
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.multi_modal.neural_network import SegmentationNetwork

"""
ref.
https://github.com/ReubenDo/U-HVED/blob/master/extensions/u_hemis/u_hemis_net.py

Pytorch implementation of U-HeMIS introduced mixing HeMIS and a U-Net architecture
"""
class Generic_HeMIS(SegmentationNetwork):

    def __init__(self, num_cls = 3, final_sigmoid=False,
                weightInitializer=InitWeights_He(1e-2),):
        super(Generic_HeMIS, self).__init__()


############################################################
        self.weightInitializer = weightInitializer
        self.num_classes = num_cls

############################################################

        self.final_sigmoid = final_sigmoid
        enc_list = []
        for i in range(4): # modality
            enc_list.append(ConvEncoder(1, 8))
        self.enc_list = nn.ModuleList(enc_list)
        
        self.abstraction_op = HeMISAbstractionBlock()
        recon_decoder_list = []
        for i in range(4): # modality
            recon_decoder_list.append(ConvDecoderImg(num_cls=1))
        self.recon_decoder_list = nn.ModuleList(recon_decoder_list)
        
        self.seg_decoder = ConvDecoderImg(num_cls=num_cls)


#######################################################################
        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
#######################################################################


    # def forward(self, x, drop=None, **kwargs):
    def forward(self, x, subset_idx_list=[14], instance_missing=False, drop=None, **kwargs):

        
        # if drop is None:
        #     drop = torch.sum(x, [2,3,4]) == 0

#########################################################################
        N, C, D, H, W = x.shape
        if instance_missing: # instance missing
            if drop is None: # 默认全为True
                drop = torch.sum(x, [2,3,4]) == 0
        else: # batch missing
            #SUBSETS_MODALITIES[subset_idx_list[0]] = (0,) | (0,1) | (0,1,2) | (0,1,2,4)
            drop = np.array([True if k in SUBSETS_MODALITIES[subset_idx_list[0]] else False for k in range(4)]) == False
#             x[:, drop] = 0 # drop
            drop = np.expand_dims(drop, 0).repeat(N, axis=0)
            drop = torch.from_numpy(drop).cuda()
#########################################################################

        list_skips = [[] for _ in range(4)] # skip level 4
        for i in range(4): # modality
            enc_out_list = self.enc_list[i](x[:, i:i+1])
            for level, enc_out in enumerate(enc_out_list):
                list_skips[level].append(ZeroLayerF.apply(enc_out, drop[:, i]))
        
        # Sample from the posterior distribution P(latent variables|input)
        skip_flow = []
        for k in range(len(list_skips)):
            sample = self.abstraction_op(list_skips[k])
            skip_flow.append(sample)
        
        recon_list = []
        for i in range(4): # modality
            recon = self.recon_decoder_list[i](skip_flow)
            recon_list.append(recon)
        
        recon_out = torch.cat(recon_list, 1)
        seg_out = self.seg_decoder(skip_flow)
        
        # if self.final_sigmoid:
        #     seg_out = F.sigmoid(seg_out)
        # else:
        #     seg_out = F.softmax(seg_out, 1)
        
        return seg_out, recon_out

class ConvEncoder(nn.Module):

    def __init__(self, in_channels = 1, n_base_ch = 8, conv_layer_order='ilc'):
        super(ConvEncoder, self).__init__()
        self.skip_ind = [1, 3, 5, 7]
        
        layers = [BasicConv(in_channels, n_base_ch, 1, 
                                    stride=1, padding=(1-1) // 2, relu=True, norm=False)]
        
        layers.append(ResBlock(n_base_ch, n_base_ch, encoder=True, kernel_size=3, # 1
                                order=conv_layer_order, padding=(3-1) // 2))
        layers.append(nn.MaxPool3d(kernel_size=2))
        layers.append(ResBlock(n_base_ch // 2, n_base_ch*2, encoder=True, kernel_size=3, # 3
                                order=conv_layer_order, padding=(3-1) // 2))
        layers.append(nn.MaxPool3d(kernel_size=2))
        layers.append(ResBlock(n_base_ch, n_base_ch*4, encoder=True, kernel_size=3, # 5
                                order=conv_layer_order, padding=(3-1) // 2))
        layers.append(nn.MaxPool3d(kernel_size=2))
        layers.append(ResBlock(n_base_ch*2, n_base_ch*8, encoder=True, kernel_size=3, # 7
                                order=conv_layer_order, padding=(3-1) // 2))
        layers.append(nn.MaxPool3d(kernel_size=2))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        
        output = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.skip_ind:
                output.append(x)
        
        return output

class HeMISAbstractionBlock(nn.Module):

    def __init__(self):
        super(HeMISAbstractionBlock, self).__init__()
        
    def forward(self, x):
        x = torch.stack(x, 0)
        mean_x = torch.mean(x, 0)
        var_x = torch.var(x, 0)
        abstraction_output = torch.cat([mean_x, var_x], 1) # channel
        
        return abstraction_output


class ConvDecoderImg(nn.Module):

    def __init__(self, n_base_ch = 8, num_cls=3, conv_layer_order='ilc'):
        super(ConvDecoderImg, self).__init__()
        
        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.d3_c1 = ResBlock(n_base_ch*8 + n_base_ch*4, n_base_ch*4, kernel_size=3,
                                order=conv_layer_order, padding=(3-1) // 2)
        
        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.d2_c1 = ResBlock(n_base_ch*4 + n_base_ch*2, n_base_ch*2, kernel_size=3,
                                order=conv_layer_order, padding=(3-1) // 2)
        
        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.d1_c1 = ResBlock(n_base_ch*2 + n_base_ch, n_base_ch, kernel_size=3,
                                order=conv_layer_order, padding=(3-1) // 2)
        
        self.final_conv = BasicConv(n_base_ch, num_cls, 1, stride=1, padding=(1-1) // 2, relu=False, norm=False)

    def forward(self, list_skips):
        x = list_skips[3]
        
        x = self.d3(x)
        x = torch.cat([x, list_skips[2]], 1) # 64 + 32
        x = self.d3_c1(x)
            
        x = self.d2(x)
        x = torch.cat([x, list_skips[1]], 1) # 32 + 16
        x = self.d2_c1(x)
        
        x = self.d1(x)
        x = torch.cat([x, list_skips[0]], 1) # 16 + 8
        x = self.d1_c1(x)
        x = self.final_conv(x)
        
        return x

class ResBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, encoder=False, kernel_size=3, pool_stride=1, order='gcr', num_groups=8, padding=1):
        super(ResBlock, self).__init__()
        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels // 2
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, 1, order, num_groups,
                                   padding=padding))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, pool_stride, order, num_groups,
                                   padding=padding))
    

################################################################################
MODALITIES = [0,1,2,3]
def all_subsets(l):
    #Does not include the empty set
    return list(chain(*map(lambda x: combinations(l, x), range(1, len(l)+1))))

SUBSETS_MODALITIES = all_subsets(MODALITIES)
################################################################################

if __name__ == "__main__":
    network = U_HeMIS(num_cls=4)
    input = torch.zeros((1, 4, 128, 128, 128))
    drop = torch.from_numpy(np.array([[True, False, True, False]]))
    seg_out, recon_out = network(input, drop = drop)
    # torch.Size([1, 4, 128, 128, 128]) torch.Size([1, 4, 128, 128, 128])
    print(seg_out.size(), recon_out.size())