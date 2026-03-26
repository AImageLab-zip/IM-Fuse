import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import config
import time
module = 'separable_adapter' # specific module type for universal model: series_adapter, parallel_adapter, separable_adapter
trainMode = 'universal'
from models.sync_batchnorm import SynchronizedBatchNorm3d


'''
Input: (N, C_{in}, D_{in}, H_{in}, W_{in})
Output: (N, C_{out}, D_{out}, H_{out}, W_{out})
'''
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
        # raise ValueError('There is no this kind of normalization method!')
    return m

class SegSEBlock(nn.Module):
    def __init__(self, in_channels, rate=2, net_mode='3d'):
        super(SegSEBlock, self).__init__()
        nn.Conv3d = nn.Conv3d
        self.in_channels = in_channels
        self.rate = rate
        self.dila_conv = nn.Conv3d(self.in_channels, self.in_channels // self.rate, 3, padding=2, dilation=self.rate)
        self.conv1 = nn.Conv3d(self.in_channels // self.rate, self.in_channels, 1)
    def forward(self, input):
        x = self.dila_conv(input)
        x = self.conv1(x)
        x = nn.Sigmoid()(x)
        return x

class RecombinationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normalization=True, kernel_size=3, net_mode='3d'):
        super(RecombinationBlock, self).__init__()
        # conv = nn.Conv3d
        bn = SynchronizedBatchNorm3d #nn.BatchNorm3d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bach_normalization = batch_normalization
        self.kerenl_size = kernel_size
        self.rate = 2  # 2
        self.expan_channels = self.out_channels * self.rate

        self.expansion_conv = nn.Conv3d(self.in_channels, self.expan_channels, 1)
        self.skip_conv = nn.Conv3d(self.in_channels, self.out_channels, 1)
        self.zoom_conv = nn.Conv3d(self.out_channels * self.rate, self.out_channels, 1)

        self.bn = bn(self.expan_channels)
        self.norm_conv = nn.Conv3d(self.expan_channels, self.expan_channels, self.kerenl_size, padding=1)

        self.segse_block = SegSEBlock(self.expan_channels, net_mode=net_mode)
    def forward(self, input):
        x = self.expansion_conv(input)
        for i in range(1):
            if self.bach_normalization:
                x = self.bn(x)
            x = nn.ReLU6(inplace=True)(x)
            x = self.norm_conv(x)
        # se_x = self.segse_block(x)
        # x = x * se_x
        x = self.zoom_conv(x)
        skip_x = self.skip_conv(input)
        out = x + skip_x
        return out

class DilatedConv3DBlock(nn.Module):
    def __init__(self, num_in, num_out, kernel_size=(1,1,1), stride=1, g=1, d=(1,1,1), norm=None):
        super(DilatedConv3DBlock, self).__init__()
        assert isinstance(kernel_size,tuple) and isinstance(d,tuple)
        padding = tuple(
            [(ks-1)//2 *dd for ks, dd in zip(kernel_size, d)])
        self.bn = nn.InstanceNorm3d(num_in, affine=True)
        self.act_fn = nn.LeakyReLU(negative_slope=1e-2,inplace=True)
        self.conv = nn.Conv3d(num_in,num_out,kernel_size=kernel_size,padding=padding,stride=stride,groups=g,dilation=d,bias=False)
    def forward(self, x):
        # h = self.act_fn(self.bn(x))
        h = self.conv(x)
        return h


def num_pool2stride_size(num_pool_per_axis):
    max_num = max(num_pool_per_axis)  # 4
    stride_size_per_pool = list()
    for i in range(max_num): 
        unit = [1,2]
        stride_size_per_pool.append([unit[i<num_pool_per_axis[0]], unit[i<num_pool_per_axis[1]], unit[i<num_pool_per_axis[2]]])
    return stride_size_per_pool  # [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]

def norm_act(nchan, only='both', args=None):
    norm = SynchronizedBatchNorm3d(nchan)    
    act = nn.LeakyReLU(negative_slope=1e-2,inplace=True)
    if only=='norm':
        return norm
    elif only=='act':
        return act
    else:
        return nn.Sequential(norm, act)

class conv1x1(nn.Module):
    def __init__(self, inChans, outChans=None, stride=1, padding=0, args=None):
        super(conv1x1, self).__init__()
        if module == 'series_adapter':
            self.op1 = nn.Sequential(
                norm_act(inChans,only='norm',args=args),
                nn.Conv3d(inChans, inChans, kernel_size=1, stride=1)
                )
        elif module == 'parallel_adapter':
            self.op1 = nn.Conv3d(inChans, outChans, kernel_size=1, stride=stride, padding=padding)
        else:
            self.op1 = nn.Conv3d(inChans, inChans, kernel_size=1, stride=1)
    def forward(self, x):
        out = self.op1(x)
        if module == 'series_adapter':
            out += x
        return out

class dwise(nn.Module):  # 3x3 Conv3d
    def __init__(self, inChans, kernel_size=3, stride=1, padding=1, dilated_conv=False, args=None):
        super(dwise, self).__init__()
        if args.dilated_conv and dilated_conv and inChans>=64:
            self.conv1 = DilatedConv3DBlock(inChans, inChans, kernel_size=(3,3,3), g=inChans)  ###
        else:
            self.conv1 = nn.Conv3d(inChans, inChans, kernel_size=kernel_size, stride=stride, padding=padding, groups=inChans)
            # self.conv1 = nn.Conv3d(inChans, inChans, kernel_size=kernel_size, stride=stride, padding=padding)
        self.op1 = norm_act(inChans,only='both',args=args)
    def forward(self, x):
        out = self.conv1(x)
        out = self.op1(out)
        return out

class pwise(nn.Module): # 1x1 Conv3d
    def __init__(self, inChans, outChans, kernel_size=1, stride=1, padding=0):
        super(pwise, self).__init__()
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding)
    def forward(self, x):
        out = self.conv1(x)
        return out

class conv_unit(nn.Module):  # 2 conv3d layers (3x3+1x1) + bn/act  OR  1 conv3d-with stride-2 to downsample + bn/act
    '''
    variants of conv3d+norm by applying adapter or not.
    '''
    def __init__(self, nb_tasks, inChans, outChans, kernel_size=3, stride=1, padding=1, second=0 , dilated_conv=False, args=None):
        super(conv_unit, self).__init__()
        self.stride = stride
        if self.stride != 1:
            self.conv = nn.Conv3d(inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding) # padding != 0 for stride != 2 if doing padding=SAME.
        elif self.stride == 1:
            if trainMode != 'universal': # independent, shared
                self.conv = nn.Conv3d(inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding) # padding != 0 for stride != 2 if doing padding=SAME.
            else:
                if module in ['series_adapter', 'parallel_adapter']:
                    self.conv = nn.Conv3d(inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding) # padding != 0 for stride != 2 if doing padding=SAME.
                    if module == 'series_adapter':
                        self.adapOps = nn.ModuleList([conv1x1(outChans, args=args) for i in range(nb_tasks)]) # based on https://github.com/srebuffi/residual_adapters/
                    elif module == 'parallel_adapter':
                        self.adapOps = nn.ModuleList([conv1x1(inChans, outChans, args=args) for i in range(nb_tasks)]) 
                    else:
                        pass
                elif module == 'separable_adapter':
                    self.adapOps = nn.ModuleList([dwise(inChans, dilated_conv=dilated_conv, args=args) for i in range(nb_tasks)])
                    self.pwise = pwise(inChans, outChans)
                else:
                    pass                
        self.op = nn.ModuleList([norm_act(outChans, only='norm',args=args) for i in range(nb_tasks)])

    def forward(self, x):
        task_idx = config.task_idx
        if self.stride != 1:
            out = self.conv(x) 
            out = self.op[task_idx](out)
            return out
        elif self.stride == 1:
            if trainMode != 'universal': # independent, shared
                out = self.conv(x)
                out = self.op[task_idx](out)
            else:
                if module in ['series_adapter', 'parallel_adapter']:
                    out = self.conv(x)
                    if module == 'series_adapter':
                        out = self.adapOps[task_idx](out)
                    elif module == 'parallel_adapter':
                        share_map = out
                        para_map = self.adapOps[task_idx](x)
                        out = out + para_map
                    else:
                        pass

                    out = self.op[task_idx](out)
                    if module == 'parallel_adapter':
                        return out, share_map, para_map # for visualization of feature maps
                    else:
                        return out
                elif module == 'separable_adapter':
                    out = self.adapOps[task_idx](x)
                    para_map = out
                    out = self.pwise(out)
                    share_map = out
                    out = self.op[task_idx](out)
                    return out, share_map, para_map
                else:
                    pass

class InputTransition(nn.Module):  # 1 Conv3d + bn_act
    '''
    task specific
    '''
    def __init__(self, inChans, base_outChans,args=None):
        super(InputTransition, self).__init__()
        self.op1 = nn.Sequential(
            nn.Conv3d(inChans, base_outChans, kernel_size=3, stride=1, padding=1),
            norm_act(base_outChans,args=args))
    def forward(self, x):
        out = self.op1(x)
        return out

class DownSample(nn.Module):
    def __init__(self, nb_tasks, inChans, outChans, kernel_size=3, stride=1, padding=1, args=None):
        super(DownSample, self).__init__()
        self.op1 = conv_unit(nb_tasks, inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding, dilated_conv=True, args=args)
        self.act1 = norm_act(outChans, only="act",args=args)
    def forward(self, x):
        out = self.op1(x)
        out = self.act1(out)
        return out

class DownBlock(nn.Module):  # 2 conv-unit: i.e.: Conv:[3x3,1x1,3x3,1x1] with 4 bn/act, also + residual connection
    def __init__(self, nb_tasks, inChans, outChans, kernel_size=3, stride=1, padding=1 ,args=None):
        super(DownBlock, self).__init__()
        self.op1 = conv_unit(nb_tasks, inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding, dilated_conv=True, args=args)
        self.act1 = norm_act(outChans, only="act",args=args)
        self.op2 = conv_unit(nb_tasks, outChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding, dilated_conv=True, args=args)
        self.act2 = norm_act(outChans, only="act",args=args)
    def forward(self, x):
        if module == 'parallel_adapter' or module == 'separable_adapter':
            out, share_map, para_map = self.op1(x)
        else:
            out = self.op1(x)
        out = self.act1(out)
        if module == 'parallel_adapter' or module == 'separable_adapter':
            out, share_map, para_map = self.op2(out)
        else:
            out = self.op2(out)
        if config.residual:
            out = self.act2(x + out)
        else:
            out = self.act2(out)
        return out

def Upsample3D(scale_factor=(2)):
    '''
    task specific
    '''
    # l = tf.keras.layers.UpSampling3D(size=up_strides, data_format=DATA_FORMAT)(l) # by tkuanlun350. # no equavalent in torch?
    # scale_factor can also be a tuple. so able to custom scale_factor for each dim.
    # upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest') # 'nearest' ignore the warnings. Only module like upsample can be shown in my visualization. # if using ConvTranspose3d, be careful to how to pad when the down sample method used padding='SAME' strategy.
    upsample = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True) 
    return upsample

class UnetUpsample(nn.Module):  # Upsample + 1-conv-unit (2-Conv3d)
    def __init__(self, nb_tasks, inChans, outChans, up_stride=(2),args=None):
        super(UnetUpsample, self).__init__()
        self.upsamples = nn.ModuleList(
            [Upsample3D(scale_factor=up_stride) for i in range(nb_tasks)])
        self.op = conv_unit(nb_tasks, inChans, outChans, kernel_size=3,stride=1, padding=1, args=args)
        self.act = norm_act(outChans, only='act',args=args)
    def forward(self, x):
        task_idx = config.task_idx
        out = self.upsamples[task_idx](x)
        if module == 'parallel_adapter' or module == 'separable_adapter':
            out, share_map, para_map = self.op(out)
        else:
            out = self.op(out)
        out = self.act(out)
        if module == 'parallel_adapter' or module == 'separable_adapter':
            return out, share_map, para_map
        else:
            return out

class UpBlock(nn.Module):   # 2-conv-unit (4 Conv3d + bn/act), w/o residual-connection
    def __init__(self, nb_tasks, inChans, outChans, kernel_size=3, stride=1, padding=1 ,args=None):
        super(UpBlock, self).__init__()
        self.op1 = conv_unit(nb_tasks, inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding, args=args)
        self.act1 = norm_act(outChans, only="act",args=args)
        self.op2 = conv_unit(nb_tasks, outChans, outChans, kernel_size=1, stride=1, padding=padding, args=args) # ori: padding=0
        self.act2 = norm_act(outChans, only="act",args=args)
        self.residual_conv = nn.Conv3d(inChans, outChans, kernel_size=1, stride=stride, padding=0)
    def forward(self, x, up_x):
        if module == 'parallel_adapter' or module == 'separable_adapter':
            out, share_map, para_map = self.op1(x)
        else:
            out = self.op1(x)
        out = self.act1(out)
        if module == 'parallel_adapter' or module == 'separable_adapter':
            out, share_map, para_map = self.op2(out)
        else:
            out = self.op2(out)
        
        if config.residual: # same to ResNet # New Add the residual connects in the upsamples
            _x = self.residual_conv(x)
            out = self.act2(_x + out)
        else:
            out = self.act2(out)
        
        return out

class DeepSupervision(nn.Module):
    '''
    task specific
    '''
    def __init__(self, inChans, num_class, up_stride=(2,2,2),args=None):
        super(DeepSupervision, self).__init__()
        self.op1 = nn.Sequential(
            nn.Conv3d(inChans, num_class, kernel_size=1, stride=1, padding=0),
            norm_act(num_class,args=args)) 
        self.op2 = Upsample3D(scale_factor=up_stride)
        self.re_channel = nn.Conv3d(inChans, 8, kernel_size=1, stride=1, padding=0) ## maybe no use

    def forward(self, x, deep_supervision):
        if config.deep_sup_type == 'add':
            if deep_supervision is None:
                out = self.op1(x)
                deep_sup_fea = out
            else:
                deep_sup_fea = self.op1(x)
                out = torch.add(deep_sup_fea, deep_supervision)  # Add 
            out = self.op2(out)
            return out, deep_sup_fea
        elif config.deep_sup_type == 'concat':
            if deep_supervision is None:
                out = self.re_channel(x)   # [1,32,x,x,x]
            else:
                out = torch.cat([self.re_channel(x), deep_supervision], axis=1)  # concat 
            out = self.op2(out)
            return out

class OutputTransition(nn.Module): # 1 Conv3d
    '''
    task specific
    '''
    def __init__(self, inChans, num_class):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, num_class, kernel_size=1, stride=1, padding=0)
      
        self.final_conv = nn.Conv3d(4*8, num_class, kernel_size=1, stride=1, padding=0)
    def forward(self, x, deep_supervision=None):
        if config.deep_sup_type == 'add':
            out = self.conv1(x)
            if deep_supervision is None:
                return out
            else:
                out = torch.add(out, deep_supervision)  # Add  deep_sup: [1,32*4,128,128,128]
                return out
        elif config.deep_sup_type == 'concat':
            out = torch.cat([x, deep_supervision], axis=1)  # Concat
            out = F.dropout3d(out, p=0.5)   # Dropout3d
            out = self.final_conv(out)
            return out


class u2net3d(nn.Module):
    def __init__(self, inChans_list=[4], base_outChans=8, num_class_list=[4], args=None, label_downsample=False, multi_branch=False):  # base_outChans=16
        '''
        Args:
        One or more tasks could be input at once. So lists of inital model settings are passed.
            inChans_list: a list of num_modality for each input task.
            base_outChans: outChans of the inputTransition, i.e. inChans of the first layer of the shared backbone of the universal model.
            depth: depth of the shared backbone.
        '''
        super(u2net3d, self).__init__()
        
        nb_tasks = len(num_class_list) # 1

        self.depth = max(config.num_pool_per_axis) + 1 # 5 num_pool_per_axis firstly defined in train_xxxx.py or main.py
        stride_sizes = num_pool2stride_size(config.num_pool_per_axis)

        self.in_tr_list = nn.ModuleList(
            [InputTransition(inChans_list[j], base_outChans,args=args) for j in range(nb_tasks)]
        ) # task-specific input layers

        outChans_list = list()
        self.down_blocks = nn.ModuleList() # # register modules from regular python list.
        self.down_samps = nn.ModuleList()
        self.down_pads = list() # used to pad as padding='same' in tensorflow

        inChans = base_outChans
        for i in range(self.depth):
            outChans = base_outChans * (2**i)
            outChans_list.append(outChans)
            self.down_blocks.append(DownBlock(nb_tasks, inChans, outChans, kernel_size=3, stride=1, padding=1,args=args))
        
            if i != self.depth-1:
                # stride for each axis could be 1 or 2, depending on tasks. # to apply padding='SAME' as tensorflow, cal and save pad num to manually pad in forward().
                pads = list() # 6 elements for one 3-D volume. originized for last dim backward to first dim, e.g. w,w,h,h,d,d # required for F.pad.
                # pad 1 to the right end if s=2 else pad 1 to both ends (s=1). 
                for j in stride_sizes[i][::-1]:
                    if j == 2:
                        pads.extend([0,1])
                    elif j == 1:
                        pads.extend([1,1])
                self.down_pads.append(pads) 
                self.down_samps.append(DownSample(nb_tasks, outChans, outChans*2, kernel_size=3, stride=tuple(stride_sizes[i]), padding=1, args=args)) # padding=0
                inChans = outChans*2
            else:
                inChans = outChans

        self.up_samps = nn.ModuleList([None] * (self.depth-1))
        self.up_blocks = nn.ModuleList([None] * (self.depth-1))
        self.dSupers = nn.ModuleList() # 1 elements if self.depth =2, or 2 elements if self.depth >= 3
        for i in range(self.depth-2, -1, -1): # i=[4,3,2,1,0]
            self.up_samps[i] = UnetUpsample(nb_tasks, inChans, outChans_list[i], up_stride=stride_sizes[i][0],args=args)
            self.up_blocks[i] = UpBlock(nb_tasks, outChans_list[i]*2, outChans_list[i], kernel_size=3,stride=1, padding=1, args=args)
            if config.deep_supervision and i <= (self.depth-3) and i > 0:
                self.dSupers.append(nn.ModuleList(
                    [DeepSupervision(outChans_list[i], num_class_list[j], up_stride=tuple(stride_sizes[i-1])) for j in range(nb_tasks)]))
            inChans = outChans_list[i]
        self.out_tr_list = nn.ModuleList(
            [OutputTransition(inChans, num_class_list[j]) for j in range(nb_tasks)])
        self.args = args

        # Edge-aware codes:
        if self.args.semantic_edge:
            self.edge_num = 3
        if (not self.args.semantic_edge):
            self.edge_num = 1

        if self.edge_num == 3: # semantic edge detection  ++++ use edge or not the if
            self.side_output_d = nn.ModuleList()
            self.upsample_d = nn.ModuleList()
            upsample_factor_list = [2**(n+1) for n in range(self.depth-2)]  # [2,4,8,16]
            self.side_output_u = nn.ModuleList([None] * (self.depth-1))
            self.upsample_u = nn.ModuleList()
            for i in range(self.depth-1):
                self.side_output_d.append(conv_unit(1, outChans_list[i], self.edge_num, 3, stride=1, padding=1))
                if i != 0:
                    self.upsample_d.append(nn.Upsample(scale_factor=upsample_factor_list[i-1], mode='trilinear', align_corners=True))
            for i in range(self.depth-2, -1, -1):
                self.side_output_u[i] = conv_unit(1, outChans_list[i], self.edge_num, 3, stride=1, padding=1)
                if i != 0:
                    self.upsample_u.append(nn.Upsample(scale_factor=upsample_factor_list[i-1], mode='trilinear', align_corners=True))
        
        if self.edge_num == 1:  # binary edge detection
            todo=1
        
            
    def forward(self, x):
        # x: [N, C, D, H, W]
        task_idx = config.task_idx  # 0
        deep_supervision = None
        
        deep_sup_fea = [] # 3 feaure maps

        out = self.in_tr_list[task_idx](x)  # 1st-3x3conv, [1, 8, 128, 128, 128]
        down_list = list()

        side_output_mid = []  # For edge-mid-outputs
        side_outputs = []    # For edge-outputs

        for i in range(self.depth): # 5/6  5/6*(nn.Conv3d+down_sample)  --Encoder
            out = self.down_blocks[i](out)

            if self.args.use_edge and i != (self.depth-1):   # For edge branch
                _side = self.side_output_d[i](out)[0] # [0]  only for using conv_unit when defining self.side_output_u
                side_output_mid.append(_side)
                if i>0:
                    side_outputs.append(self.upsample_d[i-1](_side))
                    
            # down_list.append(out)
            if i != self.depth-1:  # 5
                down_list.append(out) # will not store the deepest, so as to save memory
                out = self.down_samps[i](out) # 

        idx = 0
        for i in range(self.depth-2, -1, -1):  # i=[(4,) 3,2,1,0]  4/5 * (nn.Conv3d+up_sample+concat) --Decoder
            if module == 'parallel_adapter' or module == 'separable_adapter':
                out, share_map, para_map = self.up_samps[i](out)  # Conduct the true upsample
            else:
                out = self.up_samps[i](out)
            up_x = out
            out = torch.cat((out, down_list[i]), dim=1)
            out = self.up_blocks[i](out, up_x)   # Actually, there is not residual conntection here! (up_x is not used at all! Try?)

            if config.deep_supervision and i <= (self.depth-3) and i > 0: # On 3or4-level, expect the final-level (and smallest 4*4 level)
                deep_supervision, _deep_sup_fea = self.dSupers[idx][task_idx](out, deep_supervision)
                deep_sup_fea.append(_deep_sup_fea) ###
                idx += 1
            if (not config.deep_supervision):
                deep_sup_fea = []
            
            if self.args.use_edge:   # For edge branch
                _side = self.side_output_u[i](out)[0] # [0] only for using conv_unit when defining self.side_output_u
                side_output_mid.append(_side)
                if i>0:
                    side_outputs.append(self.upsample_u[-i+self.depth-2](_side))

        if self.args.use_edge and self.edge_num == 1:
            fusecat = torch.cat([_fea for _fea in side_outputs], dim=1) # Try to use dynamic/attention guided moduel to replace 'concat'
            fuse = self.score_final(fusecat)
        if self.args.use_edge and self.edge_num == 3:
            b,c,h,w,d = x.shape
            fusecat = torch.zeros([b,self.edge_num*len(side_outputs),h,w,d]).float()
            for i in range(self.edge_num):
                fusecat[:,len(side_outputs)*i:(len(side_outputs)*i+len(side_outputs)),:,:,:] = torch.cat([torch.unsqueeze(side[:,i,:,:,:],1) for side in side_outputs], dim=1)
            fuse = self.score_final(fusecat.cuda())

        if self.args.use_edge:
            side_outputs.append(fuse)  # len=11
            edge_results = [torch.sigmoid(r) for r in side_outputs]

        if self.args.use_edge and self.edge_num == 3:
            seg_x = self.out_tr_list[task_idx](out, deep_supervision)
            fusion = torch.cat([torch.unsqueeze(seg_x[:,0,:,:,:],1), seg_x[:,1:4,:,:,:]*fuse, fuse], axis=1) # concat[edge_out, seg_out*edge_out]
            fusion = self.branch_semantic_fusion(fusion)#[0]  ## [0] OR not
            out = fusion
        
        if (not self.args.use_edge):
            out = self.out_tr_list[task_idx](out, deep_supervision)

       
        if self.args.use_edge:
            return out, deep_sup_fea, edge_results, seg_x
        elif module == 'parallel_adapter' or module == 'separable_adapter':
            return out, share_map, para_map, deep_sup_fea
        else:
            return out
