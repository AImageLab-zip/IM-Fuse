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


class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class RecombinationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normalization=True, kernel_size=3, net_mode='3d'):
        super(RecombinationBlock, self).__init__()
        bn = SynchronizedBatchNorm3d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bach_normalization = batch_normalization
        self.kerenl_size = kernel_size
        self.rate = 2 
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
            x = nn.ReLU6(in_place=True)(x)
            x = self.norm_conv(x)
        x = self.zoom_conv(x)
        skip_x = self.skip_conv(input)
        out = x + skip_x
        return out


def num_pool2stride_size(num_pool_per_axis):
    max_num = max(num_pool_per_axis) 
    stride_size_per_pool = list()
    for i in range(max_num): 
        unit = [1,2]
        stride_size_per_pool.append([unit[i<num_pool_per_axis[0]], unit[i<num_pool_per_axis[1]], unit[i<num_pool_per_axis[2]]])
    return stride_size_per_pool  # [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]


def norm_act(nchan, only='both', _norm=None, args=None):
    if _norm == 'adain':
        norm = AdaptiveInstanceNorm3d(nchan)
    else:
        norm = SynchronizedBatchNorm3d(nchan)
  
    if config.use_dyrelu:
        act = nn.LeakyReLU(negative_slope=1e-2,inplace=True)
    else:
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
    def __init__(self, inChans, kernel_size=3, stride=1, padding=1 ,args=None):
        super(dwise, self).__init__()
        self.conv1 = nn.Conv3d(inChans, inChans, kernel_size=kernel_size, stride=stride, padding=padding, groups=inChans)
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
    def __init__(self, nb_tasks, inChans, outChans, kernel_size=3, stride=1, padding=1, second=0 ,args=None):
        super(conv_unit, self).__init__()
        self.stride = stride
        if self.stride != 1 and self.stride != (1,1,1):
            self.conv = nn.Conv3d(inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding) # padding != 0 for stride != 2 if doing padding=SAME.
        elif self.stride == 1 or self.stride == (1,1,1):
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
                    self.adapOps = nn.ModuleList([dwise(inChans,args=args) for i in range(nb_tasks)])
                    self.pwise = pwise(inChans, outChans)
                else:
                    pass                
        self.op = nn.ModuleList([norm_act(outChans, only='norm',args=args) for i in range(nb_tasks)])

    def forward(self, x):
        task_idx = config.task_idx
        if self.stride != 1 and self.stride != (1,1,1):
            out = self.conv(x) 
            out = self.op[task_idx](out)
            return out
        elif self.stride == 1 or self.stride == (1,1,1):
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
        self.args = args
        self.op1 = conv_unit(nb_tasks, inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding, args=args)
        self.op2 = conv_unit(nb_tasks, inChans, outChans, kernel_size=(kernel_size,kernel_size,1), stride=(stride[0],stride[0],1), padding=(padding,padding,0), args=args)
        self.act1 = norm_act(outChans, only="act",args=args)
    
    def forward(self, x):
        out = self.op1(x)
        out = self.act1(out)
        return out


class DownBlock(nn.Module):  # 2 conv-unit: i.e.: Conv:[3x3,1x1,3x3,1x1] with 4 bn/act, also + residual connection
    def __init__(self, nb_tasks, inChans, outChans, kernel_size=3, stride=1, padding=1 ,args=None):
        super(DownBlock, self).__init__()
        self.args = args
        self.op1 = conv_unit(nb_tasks, inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding, args=args)
        self.act1 = norm_act(outChans, only="act",args=args)
        self.op2 = conv_unit(nb_tasks, outChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding, args=args)
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
    upsample = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True) 
    return upsample


class UnetUpsample(nn.Module):  # Upsample + 1-conv-unit (2-Conv3d)
    def __init__(self, nb_tasks, inChans, outChans, up_stride=(2), norm=None, args=None):
        super(UnetUpsample, self).__init__()
        self.args = args
        self.upsamples = nn.ModuleList(
            [Upsample3D(scale_factor=up_stride) for i in range(nb_tasks)])
        self.op = conv_unit(nb_tasks, inChans, outChans, kernel_size=3,stride=1, padding=1, args=args)
        self.act = norm_act(outChans, only='both', _norm=norm, args=args) 

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
    def __init__(self, nb_tasks, inChans, outChans, kernel_size=3, stride=1, padding=1, norm=None, args=None):
        super(UpBlock, self).__init__()
        self.args = args
        self.op1 = conv_unit(nb_tasks, inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding, args=args)
        self.act1 = norm_act(outChans, only="act", _norm=norm, args=args)
        self.op2 = conv_unit(nb_tasks, outChans, outChans, kernel_size=1, stride=1, padding=padding, args=args) # ori: padding=0
        self.act2 = norm_act(outChans, only="act", _norm=norm, args=args)
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
    def __init__(self, inChans, num_class, up_stride=(2,2,2), use_kd=None, args=None):
        super(DeepSupervision, self).__init__()
        self.op1 = nn.Sequential(
            nn.Conv3d(inChans, num_class, kernel_size=1, stride=1, padding=0),
            norm_act(num_class,args=args)) 
        self.op2 = Upsample3D(scale_factor=up_stride)
        self.re_channel = nn.Conv3d(inChans, 8, kernel_size=1, stride=1, padding=0) ## maybe no use
        self.op1_FeaDistill1 = nn.Sequential(
            nn.Conv3d(inChans, args.fea_dim, kernel_size=3, stride=1, padding=1),
            norm_act(args.fea_dim,args=args))
        self.op1_FeaDistill2 = nn.Sequential(
            nn.Conv3d(args.fea_dim, num_class, kernel_size=1, stride=1, padding=0),
            norm_act(num_class,args=args)) 
        self.args = args
        self.use_kd = use_kd

    def forward(self, x, deep_supervision):
        if self.use_kd:
            if deep_supervision is None:
                fea = self.op1_FeaDistill1(x)
                logit = self.op1_FeaDistill2(fea)
            else:
                fea = self.op1_FeaDistill1(x)
                logit = self.op1_FeaDistill2(fea)
                logit = torch.add(logit, deep_supervision)  # Add 
            out = self.op2(logit)
            return out, fea, logit
        else:
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
   
    def forward(self, x):
        task_idx = config.task_idx  # 0
        deep_supervision = None
        
        deep_sup_fea = [] # 3 feaure maps

        out = self.in_tr_list[task_idx](x)  # 1st-3x3conv, [1, 8, 128, 128, 128]
        down_list = list()

        for i in range(self.depth): # 5/6  5/6*(nn.Conv3d+down_sample)  --Encoder
            out = self.down_blocks[i](out)
                    
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
       
        out = self.out_tr_list[task_idx](out, deep_supervision)

        if module == 'parallel_adapter' or module == 'separable_adapter':
            return out, share_map, para_map, deep_sup_fea
        else:
            return out


class StyleEncoder(nn.Module):  ### split/group from the start time: (2d-conv)
    def __init__(self, inChans_list=[1], base_outChans=4, style_dim=8, in_dim_2d=128):  # No bn
        super(StyleEncoder, self).__init__()
        self.model = []
        # self.model_freq = []
        self.model += [nn.Conv2d(inChans_list[0], base_outChans, kernel_size=3, stride=1, padding=1)]
        self.model += [nn.ReLU(inplace=True)]
        dim = base_outChans
        for i in range(1):
            self.model += [nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=1, padding=1)]
            self.model += [nn.ReLU(inplace=True)]
            dim *= 2
        self.model += [nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=[2,2], padding=1)]
        self.model += [nn.ReLU(inplace=True)]
        dim *= 2
        self.model += [nn.Conv2d(dim, style_dim, kernel_size=1, stride=1, padding=0)]
        self.model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*self.model)
    
    def forward(self, x):  # parallel version: [b*4,1,128,128,128] --> [bx4x128,1,128,128]
        if x.shape[1] == 1:
            x = torch.squeeze(x,1)  # [b*4,128,128,128]
        b,d,w,h = x.shape[:4]
        x = x.permute(0,3,2,1)  ###
        x_parallel = torch.unsqueeze(torch.cat([x[:,i,...] for i in range(x.shape[1])],0), 1)  # [bx4x128,1,128,128]
        freq_fea_out = self.model(x_parallel)
        out = nn.AdaptiveAvgPool2d(1)(freq_fea_out)

        return out, freq_fea_out 


class ContentEncoder(nn.Module): 
    def __init__(self, inChans_list=[1], base_outChans=8, args=None):  # base_outChans=16
        super(ContentEncoder, self).__init__()
        self.depth = max(config.num_pool_per_axis) + 1 # 5 num_pool_per_axis firstly defined in train_xxxx.py or main.py
        stride_sizes = num_pool2stride_size(config.num_pool_per_axis)
        self.in_tr_list = nn.ModuleList([InputTransition(inChans_list[j], base_outChans) for j in range(1)]) # task-specific input layers
        self.outChans_list = list()
        self.down_blocks = nn.ModuleList() # # register modules from regular python list.
        self.down_samps = nn.ModuleList()
        self.inChans = base_outChans
        for i in range(self.depth):
            outChans = base_outChans * (2**i)
            self.outChans_list.append(outChans)
            self.down_blocks.append(DownBlock(1, self.inChans, outChans, kernel_size=3, stride=1, padding=1, args=args))
            if i != self.depth-1:
                self.down_samps.append(DownSample(1, outChans, outChans*2, kernel_size=3, stride=tuple(stride_sizes[i]), padding=1, args=args)) # padding=0
                self.inChans = outChans*2
            else:
                self.inChans = outChans
    
    def forward(self, x):     # x: [N, C, D, H, W]
        out = self.in_tr_list[0](x)  # 1st-3x3conv, [1, 8, 128, 128, 128]
        down_list = list()
        for i in range(self.depth): 
            out = self.down_blocks[i](out)
            if i != self.depth-1: 
                down_list.append(out) # will not store the deepest, so as to save memory
                out = self.down_samps[i](out) # 
        return out, down_list, self.inChans, self.outChans_list


class Decoder(nn.Module):
    def __init__(self, inChans, outChans_list, concatChan_list=[None], num_class_list=[4], norm='adain', use_distill=False, use_kd=False, args=None):  # base_outChans=16, norm=['in','adain']
        super(Decoder, self).__init__()
        nb_tasks = len(num_class_list) # 1
        self.depth = max(config.num_pool_per_axis) + 1
        stride_sizes = num_pool2stride_size(config.num_pool_per_axis)
        self.up_samps = nn.ModuleList([None] * (self.depth-1))
        self.up_blocks = nn.ModuleList([None] * (self.depth-1))
        self.dSupers = nn.ModuleList() # 1 elements if self.depth =2, or 2 elements if self.depth >= 3
        self.dSupers_bin = nn.ModuleList()
        self.use_distill = use_distill
        for i in range(self.depth-2, -1, -1):
            if i == self.depth-2 and norm == 'adain':
                _norm = 'adain'
            else:
                _norm = None
            self.up_samps[i] = UnetUpsample(nb_tasks, inChans, outChans_list[i], up_stride=stride_sizes[i][0], norm=_norm, args=args)
            
            self.up_blocks[i] = UpBlock(nb_tasks, outChans_list[i]+concatChan_list[i], outChans_list[i], kernel_size=3, stride=1, padding=1, norm=_norm, args=args) 
            
            if config.deep_supervision and i <= (self.depth-3) and i > 0:
                self.dSupers.append(nn.ModuleList([DeepSupervision(outChans_list[i], num_class_list[j], up_stride=tuple(stride_sizes[i-1]),use_kd=use_kd,args=args) for j in range(nb_tasks)]))
                self.dSupers_bin.append(nn.ModuleList([DeepSupervision(outChans_list[i], 1, up_stride=tuple(stride_sizes[i-1]),use_kd=use_kd,args=args) for j in range(nb_tasks)]))
            inChans = outChans_list[i]
        self.out_tr_list_all = nn.ModuleList([OutputTransition(inChans, num_class_list[j]) for j in range(nb_tasks)])
        self.out_tr_list_binary = nn.ModuleList([OutputTransition(inChans, 1)])
        self.use_kd = use_kd
    
    def forward(self, x, down_list, is_binary=False):
        # x: [N, C, D, H, W]
        task_idx = config.task_idx
        deep_supervision = None
        deep_sup_fea = []
        distill_kd_fea = []
        idx = 0
        out = x
        
        for i in range(self.depth-2, -1, -1):  # i=[(4,) 3,2,1,0]  4/5 * (nn.Conv3d+up_sample+concat) --Decoder
            out, share_map, para_map = self.up_samps[i](out)  # Conduct the true upsample
            up_x = out
            out = torch.cat((out, down_list[i]), dim=1)
            out = self.up_blocks[i](out, up_x) 
            if config.deep_supervision and i <= (self.depth-3) and i > 0: # On 3 or 4-level, expect the final-level (and smallest 4*4 level)
                if self.use_kd:
                    if is_binary:
                        deep_supervision, _distill_fea, _logit = self.dSupers_bin[idx][task_idx](out, deep_supervision)
                    else:
                        deep_supervision, _distill_fea, _logit = self.dSupers[idx][task_idx](out, deep_supervision)
                    _deep_sup_fea = _logit
                    distill_kd_fea.append(_distill_fea)
                else:
                    if is_binary:
                        deep_supervision, _deep_sup_fea = self.dSupers_bin[idx][task_idx](out, deep_supervision)
                    else:
                        deep_supervision, _deep_sup_fea = self.dSupers[idx][task_idx](out, deep_supervision)
                deep_sup_fea.append(_deep_sup_fea)
                idx += 1
            if (not config.deep_supervision):
                deep_sup_fea = []
    
        if is_binary:
            out = self.out_tr_list_binary[0](out, None)
        else:
            out = self.out_tr_list_all[task_idx](out, deep_supervision)
        if self.use_distill:
            distill_fea = out
            if self.use_kd:
                return out, deep_sup_fea, distill_kd_fea, distill_fea
            else:
                return out, deep_sup_fea, distill_fea
        else:
            return out, deep_sup_fea


class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim=4, dim=2, decode_indim=16, style_dim=8, mlp_dim=256, num_class_list=[1], norm='adain', in_dim_2d=128, auxdec_dim=1, modal_num=4, args=None): #in_dim_2d=128):
        super(AdaINGen, self).__init__()
        # style encoder
        self.enc_style = StyleEncoder(inChans_list=[input_dim], base_outChans=dim, style_dim=style_dim, in_dim_2d=in_dim_2d)
        
        # content encoder
        self.enc_content = ContentEncoder(inChans_list=[input_dim*modal_num], base_outChans=dim, args=args)  # out, down_list, self.inChans, self.outChans_list
        _outChans_list = [dim*pow(2,i) for i in range(1+max(config.num_pool_per_axis))] # [8, 16, 32, 64] ##
        
        _outChans_list_small = [auxdec_dim*pow(2,i) for i in range(1+max(config.num_pool_per_axis))]
        self.dec = Decoder(decode_indim, _outChans_list_small, _outChans_list, num_class_list=num_class_list, norm=norm,args=args) # [8, 16, 32, 64]

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ='relu')
        self.args = args
    
    def forward(self, images, x):
        # reconstruct an image
        content, style_fake, enc_list = self.encode(images, x)
        images_recon = self.decode(content, style_fake, enc_list)
        return images_recon
    
    def encode(self, images, x):
        # encode an image to its content and style codes
        t=time.time()
        style_fake, freq_fea_out = self.enc_style(images)
        content, enc_list = self.enc_content(x)[:2]
        if self.args.use_freq_map == True:
            style_fea_map = freq_fea_out
        else:
            style_fea_map = torch.Tensor([0.0]).cuda()
        return content, style_fake, style_fea_map, enc_list
    
    def decode(self, content, style, enc_list):
        # decode content and style codes to an image
        if style != None:
            adain_params = self.mlp(style)
            self.assign_adain_params(adain_params, self.dec)
        images, deep_sup_fea = self.dec(content, enc_list)[:2]
        return images, deep_sup_fea
    
    def assign_adain_params(self, adain_params, model):
        '''
        Adopt AdaIN layer to fuse the style-aware information and feature maps
        assign the adain_params to the AdaIN layers in model
        '''
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm3d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]
    
    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm3d":
                num_adain_params += 2*m.num_features
        return num_adain_params


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)
    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)
        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'none':
            self.activation = None
    
    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class AdaptiveInstanceNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm3d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])
    
    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'





class Target_transform(nn.Module):  #  n_res x (conv+bn+relu) with residual + (conv+bn+relu)
    def __init__(self, n_res, dim, output_dim, res_norm='adain', activ='LeakyReLU', args=None):
        super(Target_transform, self).__init__()
        self.model = []
        # AdaIN residual blocks
        self.model += [nn.Conv3d(dim, dim//4, 1, 1, bias=False, padding=0)]
        dim = dim//4
        self.model += [ResBlocks(n_res, dim, res_norm, activ, args=args)]
        self.model += [Conv3dBlock(dim, output_dim, 3, 1, 1, norm='none', activation='LeakyReLU', args=args)]
        self.model = nn.Sequential(*self.model)
    
    def forward(self, x):
        return self.model(x)


class ResBlocks(nn.Module):  # num_blocks x (conv+bn+relu) with residual
    def __init__(self, num_blocks, dim, norm='in', activation='relu', args=None):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, args=args)]
        self.model = nn.Sequential(*self.model)
    
    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):  # 2x(conv+bn+relu) with residual
    def __init__(self, dim, norm='in', activation='relu', args=None):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv3dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, args=args)]
        model += [Conv3dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', args=args)]
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv3dBlock(nn.Module):  # 1x(conv+bn+relu)
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', args=None):
        super(Conv3dBlock, self).__init__()
        self.args = args
        self.use_bias = True
        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = SynchronizedBatchNorm3d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm3d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm3d(norm_dim)
        elif norm == 'none' :
            self.norm = None
        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=1e-2,inplace=True)
        elif activation == 'none':
            self.activation = None
        # initialize convolution
        self.conv = nn.Conv3d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias,padding=1)
        self.conv_1x3 = nn.Conv3d(input_dim, output_dim, kernel_size=(kernel_size,kernel_size,1), stride=(stride,stride,1), bias=self.use_bias,padding=(padding,padding,0))
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x