#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 15:52:59 2019

@author: reubendo
"""
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

MODALITIES = ['T1', 'T1c', 'T2', 'Flair', 'seg']

NB_CONV = 8

# Weight initialization function
def initialize_weights(model):
    for name, param in model.named_parameters():
        if "weight" in name:
            # Apply truncated normal initialization to weights
            torch.nn.init.trunc_normal_(param, mean=0.0, std=np.sqrt(2.0 / np.prod(param.shape[:-1])))
        elif "bias" in name:
            # Initialize biases to zero
            nn.init.zeros_(param)

class MaskModal(nn.Module):
    def __init__(self):
        super(MaskModal, self).__init__()
    
    def forward(self, x, mask):
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        return y
    
class general_conv3d_prenorm(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=(3,3,3), stride=1, padding='same', act_type='leakyrelu', relufactor=0.01):
        super(general_conv3d_prenorm, self).__init__()
        self.norm = nn.InstanceNorm3d(in_ch, eps=1e-6, affine=True)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'leakyrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, bias=False)

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x

class ConvEncoder(nn.Module):
    """
    Each modality are encoded indepedently.
    """
    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.ini_f = NB_CONV
        self.layers = [
            {'name': 'conv_0', 'n_features': self.ini_f, 'kernel_size': (1,1,1)},
            {'name': 'block_1', 'n_features': self.ini_f, 'kernels': ((3,3,3), (3,3,3)), 'downsampling':True},
            {'name': 'block_2', 'n_features': 2*self.ini_f, 'kernels': ((3,3,3), (3,3,3)), 'downsampling':True},
            {'name': 'block_3', 'n_features': 4*self.ini_f, 'kernels': ((3,3,3), (3,3,3)), 'downsampling':True},
            {'name': 'block_4', 'n_features': 8*self.ini_f, 'kernels': ((3,3,3), (3,3,3)), 'downsampling':False}]

        self.skip_ind = [1, 3, 5, 7]
        self.skip_flows = [[] for k in range(len(self.skip_ind))]
        #self.hidden = [self.layers[k]['n_features'] for k in range(1,len(self.layers))] 
        #self.hidden = [int(k/2) for k in self.hidden] #[4, 8, 16, 32]

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=self.ini_f, kernel_size=(1,1,1), bias=False, padding='same')
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        self.e1_c1 = general_conv3d_prenorm(in_ch=self.ini_f, out_ch=self.ini_f, k_size=(3,3,3), act_type='leakyrelu')
        self.e1_c2 = general_conv3d_prenorm(in_ch=self.ini_f, out_ch=self.ini_f, k_size=(3,3,3), act_type='leakyrelu')
        self.d1 = nn.MaxPool3d(kernel_size=2, stride=2) 

        self.e2_c1 = general_conv3d_prenorm(in_ch=self.ini_f, out_ch=2*self.ini_f, k_size=(3,3,3), act_type='leakyrelu')
        self.e2_c2 = general_conv3d_prenorm(in_ch=2*self.ini_f, out_ch=2*self.ini_f, k_size=(3,3,3), act_type='leakyrelu')
        self.d2 = nn.MaxPool3d(kernel_size=2, stride=2) 

        self.e3_c1 = general_conv3d_prenorm(in_ch=2*self.ini_f, out_ch=4*self.ini_f, k_size=(3,3,3), act_type='leakyrelu')
        self.e3_c2 = general_conv3d_prenorm(in_ch=4*self.ini_f, out_ch=4*self.ini_f, k_size=(3,3,3), act_type='leakyrelu')
        self.d3 = nn.MaxPool3d(kernel_size=2, stride=2) 
        
        self.e4_c1 = general_conv3d_prenorm(in_ch=4*self.ini_f, out_ch=8*self.ini_f, k_size=(3,3,3), act_type='leakyrelu')
        self.e4_c2 = general_conv3d_prenorm(in_ch=8*self.ini_f, out_ch=8*self.ini_f, k_size=(3,3,3), act_type='leakyrelu')

    def forward(self, x): 
        x = self.act1(self.conv1(x))        #(B, 8, 112, 112, 112)=first_conv
        x = self.e1_c2(self.e1_c1(x))       #(B, 8, 112, 112, 112)=block_1
        ## 1
        self.skip_flows[0].append(x)        #skip_flows[0]: (B, 8, 112, 112, 112)
        x = self.d1(x)                      #(B, 8, 56, 56, 56)=downsample_1

        x = self.e2_c2(self.e2_c1(x))       #(B, 16, 56, 56, 56)=block_2
        ## 3
        self.skip_flows[1].append(x)        #skip_flows[1]: (B, 16, 56, 56, 56)
        x = self.d2(x)                      #(B, 16, 28, 28, 28)=downsample_2
        
        x = self.e3_c2(self.e3_c1(x))       #(B, 32, 28, 28, 28)=block_3
        ## 5
        self.skip_flows[2].append(x)        #skip_flows[2]: (B, 32, 28, 28, 28)
        x = self.d3(x)                      #(B, 32, 14, 14, 14)=downsample_3

        x = self.e4_c2(self.e4_c1(x))       #(B, 64, 14, 14, 14)=block_4
        ## 7
        self.skip_flows[3].append(x)        #skip_flows[3]: (B, 64, 14, 14, 14)

        return self.skip_flows
    
class HeMISAbstractionBlock(nn.Module):
    """
    Written by Thomas Varsavsky.

    Function will drop all zero columns and compute E[C] and Var[C]
    :param backend_output: backend_output
    :return: 1xC tensor where C is the number of features.
    """

    def __init__(self):
        super(HeMISAbstractionBlock, self).__init__()
        self.eps=1e-7
        self.masker = MaskModal()

    def forward(self, input_tensor, mask):
        # input_tensor = [skip_i_T1, skip_i_T1c, skip_i_T2, skip_i_Flair]
        #input_tensor = torch.stack(input_tensor, dim=0)
        #input_tensor: (4, B, 8, 112, 112, 112)/(4, B, 16, 56, 56, 56)/(4, B, 32, 28, 28, 28)/(4, B, 64, 14, 14, 14)
        
        #MaskModal
        #T = torch.zeros(4,input_tensor.shape[0],input_tensor.shape[1],input_tensor.shape[2],input_tensor.shape[3],input_tensor.shape[4]).cuda()
        #for i in range(4):
        #    if mask[0][i]: #(B, 4)
        #        T[i,...] = input_tensor[i]
        #filtered_input = torch.zeros_like(input_tensor)
        #mask_to_select = mask.permute(1, 0) #(4, B)
        #filtered_input[mask_to_select, ...] = input_tensor[mask_to_select, ...] 
        
        average_over_modalities = torch.mean(self.masker(torch.stack(input_tensor, dim=0), mask.permute(1, 0)), dim=0)
        variance_between_modalities = torch.var(self.masker(torch.stack(input_tensor, dim=0), mask.permute(1, 0)), dim=0, unbiased=False)

        abstraction_output = torch.cat([average_over_modalities, variance_between_modalities], dim=1)   
        return abstraction_output   #(B, 16, 112, 112, 112)/(B, 32, 56, 56, 56)/(B, 64, 28, 28, 28)/(B, 128, 14, 14, 14)

class ConvDecoderImg(nn.Module):
    """
    Each modality are decoded indepedently using the multi-scale hidden samples.
    """
    def __init__(self, num_cls=4):

        super(ConvDecoderImg, self).__init__()
        self.ini_f = NB_CONV
        self.num_cls = num_cls
        self.layers = [
            {'name': 'block_1', 'n_features': 4*self.ini_f, 'kernels': ((3,3,3), (3,3,3))},
            {'name': 'block_2', 'n_features': 2*self.ini_f, 'kernels': ((3,3,3), (3,3,3))},
            {'name': 'block_3', 'n_features': self.ini_f, 'kernels': ((3,3,3), (3,3,3))}]
        
        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d_prenorm(in_ch=24*self.ini_f, out_ch=4*self.ini_f, k_size=(3,3,3))
        self.d1_c2 = general_conv3d_prenorm(in_ch=4*self.ini_f, out_ch=2*self.ini_f, k_size=(3,3,3))

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d_prenorm(6*self.ini_f, 2*self.ini_f, k_size=(3,3,3))
        self.d2_c2 = general_conv3d_prenorm(2*self.ini_f, self.ini_f, k_size=(3,3,3))

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d_prenorm(3*self.ini_f, self.ini_f, k_size=(3,3,3))
        self.d3_c2 = general_conv3d_prenorm(self.ini_f, int(self.ini_f/2), k_size=(3,3,3))
    
        self.conv_out = nn.Conv3d(in_channels=int(self.ini_f/2), out_channels=num_cls, kernel_size=1, bias=False, padding='same')
        self.softmax = nn.Softmax(dim=1)

    def forward(self, list_skips):
        list_skips = list_skips[::-1] #reverse order:(B, 128, 14, 14, 14)/(B, 64, 28, 28, 28)/(B, 32, 56, 56, 56)/(B, 16, 112, 112, 112)

        de_x = self.d1(list_skips[0])                       #(B, 128, 28, 28, 28)=upsampling_1
        cat_x = torch.cat((de_x, list_skips[1]), dim=1)     #(B, 192, 28, 28, 28)
        de_x = self.d1_c2(self.d1_c1(cat_x))                #(B, 16, 28, 28, 28)=block_1

        de_x = self.d2(de_x)                                #(B, 16, 56, 56, 56)=upsampling_2
        cat_x = torch.cat((de_x, list_skips[2]), dim=1)     #(B, 48, 56, 56, 56)
        de_x = self.d2_c2(self.d2_c1(cat_x))                #(B, 8, 56, 56, 56)=block_2
        
        de_x = self.d3(de_x)                                #(B, 8, 112, 112, 112)=upsampling_3
        cat_x = torch.cat((de_x, list_skips[3]), dim=1)     #(B, 24, 112, 112, 112)
        de_x = self.d3_c2(self.d3_c1(cat_x))                #(B, 4, 112, 112, 112)=block_3

        logits = self.conv_out(de_x)                        #(B, num_classes, 112, 112, 112)
        logits = self.softmax(logits)

        return logits

class U_HeMIS(nn.Module):
    """
    Implementation of U-HeMIS introduced [1] mixing HeMIS [2] and a U-Net architecture [3]
    [1] Dorent, et al. "Hetero-Modal Variational Encoder-Decoder for
        Joint Modality Completion and Segmentation". 
        MICCAI 2019.
    [2] Havaei, et al. "HeMIS: Hetero-Modal Image Segmentation". 
        MICCAI 2016. https://arxiv.org/abs/1607.05194
    [3] Ronneberger, et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation". 
        MICCAI 2015. https://arxiv.org/abs/1505.04597
    """

    def __init__(self,
                num_classes,
                acti_func='leakyrelu'):

        super(U_HeMIS, self).__init__()
        self.num_classes = num_classes

        self.flair_encoder = ConvEncoder()
        self.t1ce_encoder = ConvEncoder()
        self.t1_encoder = ConvEncoder()
        self.t2_encoder = ConvEncoder()

        self.abstraction_op = HeMISAbstractionBlock()

        self.decoder = ConvDecoderImg(num_cls=num_classes) 

        self.mod_img = MODALITIES[:4]

        initialize_weights(self)

    def forward(self, x, mask, is_inference=False):
        # Encode the input ['T1', 'T1c', 'T2', 'Flair']
        # mod_param: [[skip_1], [skip_2], [skip_3], [skip_4]]
        t1_param = self.t1_encoder(x['T1'])  
        t1c_param = self.t1ce_encoder(x['T1c'])
        t2_param = self.t2_encoder(x['T2'])
        flair_param = self.flair_encoder(x['Flair'])

        post_param = []
        # [[skip1_T1, skip1_T1c, skip1_T2, skip1_Flair], ... ]
        for i in range(len(t1_param)):
            flow = [t1_param[i][0], t1c_param[i][0], t2_param[i][0], flair_param[i][0]]
            post_param.append(flow)
           
        # Sample from the posterior distribution P(latent variables|input)
        skip_flow = []
        for k in range(len(post_param)):
            sample = self.abstraction_op(post_param[k], mask)
            skip_flow.append(sample)
 
        # Decode the input         
        output = self.decoder(skip_flow)   #(B, 4, 112, 112, 112)
        return output 
