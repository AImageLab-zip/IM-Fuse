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
        self.skip_flows = [dict() for k in range(len(self.skip_ind))]
        self.hidden = [self.layers[k]['n_features'] for k in range(1,len(self.layers))] 
        self.hidden = [int(k/2) for k in self.hidden] #[4, 8, 16, 32]

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

    def clip(self, input):
        # This is for clipping logvars,
        # so that variances = exp(logvars) behaves well
        # logvars: torch.clamp(input, min=-50, max=50) -> Min var: 1.929e-22, Max var: 5.185e+21
        # logvars: torch.clamp(input, min=-2.3, max=0.7) -> Min var: 0.1, Max var: 2
        output = torch.clamp(input, min=-50, max=50)
        return output
    
    def forward(self, x): 
        x = self.act1(self.conv1(x))        #(B, 8, 112, 112, 112)=first_conv
        x = self.e1_c2(self.e1_c1(x))       #(B, 8, 112, 112, 112)=block_1
        ## 1
        self.skip_flows[0]['mu'] = x[:,:self.hidden[0]]                 #(B, 4, 112, 112, 112)
        self.skip_flows[0]['logvar'] = self.clip(x[:,self.hidden[0]:])  #(B, 4, 112, 112, 112)
        x = self.d1(x)                      #(B, 8, 56, 56, 56)=downsample_1

        x = self.e2_c2(self.e2_c1(x))       #(B, 16, 56, 56, 56)=block_2
        ## 3
        self.skip_flows[1]['mu'] = x[:,:self.hidden[1]]                 #(B, 8, 56, 56, 56)
        self.skip_flows[1]['logvar'] = self.clip(x[:,self.hidden[1]:])  #(B, 8, 56, 56, 56)
        x = self.d2(x)                      #(B, 16, 28, 28, 28)=downsample_2
        
        x = self.e3_c2(self.e3_c1(x))       #(B, 32, 28, 28, 28)=block_3
        ## 5
        self.skip_flows[2]['mu'] = x[:,:self.hidden[2]]                 #(B, 16, 28, 28, 28)
        self.skip_flows[2]['logvar'] = self.clip(x[:,self.hidden[2]:])  #(B, 16, 28, 28, 28)
        x = self.d3(x)                      #(B, 32, 14, 14, 14)=downsample_3

        x = self.e4_c2(self.e4_c1(x))       #(B, 64, 14, 14, 14)=block_4
        ## 7
        self.skip_flows[3]['mu'] = x[:,:self.hidden[3]]                 #(B, 32, 14, 14, 14)
        self.skip_flows[3]['logvar'] = self.clip(x[:,self.hidden[3]:])  #(B, 32, 14, 14, 14)

        return self.skip_flows
    
class GaussianSampler(nn.Module):
    """
        This predicts the mean and logvariance parameters,
        then generates an approximate sample from the posterior.
    """

    def __init__(self):
        super(GaussianSampler, self).__init__()
        self.eps=1e-7
        self.masker = MaskModal()

    def forward(self, means, logvars, list_mod, mask, is_inference):
        # means = {'T1': mu1_T1, 'T1c': mu1_T1c, 'T2': mu1_T2, 'Flair': mu1_Flair}, 
        # logvars = {'T1': logvar1_T1, 'T1c': logvar1_T1c, 'T2': logvar1_T2, 'Flair': logvar1_Flair}}
        mu_prior = torch.zeros_like(means[list_mod[0]])                     #(B, 4, 112, 112, 112)/(B, 8, 56, 56, 56)/(B, 16, 28, 28, 28)/(B, 32, 14, 14, 14)
        log_prior = torch.zeros_like(means[list_mod[0]])

        T = self.masker(torch.stack([1 / (torch.exp(logvars[mod]) + self.eps) for mod in list_mod], dim=0), mask.permute(1, 0)) #(0-4, B, 4, 112, 112, 112)
        mu = self.masker(torch.stack([means[mod] / (torch.exp(logvars[mod]) + self.eps) for mod in list_mod], dim=0), mask.permute(1, 0))

        #filtered_T = torch.zeros_like(T)
        #filtered_mu = torch.zeros_like(mu)
        #mask_to_select = mask.permute(1, 0) #(4, B)
        #filtered_T[mask_to_select, ...] = T[mask_to_select, ...] 
        #filtered_mu[mask_to_select, ...] = mu[mask_to_select, ...]

        T = torch.cat([T, (1+log_prior).unsqueeze(0)], dim=0)   
        mu = torch.cat([mu, (mu_prior).unsqueeze(0)], dim=0)       #(2~5, B, 4, 112, 112, 112)/(5, B, 8, 56, 56, 56)/(5, B, 16, 28, 28, 28)/(5, B, 32, 14, 14, 14)
        
        posterior_means = torch.sum(mu, dim=0) / torch.sum(T, dim=0)        #(B, 4, 112, 112, 112)/(B, 8, 56, 56, 56)/(B, 16, 28, 28, 28)/(B, 32, 14, 14, 14)
        var = 1 / torch.sum(T, dim=0)                                       
        posterior_logvars = torch.log(var + self.eps)

        if is_inference:
            return posterior_means
        else:
            noise_sample = torch.randn(posterior_means.shape, device=posterior_means.device)
            #noise_sample = torch.normal(0.0, 1.0, size=posterior_means.shape).to(posterior_means.device)
            output = posterior_means + torch.exp(0.5 * posterior_logvars) * noise_sample
            return output                                                   #(B, 4, 56, 56, 56)/(B, 8, 28, 28, 28)/(B, 16, 14, 14, 14)/(B, 32, 14, 14, 14)

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
        self.d1_c1 = general_conv3d_prenorm(in_ch=6*self.ini_f, out_ch=4*self.ini_f, k_size=(3,3,3))
        self.d1_c2 = general_conv3d_prenorm(in_ch=4*self.ini_f, out_ch=2*self.ini_f, k_size=(3,3,3))

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d_prenorm(3*self.ini_f, 2*self.ini_f, k_size=(3,3,3))
        self.d2_c2 = general_conv3d_prenorm(2*self.ini_f, self.ini_f, k_size=(3,3,3))

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d_prenorm(int((3/2)*self.ini_f), self.ini_f, k_size=(3,3,3))
        self.d3_c2 = general_conv3d_prenorm(self.ini_f, int(self.ini_f/2), k_size=(3,3,3))
    
        self.conv_out = nn.Conv3d(in_channels=int(self.ini_f/2), out_channels=num_cls, kernel_size=1, bias=False, padding='same')
        self.softmax = nn.Softmax(dim=1)

    def forward(self, list_skips, use_softmax=False):
        list_skips = list_skips[::-1] #reverse order:(B, 32, 14, 14, 14)/(B, 16, 28, 28, 28)/(B, 8, 56, 56, 56)/(B, 4, 112, 112, 112)

        de_x = self.d1(list_skips[0])                       #(B, 32, 28, 28, 28)=upsampling_1
        cat_x = torch.cat((de_x, list_skips[1]), dim=1)     #(B, 48, 28, 28, 28)
        de_x = self.d1_c2(self.d1_c1(cat_x))                #(B, 16, 28, 28, 28)=block_1

        de_x = self.d2(de_x)                                #(B, 16, 56, 56, 56)=upsampling_2
        cat_x = torch.cat((de_x, list_skips[2]), dim=1)     #(B, 24, 56, 56, 56)
        de_x = self.d2_c2(self.d2_c1(cat_x))                #(B, 8, 56, 56, 56)=block_2
        
        de_x = self.d3(de_x)                                #(B, 8, 112, 112, 112)=upsampling_3
        cat_x = torch.cat((de_x, list_skips[3]), dim=1)     #(B, 12, 112, 112, 112)
        de_x = self.d3_c2(self.d3_c1(cat_x))                #(B, 4, 112, 112, 112)=block_3

        logits = self.conv_out(de_x)                        #(B, num_classes, 112, 112, 112)

        if use_softmax:
            logits = self.softmax(logits)

        return logits

class U_HVED(nn.Module):
    """
    Implementation of U-MVAE introduced [1] mixing MVAE [2] and a U-Net architecture [3]
    [1] Dorent, et al. "Hetero-Modal Variational Encoder-Decoder for
        Joint Modality Completion and Segmentation". 
        MICCAI 2019. 
    [2] Wu, et al. "Multimodal Generative Models for Scalable Weakly-Supervised Learning"
        NIPS 2018. https://arxiv.org/abs/1802.05335
    [3] Ronneberger, et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation". 
        MICCAI 2015. https://arxiv.org/abs/1505.04597
    """

    def __init__(self,
                num_classes,
                acti_func='leakyrelu'):

        super(U_HVED, self).__init__()
        self.num_classes = num_classes

        self.flair_encoder = ConvEncoder()
        self.t1ce_encoder = ConvEncoder()
        self.t1_encoder = ConvEncoder()
        self.t2_encoder = ConvEncoder()

        self.approximate_sampler = GaussianSampler()

        self.flair_decoder = ConvDecoderImg(num_cls=1)
        self.t1c_decoder = ConvDecoderImg(num_cls=1)
        self.t1_decoder = ConvDecoderImg(num_cls=1)
        self.t2_decoder = ConvDecoderImg(num_cls=1) 
        self.seg_decoder = ConvDecoderImg(num_cls=num_classes) 

        self.mod_img = MODALITIES[:4]

        #for m in self.modules():
        #    if isinstance(m, nn.Conv3d):
        #        torch.nn.init.kaiming_normal_(m.weight) 

        initialize_weights(self)

    def forward(self, x, mask, is_training=True, is_inference=False):
        # Encode the input ['T1', 'T1c', 'T2', 'Flair']
        ##[{'mu': mu1, 'logvar': logvar1}, {'mu': mu2, 'logvar': logvar2}, {'mu': mu3, 'logvar': logvar3}, {'mu': mu4, 'logvar': logvar4}]
        t1_param = self.t1_encoder(x['T1'])  
        t1c_param = self.t1ce_encoder(x['T1c'])
        t2_param = self.t2_encoder(x['T2'])
        flair_param = self.flair_encoder(x['Flair'])

        post_param = []
        '''
                post_param{
        id=0        {mu:{Flair:[B,4,112,112,112],T1c:[B,4,112,112,112],T1:[B,4,112,112,112],T2:[B,4,112,112,112]},
                    logvar:{Flair:[B,4,112,112,112],T1c:[B,4,112,112,112],T1:[B,4,112,112,112],T2:[B,4,112,112,112]}},

        id=1        {mu:{Flair:[B,8,56,56,56],T1c:[B,8,56,56,56],T1:[B,8,56,56,56],T2:[B,8,56,56,56]},
                    logvar:{Flair:[B,8,56,56,56],T1c:[B,8,56,56,56],T1:[B,8,56,56,56],T2:[B,8,56,56,56]}},

        id=2        {mu:{Flair:[B,16,28,28,28],T1c:[B,16,28,28,28],T1:[B,16,28,28,28],T2:[B,16,28,28,28]},
                    logvar:{Flair:[B,16,28,28,28],T1c:[B,16,28,28,28],T1:[B,16,28,28,28],T2:[B,16,28,28,28]}},

        id=3        {mu:{Flair:[B,32,14,14,14],T1c:[B,32,14,14,14],T1:[B,32,14,14,14],T2:[B,32,14,14,14]},
                    logvar:{Flair:[B,32,14,14,14],T1c:[B,32,14,14,14],T1:[B,32,14,14,14],T2:[B,32,14,14,14]}}             
                }
        '''
        for i in range(len(t1_param)):
            flow = {
                'mu':{'T1': t1_param[i]['mu'], 'T1c': t1c_param[i]['mu'], 'T2': t2_param[i]['mu'], 'Flair': flair_param[i]['mu']},
                'logvar': {'T1': t1_param[i]['logvar'], 'T1c': t1c_param[i]['logvar'], 'T2': t2_param[i]['logvar'], 'Flair': flair_param[i]['logvar']}
            }
            post_param.append(flow)
           
        # Sample from the posterior distribution P(latent variables|input)
        skip_flow = []
        for k in range(len(post_param)):
            sample = self.approximate_sampler(post_param[k]['mu'], post_param[k]['logvar'], self.mod_img, mask, is_inference=is_inference)
            skip_flow.append(sample)
 
        # Decode the input
        output = dict()
        if is_inference:
            return self.seg_decoder(skip_flow)
        else:
            output['T1'] = self.t1_decoder(skip_flow)
            output['T1c'] = self.t1c_decoder(skip_flow)
            output['T2'] = self.t2_decoder(skip_flow)
            output['Flair'] = self.flair_decoder(skip_flow)     #(B, 1, 112, 112, 112)
            output['seg'] = self.seg_decoder(skip_flow, use_softmax=True)          #(B, 4, 112, 112, 112)
            return output, post_param
