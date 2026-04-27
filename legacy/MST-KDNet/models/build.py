#!/usr/bin/env python3
# encoding: utf-8
#from .unet import Unet_module, UNet3D
#from .smunetr import SMUNetr
from .MSTKDNet import MSTKDNet
#from .MSTKDNet_wo_gsm import MSTKDNet_wo_gsm

"""
def build_model(inp_shape = (160, 192, 128), inp_dim1=4, inp_dim2 = 1):
    model_full    = UNet3D(inp_shape,
                      in_channels=inp_dim1,
                      out_channels=4,
                      init_channels=16,
                      p=0.2)

    model_missing = UNet3D(inp_shape,
                      in_channels=inp_dim2,
                      out_channels=4,
                      init_channels=16,
                      p=0.2)
    return model_full, model_missing

def build_smunetr(inp_shape = (160, 192, 128), inp_dim1=4, inp_dim2 = 1):
    model_full    = SMUNetr(inp_shape,
                      in_channels=inp_dim1,
                      out_channels=4,
                      init_channels=16,
                      p=0.2, unetr=True)

    model_missing = SMUNetr(inp_shape,
                      in_channels=inp_dim2,
                      out_channels=4,
                      init_channels=16,
                      p=0.2, unetr=True)
    return model_full, model_missing
"""

def build_MSTKDNet(inp_shape = (160, 192, 128), inp_dim1=4, inp_dim2 = 1):
    model_full    = MSTKDNet(inp_shape,
                      in_channels=inp_dim1,
                      out_channels=4,
                      init_channels=16,
                      p=0.2, unetr=True)

    model_missing = MSTKDNet(inp_shape,
                      in_channels=inp_dim2,
                      out_channels=4,
                      init_channels=16,
                      p=0.2, unetr=True)
    return model_full, model_missing
"""
def build_MSTKDNet_wo_gsm(inp_shape = (160, 192, 128), inp_dim1=4, inp_dim2 = 1):
    model_full    = MSTKDNet_wo_gsm(inp_shape,
                      in_channels=inp_dim1,
                      out_channels=4,
                      init_channels=16,
                      p=0.2, unetr=True)

    model_missing = MSTKDNet_wo_gsm(inp_shape,
                      in_channels=inp_dim2,
                      out_channels=4,
                      init_channels=16,
                      p=0.2, unetr=True)
    return model_full, model_missing
"""
