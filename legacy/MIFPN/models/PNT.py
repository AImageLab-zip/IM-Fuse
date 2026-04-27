import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import (general_conv3d, normalization, prm_generator, prm_fusion, general_conv3d_prenorm,
                    prm_generator_laststage, region_aware_modal_fusion, fusion_postnorm,ada)
from models.blocks import nchwd2nlc2nchwd, DepthWiseConvBlock, ResBlock, GroupConvBlock, MultiMaskAttentionLayer, MultiMaskCrossBlock
from torch.nn.init import constant_, xavier_uniform_
from models.mask import mask_gen_fusion, mask_gen_skip

# from visualizer import get_local

basic_dims = 16
transformer_basic_dims = 512
mlp_dim = 4096
num_heads = 8
depth = 3
num_modals = 4
patch_size = 5
HWD = 80

class MultiCrossToken(nn.Module):
    def __init__(
            self,
            image_h=80,
            image_w=80,
            image_d=80,
            h_stride=16,
            w_stride=16,
            d_stride=16,
            num_layers=2,
            mlp_ratio=4,
            drop_rate=0.1,
            attn_drop_rate=0.0,
            interpolate_mode='trilinear',
            channel=basic_dims*16):
        super(MultiCrossToken, self).__init__()

        self.channels = channel
        self.H = image_h // h_stride
        self.W = image_w // w_stride
        self.D = image_d // d_stride
        self.interpolate_mode = interpolate_mode
        self.layers = nn.ModuleList([
            MultiMaskCrossBlock(feature_channels=self.channels,
                                      num_classes=self.channels,
                                      expand_ratio=mlp_ratio,
                                      drop_rate=drop_rate,
                                      attn_drop_rate=attn_drop_rate,
                                      ffn_feature_maps=i != num_layers - 1,
                                      ) for i in range(num_layers)])

    def forward(self, inputs, kernels, mask):
        feature_maps = inputs
        for layer in self.layers:
            kernels, feature_maps = layer(kernels, feature_maps, mask)

        return kernels

class S_Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_conv3d_prenorm, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)


    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.e1_c1 = general_conv3d(1, basic_dims, pad_type='reflect')
        self.e1_c2 = general_conv3d(basic_dims, basic_dims, pad_type='reflect')
        self.e1_c3 = general_conv3d(basic_dims, basic_dims, pad_type='reflect')

        self.e2_c1 = general_conv3d(basic_dims, basic_dims*2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv3d(basic_dims*2, basic_dims*2, pad_type='reflect')
        self.e2_c3 = general_conv3d(basic_dims*2, basic_dims*2, pad_type='reflect')

        self.e3_c1 = general_conv3d(basic_dims*2, basic_dims*4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv3d(basic_dims*4, basic_dims*4, pad_type='reflect')
        self.e3_c3 = general_conv3d(basic_dims*4, basic_dims*4, pad_type='reflect')

        self.e4_c1 = general_conv3d(basic_dims*4, basic_dims*8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv3d(basic_dims*8, basic_dims*8, pad_type='reflect')
        self.e4_c3 = general_conv3d(basic_dims*8, basic_dims*8, pad_type='reflect')

        self.e5_c1 = general_conv3d(basic_dims*8, basic_dims*16, stride=2, pad_type='reflect')
        self.e5_c2 = general_conv3d(basic_dims*16, basic_dims*16, pad_type='reflect')
        self.e5_c3 = general_conv3d(basic_dims*16, basic_dims*16, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))

        return x1, x2, x3, x4, x5

class Decoder_sep(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_sep, self).__init__()

        self.d4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d4_c1 = general_conv3d(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_c2 = general_conv3d(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_out = general_conv3d(basic_dims*8, basic_dims*8, k_size=1, padding=0, pad_type='reflect')

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4, x5):

        de_x5 = self.d4_c1(self.d4(x5))
        cat_x4 = torch.cat((de_x5, x4), dim=1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))

        de_x4 = self.d3_c1(self.d3(de_x4))
        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))

        de_x3 = self.d2_c1(self.d2(de_x3))
        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))

        de_x2 = self.d1_c1(self.d1(de_x2))
        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred

class msfs(nn.Module):
    def __init__(self, num_cls=4):
        super(msfs, self).__init__()

        self.d5_c2 = general_conv3d(basic_dims*32, basic_dims*16, pad_type='reflect')
        self.d5_out = general_conv3d(basic_dims*16, basic_dims*16, k_size=1, padding=0, pad_type='reflect')

        self.CT5 = MultiCrossToken(h_stride=16, w_stride=16, d_stride=16, channel=basic_dims*16)
        self.CT4 = MultiCrossToken(h_stride=8, w_stride=8, d_stride=8, channel=basic_dims*8)
        
        self.d4_c1 = general_conv3d(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_c2 = general_conv3d(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_out = general_conv3d(basic_dims*8, basic_dims*8, k_size=1, padding=0, pad_type='reflect')

        self.d3_c1 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)

        self.msm5 = fusion_postnorm(in_channel=basic_dims*16, num_cls=num_cls, id=125, hd =64)
        self.msm4 = fusion_postnorm(in_channel=basic_dims*8, num_cls=num_cls, id=1000, hd =64)
        self.msm3 = fusion_postnorm(in_channel=basic_dims*4, num_cls=num_cls, id=8000, hd =64)
        self.msm2 = fusion_postnorm(in_channel=basic_dims*2, num_cls=num_cls, id=64000, hd =64)
        self.msm1 = fusion_postnorm(in_channel=basic_dims*1, num_cls=num_cls, id=512000, hd =64)

        self.prm_fusion5 = prm_fusion(in_channel=basic_dims*16, num_cls=num_cls)
        self.prm_fusion4 = prm_fusion(in_channel=basic_dims*8, num_cls=num_cls)
        self.prm_fusion3 = prm_fusion(in_channel=basic_dims*4, num_cls=num_cls)
        self.prm_fusion2 = prm_fusion(in_channel=basic_dims*2, num_cls=num_cls)
        self.prm_fusion1 = prm_fusion(in_channel=basic_dims*1, num_cls=num_cls)


    def forward(self, dx1, dx2, dx3, dx4, dx5, fusion, mask):

        prm_pred5 = self.prm_fusion5(fusion)
        # de_x5 = self.CT5(dx5, fusion, mask)
        de_x5 = self.msm5(dx5, mask)
        de_x5 = torch.cat((de_x5, fusion), dim=1)
        de_x5 = self.d5_out(self.d5_c2(de_x5))
        de_x5 = self.d4_c1(self.up2(fusion))

        prm_pred4 = self.prm_fusion4(de_x5)
        # de_x4 = self.CT4(dx4, de_x5, mask)
        de_x4 = self.msm4(dx4, mask)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))

        prm_pred3 = self.prm_fusion3(de_x4)
        de_x3 = self.msm3(dx3, mask)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        prm_pred2 = self.prm_fusion2(de_x3)
        de_x2 = self.msm2(dx2, mask)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        prm_pred1 = self.prm_fusion1(de_x2)
        de_x1 = self.msm1(dx1, mask)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred, (prm_pred1, self.up2(prm_pred2), self.up4(prm_pred3), self.up8(prm_pred4), self.up16(prm_pred5))



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        x = self.net(x)
        x = (x+x.mean(dim=1, keepdim = True))*0.5
        return x



class MaskedResidual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, mask):
        y, attn = self.fn(x, mask)
        return y + x, attn


class MaskedPreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x, mask):
        x = self.norm(x)
        x, attn = self.fn(x, mask)
        return self.dropout(x), attn


class MaskedAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0, num_class=4
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_class = num_class

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    # @get_local('attn')
    def forward(self, x, mask):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        self_mask = mask_gen_fusion(B, self.num_heads, N // (self.num_class+1), self.num_class, mask).cuda(non_blocking=True)
        attn = attn.masked_fill(self_mask==0, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn



class MaskedInteration(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1, n_levels=1, n_points=4):
        super(MaskedInteration, self).__init__()
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.depth = depth
        for j in range(self.depth):
            self.cross_attention_list.append(
                MaskedResidual(
                    MaskedPreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        MaskedAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )
            self.cross_ffn_list.append(
                Residual(
                    PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )

        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.cross_ffn_list = nn.ModuleList(self.cross_ffn_list)


    def forward(self, x, mask):
        attn_list=[]
        for j in range(self.depth):
            x, attn = self.cross_attention_list[j](x, mask)
            attn_list.append(attn.detach())
            x = self.cross_ffn_list[j](x)
        return x, attn_list



class Mpvt(nn.Module):
    def __init__(self):
        super(Mpvt, self).__init__()

        self.trans_bottle = MaskedInteration(embedding_dim=basic_dims*16, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.num_cls = num_modals

    def forward(self, x, mask, fusion, pos):
        flair, t1ce, t1, t2 = x
        embed_flair = flair.flatten(2).transpose(1, 2).contiguous()
        embed_t1ce = t1ce.flatten(2).transpose(1, 2).contiguous()
        embed_t1 = t1.flatten(2).transpose(1, 2).contiguous()
        embed_t2 = t2.flatten(2).transpose(1, 2).contiguous()

        embed_cat = torch.cat((embed_flair, embed_t1ce, embed_t1, embed_t2, fusion), dim=1)
        embed_cat = embed_cat + pos
        embed_cat_trans, attn = self.trans_bottle(embed_cat, mask)
        flair_trans, t1ce_trans, t1_trans, t2_trans, fusion_trans = torch.chunk(embed_cat_trans, self.num_cls+1, dim=1)

        return flair_trans, t1ce_trans, t1_trans, t2_trans, fusion_trans, attn

class Pgn(nn.Module):
    def __init__(self):
        super(Pgn, self).__init__()
        self.trans_bottle = MaskedInteration(embedding_dim=basic_dims*16, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.num_cls = num_modals

    def forward(self, x, mask, fusion, pos):
        flair, t1ce, t1, t2 = x
        embed_flair = flair.flatten(2).transpose(1, 2).contiguous()
        embed_t1ce = t1ce.flatten(2).transpose(1, 2).contiguous()
        embed_t1 = t1.flatten(2).transpose(1, 2).contiguous()
        embed_t2 = t2.flatten(2).transpose(1, 2).contiguous()

        embed_cat = torch.cat((embed_flair, embed_t1ce, embed_t1, embed_t2, fusion), dim=1)
        embed_cat = embed_cat + pos
        embed_cat_trans, attn = self.trans_bottle(embed_cat, mask)
        flair_trans, t1ce_trans, t1_trans, t2_trans, fusion_trans = torch.chunk(embed_cat_trans, self.num_cls+1, dim=1)

        return flair_trans, t1ce_trans, t1_trans, t2_trans, fusion_trans, attn
    
class Adapter(nn.Module):
    def __init__(self):
        super(Adapter, self).__init__()
        self.share_encoder = general_conv3d(basic_dims*16, basic_dims*16, pad_type='reflect')
        self.trans_bottle = MaskedInteration(embedding_dim=basic_dims*16, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.num_cls = num_modals

    def forward(self, x, mask, fusion, pos):
        flair, t1ce, t1, t2 = x
        embed_flair = flair.flatten(2).transpose(1, 2).contiguous()
        embed_t1ce = t1ce.flatten(2).transpose(1, 2).contiguous()
        embed_t1 = t1.flatten(2).transpose(1, 2).contiguous()
        embed_t2 = t2.flatten(2).transpose(1, 2).contiguous()

        embed_cat = torch.cat((embed_flair, embed_t1ce, embed_t1, embed_t2, fusion), dim=1)
        embed_cat = embed_cat + pos
        embed_cat_trans, attn = self.trans_bottle(embed_cat, mask)
        flair_trans, t1ce_trans, t1_trans, t2_trans, fusion_trans = torch.chunk(embed_cat_trans, self.num_cls+1, dim=1)

        return flair_trans, t1ce_trans, t1_trans, t2_trans, fusion_trans, attn


class maf(nn.Module):
    def __init__(self):
        super(maf, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv3d(in_channels=basic_dims*16, out_channels=basic_dims*16, kernel_size=1, stride=1,  padding_mode='reflect')
        self.conv2 = nn.Conv3d(in_channels=basic_dims*16, out_channels=basic_dims*16, kernel_size=1, stride=1,  padding_mode='reflect')
        self.conv3 = nn.Conv3d(in_channels=basic_dims*16, out_channels=basic_dims*16, kernel_size=1, stride=1,  padding_mode='reflect')
        self.conv4 = nn.Conv3d(in_channels=basic_dims*16, out_channels=basic_dims*16, kernel_size=1, stride=1,  padding_mode='reflect')
        self.softmax = nn.Softmax(dim=1)
        self.conv5 = nn.Conv3d(in_channels=basic_dims*16, out_channels=basic_dims*8, kernel_size=1, stride=1,  padding_mode='reflect')
        self.conv6 = nn.Conv3d(in_channels=basic_dims*8, out_channels=basic_dims*4, kernel_size=1, stride=1,  padding_mode='reflect')

        self.conv7 = nn.Conv3d(in_channels=basic_dims*4, out_channels=basic_dims*2, kernel_size=1, stride=1,  padding_mode='reflect')

        self.conv8 = nn.Conv3d(in_channels=basic_dims*2, out_channels=basic_dims*1, kernel_size=1, stride=1,  padding_mode='reflect')
        self.relu = nn.ReLU()
        self.fc = nn.Linear(500, 4)
        self.conv11 = ada(basic_dims*16, basic_dims*16, k_size=1, padding=0, pad_type='reflect')
        self.conv22 = ada(basic_dims*16, basic_dims*16, k_size=1, padding=0, pad_type='reflect')

        self.conv33 = ada(basic_dims*16, basic_dims*16, k_size=1, padding=0, pad_type='reflect')

        self.conv44 = ada(basic_dims*16, basic_dims*16, k_size=1, padding=0, pad_type='reflect')




        # self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)


    def forward(self, de_x1, de_x2, de_x3, de_x4, de_x5, attn):

        flair_tra, t1ce_tra, t1_tra, t2_tra = de_x5
        flair_x4, t1ce_x4, t1_x4, t2_x4 = de_x4
        flair_x3, t1ce_x3, t1_x3, t2_x3 = de_x3
        flair_x2, t1ce_x2, t1_x2, t2_x2 = de_x2
        flair_x1, t1ce_x1, t1_x1, t2_x1 = de_x1




        # attn_0 = attn[0]
        # attn_fusion = attn_0[:, :, (patch_size**3)*4 :, :]
        # attn_flair, attn_t1ce, attn_t1, attn_t2, attn_self = torch.chunk(attn_fusion, num_modals+1, dim=-1)
        # attn_flair = self.conv11(attn)
        # attn_t1ce = self.conv22(attn) 
        # attn_t1 = self.conv33(attn) 
        # attn_t2 = self.conv44(attn) 
        attn_flair = self.conv11(flair_tra)
        attn_t1ce = self.conv22(t1ce_tra) 
        attn_t1 = self.conv33(t1_tra) 
        attn_t2 = self.conv44(t2_tra)
        # t = torch.cat((attn_flair,attn_t1ce,attn_t1,attn_t2),dim=1)
        # s = self.fc(t.reshape(flair_tra.size(0), flair_tra.size(1),4*patch_size*patch_size*patch_size))
        # s = self.relu(torch.sum(s, dim=1))
        # s = s.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
        # w = s[:,-1][0]
        # w = s[:,-1][4]

      

        attn_flair = torch.sum(attn_flair, dim=1).reshape(flair_tra.size(0), patch_size, patch_size, patch_size).unsqueeze(dim=1)
        attn_t1ce = torch.sum(attn_t1ce, dim=1).reshape(flair_tra.size(0), patch_size, patch_size, patch_size).unsqueeze(dim=1)
        attn_t1 = torch.sum(attn_t1, dim=1).reshape(flair_tra.size(0), patch_size, patch_size, patch_size).unsqueeze(dim=1)
        attn_t2 = torch.sum(attn_t2, dim=1).reshape(flair_tra.size(0), patch_size, patch_size, patch_size).unsqueeze(dim=1)

        attn_flair = self.softmax(attn_flair)
        attn_t1ce = self.softmax(attn_t1ce)
        attn_t1 = self.softmax(attn_t1)
        attn_t2 = self.softmax(attn_t2)


        dex5 = (flair_tra*(attn_flair), t1ce_tra*(attn_t1ce), t1_tra*(attn_t1), t2_tra*(attn_t2))
        # attn_flair = self.conv5(attn_flair)
        # attn_t1ce = self.conv5(attn_t1ce) 
        # attn_t1 = self.conv5(attn_t1) 
        # attn_t2 = self.conv5(attn_t2)

        attn_flair, attn_t1ce, attn_t1, attn_t2 = self.upsample(attn_flair), self.upsample(attn_t1ce), self.upsample(attn_t1), self.upsample(attn_t2)
        
        dex4 = (flair_x4*(attn_flair), t1ce_x4*(attn_t1ce), t1_x4*(attn_t1), t2_x4*(attn_t2))

        attn_flair, attn_t1ce, attn_t1, attn_t2 = self.upsample(attn_flair), self.upsample(attn_t1ce), self.upsample(attn_t1), self.upsample(attn_t2)
        # attn_flair = self.conv6(attn_flair)
        # attn_t1ce = self.conv6(attn_t1ce) 
        # attn_t1 = self.conv6(attn_t1) 
        # attn_t2 = self.conv6(attn_t2)
        dex3 = (flair_x3*(attn_flair), t1ce_x3*(attn_t1ce), t1_x3*(attn_t1), t2_x3*(attn_t2))

        attn_flair, attn_t1ce, attn_t1, attn_t2 = self.upsample(attn_flair), self.upsample(attn_t1ce), self.upsample(attn_t1), self.upsample(attn_t2)
        # attn_flair = self.conv7(attn_flair)
        # attn_t1ce = self.conv7(attn_t1ce) 
        # attn_t1 = self.conv7(attn_t1) 
        # attn_t2 = self.conv7(attn_t2)
        # dex2 = (flair_x2*(attn_flair), t1ce_x2*(attn_t1ce), t1_x2*(attn_t1), t2_x2*(attn_t2))
        dex2 = (flair_x2, t1ce_x2, t1_x2, t2_x2)


        attn_flair, attn_t1ce, attn_t1, attn_t2 = self.upsample(attn_flair), self.upsample(attn_t1ce), self.upsample(attn_t1), self.upsample(attn_t2)
        # attn_flair = self.conv8(attn_flair)
        # attn_t1ce = self.conv8(attn_t1ce) 
        # attn_t1 = self.conv8(attn_t1) 
        # attn_t2 = self.conv8(attn_t2)
        # dex1 = (flair_x1*(attn_flair), t1ce_x1*(attn_t1ce), t1_x1*(attn_t1), t2_x1*(attn_t2))
        dex1 = (flair_x1, t1ce_x1, t1_x1, t2_x1)

        return dex1, dex2, dex3, dex4, dex5

# class fusion(nn.Module):
#     def __init__(self):
#         super(fusion, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool3d(1)
#         self.max_pool = nn.AdaptiveMaxPool3d(1)
#         self.conv = nn.Conv3d(in_channels=basic_dims*32, out_channels=basic_dims*16, kernel_size=1, stride=1,  padding_mode='reflect')
#         self.conv1 = nn.Conv3d(in_channels=basic_dims*16, out_channels=basic_dims*16, kernel_size=1, stride=1,  padding_mode='reflect')
#
#         self.conv2 = nn.Conv3d(in_channels=basic_dims*16, out_channels=basic_dims*16, kernel_size=1, stride=1,  padding_mode='reflect')
#         self.relu1 = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#
#
#     def forward(self, x, t):
#         temp = torch.cat((x,t),dim=1)
#         temp = self.conv(temp)
#         avg = self.conv2(self.relu1(self.conv1(self.avg_pool(x))))
#         max = self.conv2(self.relu1(self.conv1(self.max_pool(x))))
#         out = avg + max
#         return temp*self.sigmoid(out)


class CrossAttention(nn.Module):
    def __init__(self, in_channels = basic_dims*16, basic_dims=basic_dims):
        super(CrossAttention, self).__init__()

        self.query_conv = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                                    padding_mode='reflect')
        self.key_conv = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                                  padding_mode='reflect')
        self.value_conv = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                                    padding_mode='reflect')

        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                                  padding_mode='reflect')

    def forward(self, x, t):
        batch_size, C, D, H, W = x.size()

        query = self.query_conv(x).view(batch_size, -1, D * H * W).permute(0, 2, 1)  # B x N x C'
        key = self.key_conv(t).view(batch_size, -1, D * H * W)  # B x C' x N
        value = self.value_conv(t).view(batch_size, -1, D * H * W).permute(0, 2, 1)  # B x N x C'

        attention = self.softmax(torch.bmm(query, key))  # B x N x N
        out = torch.bmm(attention, value).permute(0, 2, 1).view(batch_size, C, D, H, W)  # B x C x D x H x W

        out = self.out_conv(out)

        return out + x


class fusion(nn.Module):
    def __init__(self, in_channels = basic_dims*32, basic_dims=basic_dims):
        super(fusion, self).__init__()

        self.cross_attention = CrossAttention(in_channels = basic_dims*16, basic_dims=8)
        self.conv = nn.Conv3d(in_channels=basic_dims*32, out_channels=basic_dims*16, kernel_size=1, stride=1,
                              padding_mode='reflect')
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # Apply cross attention between x and t
        x_cross = self.cross_attention(x, t)
        t_cross = self.cross_attention(t, x)

        # Concatenate the attended features
        combined = torch.cat((x_cross, t_cross), dim=1)

        # Apply a convolution to fuse the features
        out = self.conv(combined)
        out = self.relu(out)

        return out

class Model(nn.Module):
    def __init__(self, num_cls=4):
        super(Model, self).__init__()
        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()
        self.s_encoder = Encoder()
        self.share1_encoder = general_conv3d(basic_dims*16, basic_dims*16, pad_type='reflect')
        self.share2_encoder = general_conv3d(basic_dims*16, basic_dims*16, pad_type='reflect')
        self.share3_encoder = general_conv3d(basic_dims*16, basic_dims*16, pad_type='reflect')
        self.share4_encoder = general_conv3d(basic_dims*16, basic_dims*16, pad_type='reflect')

        self.Mpvt = Mpvt()
        self.Pgn = Pgn()
        self.msfs = msfs(num_cls=num_cls)
        self.decoder_sep = Decoder_sep(num_cls=num_cls)
        self.maf = maf()
        self.f1 = fusion()
        self.f2 = fusion()
        self.f3 = fusion()
        self.f4 = fusion()

        self.pos = nn.Parameter(torch.zeros(1, (patch_size**3)*5, basic_dims*16))
        self.fusion = nn.Parameter(nn.init.normal_(torch.zeros(1, patch_size**3, basic_dims*16), mean=0.0, std=1.0))

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #

    def forward(self, x, mask):
        # extract feature from specific encoders 
        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(x[:, 0:1, :, :, :])           # (B, 16, 80, 80, 80) ... (B, 256, 5, 5, 5) 
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(x[:, 1:2, :, :, :]) 
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 3:4, :, :, :])
        # extract feature from shared encoder
        _, _, _, _, flair_s5 = self.s_encoder(x[:, 0:1, :, :, :])
        _, _, _, _, t1ce_s5 = self.s_encoder(x[:, 1:2, :, :, :])
        _, _, _, _, t1_s5 = self.s_encoder(x[:, 2:3, :, :, :])
        _, _, _, _, t2_s5 = self.s_encoder(x[:, 3:4, :, :, :])

        m_share = (flair_s5, t1ce_s5, t1_s5, t2_s5)
        B = x.size(0)
        fusion = torch.tile(self.fusion, [B, 1, 1])                                                         # (B, 125, 256)

        flair_p, t1ce_p, t1_p, t2_p, mi_prompt, _ = self.Pgn(m_share, mask, fusion, self.pos)               # (B, 125, 256)

        flair_prompt = flair_p.view(x.size(0), patch_size, patch_size, patch_size, basic_dims*16).permute(0, 4, 1, 2, 3).contiguous()       # (B, 256, 5, 5, 5)
        t1ce_prompt = t1ce_p.view(x.size(0), patch_size, patch_size, patch_size, basic_dims*16).permute(0, 4, 1, 2, 3).contiguous()
        t1_prompt = t1_p.view(x.size(0), patch_size, patch_size, patch_size, basic_dims*16).permute(0, 4, 1, 2, 3).contiguous()
        t2_prompt = t2_p.view(x.size(0), patch_size, patch_size, patch_size, basic_dims*16).permute(0, 4, 1, 2, 3).contiguous()
        mi_L =mi_prompt.view(x.size(0), patch_size, patch_size, patch_size, basic_dims*16).permute(0, 4, 1, 2, 3).contiguous()

        flair_p = self.share1_encoder(flair_prompt)
        t1ce_p =self.share2_encoder(t1ce_prompt)
        t1_p =self.share3_encoder(t1_prompt)
        t2_p =self.share4_encoder(t2_prompt)

        # flair_x5, t1ce_x5, t1_x5, t2_x5 = flair_x5 + flair_p, t1ce_x5 + t1ce_p, t1_x5 + t1_p, t2_x5 + t2_p
        flair_x5 = self.f1(flair_x5 ,flair_p)   # (B, 256, 5, 5, 5)
        t1ce_x5 = self.f2(t1ce_x5 ,t1ce_p)
        t1_x5 = self.f3(t1_x5 ,t1_p)
        t2_x5 = self.f4(t2_x5 ,t2_p)

        m_sbottle = (flair_x5, t1ce_x5, t1_x5, t2_x5)

        flair_trans, t1ce_trans, t1_trans, t2_trans, fusion_trans, _ = self.Mpvt(m_sbottle, mask, mi_prompt, self.pos)
   
        flair_tra = flair_trans.view(x.size(0), patch_size, patch_size, patch_size, basic_dims*16).permute(0, 4, 1, 2, 3).contiguous()
        t1ce_tra = t1ce_trans.view(x.size(0), patch_size, patch_size, patch_size, basic_dims*16).permute(0, 4, 1, 2, 3).contiguous()
        t1_tra = t1_trans.view(x.size(0), patch_size, patch_size, patch_size, basic_dims*16).permute(0, 4, 1, 2, 3).contiguous()
        t2_tra = t2_trans.view(x.size(0), patch_size, patch_size, patch_size, basic_dims*16).permute(0, 4, 1, 2, 3).contiguous()
        fusion_tra = fusion_trans.view(x.size(0), patch_size, patch_size, patch_size, basic_dims*16).permute(0, 4, 1, 2, 3).contiguous()

        # x5_tra = torch.stack((flair_tra, t1ce_tra, t1_tra, t2_tra), dim=1)

        de_x5 = (flair_tra, t1ce_tra, t1_tra, t2_tra)
        de_x4 = (flair_x4, t1ce_x4, t1_x4, t2_x4)
        de_x3 = (flair_x3, t1ce_x3, t1_x3, t2_x3)
        de_x2 = (flair_x2, t1ce_x2, t1_x2, t2_x2)
        de_x1 = (flair_x1, t1ce_x1, t1_x1, t2_x1)

        # de_x1, de_x2, de_x3, de_x4, de_x5 = self.weight_attention(de_x1, de_x2, de_x3, de_x4, de_x5, attn)
        de_x1, de_x2, de_x3, de_x4, de_x5 = self.maf(de_x1, de_x2, de_x3, de_x4, de_x5, fusion_tra)


        de_x3 = torch.stack(de_x3, dim=1)
        de_x2 = torch.stack(de_x2, dim=1)
        de_x1 = torch.stack(de_x1, dim=1)
        de_x4 = torch.stack(de_x4, dim=1)
        de_x5 = torch.stack(de_x5, dim=1)


        fuse_pred, prm_preds = self.msfs(de_x1, de_x2, de_x3, de_x4, de_x5, fusion_tra, mask)

        if self.is_training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4, flair_x5)
            t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5)
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4, t1_x5)
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)
            return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), prm_preds,(flair_prompt, t1ce_prompt, t1_prompt, t2_prompt, mi_L)
        return fuse_pred