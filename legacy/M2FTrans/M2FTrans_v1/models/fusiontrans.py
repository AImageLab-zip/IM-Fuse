import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import (general_conv3d, normalization, prm_generator, prm_fusion,
                    prm_generator_laststage, region_aware_modal_fusion, fusion_postnorm)
from models.blocks import nchwd2nlc2nchwd, DepthWiseConvBlock, ResBlock, GroupConvBlock, MultiMaskAttentionLayer, MultiMaskCrossBlock
from torch.nn.init import constant_, xavier_uniform_
from models.mask import mask_gen_fusion

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


class Decoder_fusion(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_fusion, self).__init__()

        self.d5_c2 = general_conv3d(basic_dims*32, basic_dims*16, pad_type='reflect')
        self.d5_out = general_conv3d(basic_dims*16, basic_dims*16, k_size=1, padding=0, pad_type='reflect')

        self.CT5 = MultiCrossToken(h_stride=16, w_stride=16, d_stride=16, channel=basic_dims*16)
        self.CT4 = MultiCrossToken(h_stride=8, w_stride=8, d_stride=8, channel=basic_dims*8)
        # self.CT3 = MultiCrossToken(h_stride=4, w_stride=4, d_stride=4, channel=basic_dims*4)
        # self.CT2 = MultiCrossToken(h_stride=2, w_stride=2, d_stride=2, channel=basic_dims*2)
        # self.CT1 = MultiCrossToken(h_stride=1, w_stride=1, d_stride=1, channel=basic_dims*1)

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

        # self.RFM5 = fusion_postnorm(in_channel=basic_dims*16, num_cls=num_cls)
        # self.RFM4 = fusion_postnorm(in_channel=basic_dims*8, num_cls=num_cls)
        self.RFM3 = fusion_postnorm(in_channel=basic_dims*4, num_cls=num_cls)
        self.RFM2 = fusion_postnorm(in_channel=basic_dims*2, num_cls=num_cls)
        self.RFM1 = fusion_postnorm(in_channel=basic_dims*1, num_cls=num_cls)

        self.prm_fusion5 = prm_fusion(in_channel=basic_dims*16, num_cls=num_cls)
        self.prm_fusion4 = prm_fusion(in_channel=basic_dims*8, num_cls=num_cls)
        self.prm_fusion3 = prm_fusion(in_channel=basic_dims*4, num_cls=num_cls)
        self.prm_fusion2 = prm_fusion(in_channel=basic_dims*2, num_cls=num_cls)
        self.prm_fusion1 = prm_fusion(in_channel=basic_dims*1, num_cls=num_cls)


    def forward(self, dx1, dx2, dx3, dx4, dx5, fusion, mask):
        prm_pred5 = self.prm_fusion5(fusion)        # (B, 4, 5, 5, 5)
        de_x5 = self.CT5(dx5, fusion, mask)         # (B, 256, 5, 5, 5)
        de_x5 = torch.cat((de_x5, fusion), dim=1)   # (B, 512, 5, 5, 5)
        de_x5 = self.d5_out(self.d5_c2(de_x5))      # (B, 256, 5, 5, 5)      
        de_x5 = self.d4_c1(self.up2(de_x5))         # (B, 256, 10, 10, 10)  

        prm_pred4 = self.prm_fusion4(de_x5)         # (B, 4, 10, 10, 10)
        de_x4 = self.CT4(dx4, de_x5, mask)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))

        prm_pred3 = self.prm_fusion3(de_x4)
        de_x3 = self.RFM3(dx3, mask)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        prm_pred2 = self.prm_fusion2(de_x3)
        de_x2 = self.RFM2(dx2, mask)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        prm_pred1 = self.prm_fusion1(de_x2)
        de_x1 = self.RFM1(dx1, mask)
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
        return self.net(x)



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



class MaskedTransformer(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1, n_levels=1, n_points=4):
        super(MaskedTransformer, self).__init__()
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
        for j in range(self.depth): #3
            x, attn = self.cross_attention_list[j](x, mask)
            attn_list.append(attn.detach())
            x = self.cross_ffn_list[j](x)
        return x, attn_list 



class Bottleneck(nn.Module):
    def __init__(self):
        super(Bottleneck, self).__init__()
        self.trans_bottle = MaskedTransformer(embedding_dim=basic_dims*16, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.num_cls = num_modals

    def forward(self, x, mask, fusion, pos):
        flair, t1ce, t1, t2 = x
        embed_flair = flair.flatten(2).transpose(1, 2).contiguous() # (B, 125, 256)
        embed_t1ce = t1ce.flatten(2).transpose(1, 2).contiguous()   # (B, 125, 256)
        embed_t1 = t1.flatten(2).transpose(1, 2).contiguous()       # (B, 125, 256)
        embed_t2 = t2.flatten(2).transpose(1, 2).contiguous()       # (B, 125, 256)

        embed_cat = torch.cat((embed_flair, embed_t1ce, embed_t1, embed_t2, fusion), dim=1) # (B, 625, 256)
        embed_cat = embed_cat + pos
        embed_cat_trans, attn = self.trans_bottle(embed_cat, mask)  # (B, 625, 256)
        flair_trans, t1ce_trans, t1_trans, t2_trans, fusion_trans = torch.chunk(embed_cat_trans, self.num_cls+1, dim=1) # 5 x (B, 125, 256)

        return flair_trans, t1ce_trans, t1_trans, t2_trans, fusion_trans, attn

class Weight_Attention(nn.Module):
    def __init__(self):
        super(Weight_Attention, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, de_x1, de_x2, de_x3, de_x4, de_x5, attn):
        flair_tra, t1ce_tra, t1_tra, t2_tra = de_x5
        flair_x4, t1ce_x4, t1_x4, t2_x4 = de_x4
        flair_x3, t1ce_x3, t1_x3, t2_x3 = de_x3
        flair_x2, t1ce_x2, t1_x2, t2_x2 = de_x2
        flair_x1, t1ce_x1, t1_x1, t2_x1 = de_x1

        attn_0 = attn[0]                                        # (B, 8, 625, 625)
        attn_fusion = attn_0[:, :, (patch_size**3)*4:, :]       # (B, 8, 500:625, 625)=(B, 8, 125, 625) = attention between fusion and all modalities (itself included)
        attn_flair, attn_t1ce, attn_t1, attn_t2, attn_self = torch.chunk(attn_fusion, num_modals+1, dim=-1) # 5 x (B, 8, 125, 125) = attention between fusion and single modalities/itself

        attn_flair = torch.sum(torch.sum(attn_flair, dim=1), dim=-2).reshape(flair_tra.size(0), patch_size, patch_size, patch_size).unsqueeze(dim=1)    # (B, 1, 5, 5, 5)
        attn_t1ce = torch.sum(torch.sum(attn_t1ce, dim=1), dim=-2).reshape(flair_tra.size(0), patch_size, patch_size, patch_size).unsqueeze(dim=1)      # (B, 1, 5, 5, 5)
        attn_t1 = torch.sum(torch.sum(attn_t1, dim=1), dim=-2).reshape(flair_tra.size(0), patch_size, patch_size, patch_size).unsqueeze(dim=1)          # (B, 1, 5, 5, 5)
        attn_t2 = torch.sum(torch.sum(attn_t2, dim=1), dim=-2).reshape(flair_tra.size(0), patch_size, patch_size, patch_size).unsqueeze(dim=1)          # (B, 1, 5, 5, 5)

        dex5 = (flair_tra*(attn_flair), t1ce_tra*(attn_t1ce), t1_tra*(attn_t1), t2_tra*(attn_t2))

        attn_flair, attn_t1ce, attn_t1, attn_t2 = self.upsample(attn_flair), self.upsample(attn_t1ce), self.upsample(attn_t1), self.upsample(attn_t2)   # (B, 1, 10, 10, 10)
        dex4 = (flair_x4*(attn_flair), t1ce_x4*(attn_t1ce), t1_x4*(attn_t1), t2_x4*(attn_t2))

        attn_flair, attn_t1ce, attn_t1, attn_t2 = self.upsample(attn_flair), self.upsample(attn_t1ce), self.upsample(attn_t1), self.upsample(attn_t2)   # (B, 1, 20, 20, 20)
        dex3 = (flair_x3*(attn_flair), t1ce_x3*(attn_t1ce), t1_x3*(attn_t1), t2_x3*(attn_t2))

        attn_flair, attn_t1ce, attn_t1, attn_t2 = self.upsample(attn_flair), self.upsample(attn_t1ce), self.upsample(attn_t1), self.upsample(attn_t2)   # (B, 1, 40, 40, 40)
        dex2 = (flair_x2*(attn_flair), t1ce_x2*(attn_t1ce), t1_x2*(attn_t1), t2_x2*(attn_t2))

        attn_flair, attn_t1ce, attn_t1, attn_t2 = self.upsample(attn_flair), self.upsample(attn_t1ce), self.upsample(attn_t1), self.upsample(attn_t2)   # (B, 1, 80, 80, 80)
        dex1 = (flair_x1*(attn_flair), t1ce_x1*(attn_t1ce), t1_x1*(attn_t1), t2_x1*(attn_t2))

        return dex1, dex2, dex3, dex4, dex5



class Model(nn.Module):
    def __init__(self, num_cls=4):
        super(Model, self).__init__()
        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()
        self.Bottleneck = Bottleneck()
        self.decoder_fusion = Decoder_fusion(num_cls=num_cls)
        self.decoder_sep = Decoder_sep(num_cls=num_cls)
        self.weight_attention = Weight_Attention()

        self.pos = nn.Parameter(torch.zeros(1, (patch_size**3)*5, basic_dims*16))
        self.fusion = nn.Parameter(nn.init.normal_(torch.zeros(1, patch_size**3, basic_dims*16), mean=0.0, std=1.0))

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #

    def forward(self, x, mask):
        #extract feature from different layers
        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(x[:, 0:1, :, :, :])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(x[:, 1:2, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 3:4, :, :, :])

        x_bottle = (flair_x5, t1ce_x5, t1_x5, t2_x5) # 4 x (B, 256, 5, 5, 5)

        B = x.size(0)
        fusion = torch.tile(self.fusion, [B, 1, 1])  # (B, 125, 256)

        flair_trans, t1ce_trans, t1_trans, t2_trans, fusion_trans, attn = self.Bottleneck(x_bottle, mask, fusion, self.pos) # 5 x (B, 125, 256)

        flair_tra = flair_trans.view(x.size(0), patch_size, patch_size, patch_size, basic_dims*16).permute(0, 4, 1, 2, 3).contiguous()      # (B, 256, 5, 5, 5)
        t1ce_tra = t1ce_trans.view(x.size(0), patch_size, patch_size, patch_size, basic_dims*16).permute(0, 4, 1, 2, 3).contiguous()        # (B, 256, 5, 5, 5)
        t1_tra = t1_trans.view(x.size(0), patch_size, patch_size, patch_size, basic_dims*16).permute(0, 4, 1, 2, 3).contiguous()            # (B, 256, 5, 5, 5)
        t2_tra = t2_trans.view(x.size(0), patch_size, patch_size, patch_size, basic_dims*16).permute(0, 4, 1, 2, 3).contiguous()            # (B, 256, 5, 5, 5)
        fusion_tra = fusion_trans.view(x.size(0), patch_size, patch_size, patch_size, basic_dims*16).permute(0, 4, 1, 2, 3).contiguous()    # (B, 256, 5, 5, 5)

        # x5_tra = torch.stack((flair_tra, t1ce_tra, t1_tra, t2_tra), dim=1)

        de_x5 = (flair_tra, t1ce_tra, t1_tra, t2_tra)
        de_x4 = (flair_x4, t1ce_x4, t1_x4, t2_x4)
        de_x3 = (flair_x3, t1ce_x3, t1_x3, t2_x3)
        de_x2 = (flair_x2, t1ce_x2, t1_x2, t2_x2)
        de_x1 = (flair_x1, t1ce_x1, t1_x1, t2_x1)

        de_x1, de_x2, de_x3, de_x4, de_x5 = self.weight_attention(de_x1, de_x2, de_x3, de_x4, de_x5, attn)

        de_x3 = torch.stack(de_x3, dim=1)   # (B, 4, 64, 20, 20, 20)
        de_x2 = torch.stack(de_x2, dim=1)   # (B, 4, 32, 40, 40, 40)
        de_x1 = torch.stack(de_x1, dim=1)   # (B, 4, 16, 80, 80, 80)

        fuse_pred, prm_preds = self.decoder_fusion(de_x1, de_x2, de_x3, de_x4, de_x5, fusion_tra, mask)

        if self.is_training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4, flair_x5)
            t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5)
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4, t1_x5)
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)
            return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), prm_preds
        return fuse_pred
