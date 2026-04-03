import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import torch
import math
from mamba_ssm import Mamba
from torch.amp import autocast


_basic_dims = 8  # 5 for SS
_transformer_basic_dims = 512
_mlp_dim = 4096
_num_heads = 8
_depth = 1
_num_modals = 4
_patch_size = 8
_input_patch_size = 128

import torch
import torch.nn as nn

def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

class _GeneralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(_GeneralConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x

class _GeneralConv3dPrenorm(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(_GeneralConv3dPrenorm, self).__init__()
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

class _GeneralConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(_GeneralConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class _PrmGeneratorLaststage(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=4):
        super(_PrmGeneratorLaststage, self).__init__()

        self.embedding_layer = nn.Sequential(
                            _GeneralConv3d(in_channel * 4, int(in_channel // 4), k_size=1, padding=0, stride=1),
                            _GeneralConv3d(int(in_channel // 4), int(in_channel // 4), k_size=3, padding=1, stride=1),
                            _GeneralConv3d(int(in_channel // 4), in_channel, k_size=1, padding=0, stride=1))

        self.prm_layer = nn.Sequential(
                            _GeneralConv3d(in_channel, 16, k_size=1, stride=1, padding=0),
                            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
                            nn.Softmax(dim=1))

    def forward(self, x):
        seg = self.prm_layer(self.embedding_layer(x))
        return seg

class _PrmGenerator(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=4):
        super(_PrmGenerator, self).__init__()

        self.embedding_layer = nn.Sequential(
                            _GeneralConv3d(in_channel * 4, int(in_channel // 4), k_size=1, padding=0, stride=1),
                            _GeneralConv3d(int(in_channel // 4), int(in_channel // 4), k_size=3, padding=1, stride=1),
                            _GeneralConv3d(int(in_channel // 4), in_channel, k_size=1, padding=0, stride=1))


        self.prm_layer = nn.Sequential(
                            _GeneralConv3d(in_channel * 2, 16, k_size=1, stride=1, padding=0),
                            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
                            nn.Softmax(dim=1))

    def forward(self, x1, x2):
        seg = self.prm_layer(torch.cat((x1, self.embedding_layer(x2)), dim=1))
        return seg

####modal fusion in each region
class _ModalFusion(nn.Module):
    def __init__(self, in_channel=64):
        super(_ModalFusion, self).__init__()
        self.weight_layer = nn.Sequential(
                            nn.Conv3d(4*in_channel+1, 128, 1, padding=0, bias=True),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Conv3d(128, 4, 1, padding=0, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, prm, region_name):
        B, K, C, H, W, Z = x.size()

        prm_avg = torch.mean(prm, dim=(3,4,5), keepdim=False) + 1e-7
        feat_avg = torch.mean(x, dim=(3,4,5), keepdim=False) / prm_avg

        feat_avg = feat_avg.view(B, K*C, 1, 1, 1)
        feat_avg = torch.cat((feat_avg, prm_avg[:, 0, 0, ...].view(B, 1, 1, 1, 1)), dim=1)
        weight = torch.reshape(self.weight_layer(feat_avg), (B, K, 1))
        weight = self.sigmoid(weight).view(B, K, 1, 1, 1, 1)

        ###we find directly using weighted sum still achieve competing performance
        region_feat = torch.sum(x * weight, dim=1)
        return region_feat

###fuse region feature
class _RegionFusionLaststage(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super(_RegionFusionLaststage, self).__init__()
        self.fusion_layer = nn.Sequential(
                        _GeneralConv3d(in_channel * num_cls, in_channel, k_size=1, padding=0, stride=1),
                        _GeneralConv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        _GeneralConv3d(in_channel, in_channel, k_size=1, padding=0, stride=1))

    def forward(self, x):
        B, _, _, H, W, Z = x.size()
        x = torch.reshape(x, (B, -1, H, W, Z))
        return self.fusion_layer(x)

class _RegionFusion(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super(_RegionFusion, self).__init__()
        self.fusion_layer = nn.Sequential(
                        _GeneralConv3d(in_channel * num_cls, in_channel, k_size=1, padding=0, stride=1),
                        _GeneralConv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        # general_conv3d(in_channel, in_channel, k_size=1, padding=0, stride=1)
                        )

    def forward(self, x):
        return self.fusion_layer(x)

class _FusionPrenorm(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super(_FusionPrenorm, self).__init__()
        self.fusion_layer = nn.Sequential(
                        _GeneralConv3dPrenorm(in_channel * num_cls, in_channel, k_size=1, padding=0, stride=1),
                        _GeneralConv3dPrenorm(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        _GeneralConv3dPrenorm(in_channel, in_channel, k_size=1, padding=0, stride=1))

    def forward(self, x):
        return self.fusion_layer(x)

class _RegionAwareModalFusion(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=4):
        super(_RegionAwareModalFusion, self).__init__()
        self.num_cls = num_cls

        self.modal_fusion = nn.ModuleList([_ModalFusion(in_channel=in_channel) for i in range(num_cls)])
        self.region_fusion = _RegionFusion(in_channel=in_channel, num_cls=num_cls)
        self.short_cut = nn.Sequential(
                        _GeneralConv3d(in_channel * 4, in_channel, k_size=1, padding=0, stride=1),
                        _GeneralConv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        _GeneralConv3d(in_channel, in_channel // 2, k_size=1, padding=0, stride=1))

        self.clsname_list = ['BG', 'NCR/NET', 'ED', 'ET'] ##BRATS2020 and BRATS2018
        self.clsname_list = ['BG', 'NCR', 'ED', 'NET', 'ET'] ##BRATS2015

    def forward(self, x, prm):
        B, _, H, W, Z = x.size()
        y = x.view(B, 4, -1, H, W, Z)
        B, K, C, H, W, Z = y.size()

        prm = torch.unsqueeze(prm, 2).repeat(1, 1, C, 1, 1, 1)
        ###divide modal features into different regions
        flair = y[:, 0:1, ...] * prm
        t1ce = y[:, 1:2, ...] * prm
        t1 = y[:, 2:3, ...] * prm
        t2 = y[:, 3:4, ...] * prm

        modal_feat = torch.stack((flair, t1ce, t1, t2), dim=1)
        region_feat = [modal_feat[:, :, i, :, :] for i in range(self.num_cls)]

        ###modal fusion in each region
        region_fused_feat = []
        for i in range(self.num_cls):
            region_fused_feat.append(self.modal_fusion[i](region_feat[i], prm[:, i:i+1, ...], self.clsname_list[i]))
        region_fused_feat = torch.stack(region_fused_feat, dim=1)
        '''
        region_fused_feat = torch.stack((self.modal_fusion[0](region_feat[0], prm[:, 0:1, ...], 'BG'),
                                         self.modal_fusion[1](region_feat[1], prm[:, 1:2, ...], 'NCR/NET'),
                                         self.modal_fusion[2](region_feat[2], prm[:, 2:3, ...], 'ED'),
                                         self.modal_fusion[3](region_feat[3], prm[:, 3:4, ...], 'ET')), dim=1)
        '''

        ###gain final feat with a short cut
        final_feat = torch.cat((self.region_fusion(region_fused_feat), self.short_cut(y.view(B, -1, H, W, Z))), dim=1)
        return final_feat


class _InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
class _MambaTrans(nn.Module):
    def __init__(self, channels):
        super(_MambaTrans, self).__init__()
        self.mamba = Mamba(
            d_model=channels,
            d_state=min(channels, 256),
            d_conv=4,
            expand=2,
        )
        self.norm1 = nn.LayerNorm(channels, )
        self.norm2 = nn.LayerNorm(channels, )
        self.head = nn.Linear(channels, channels)

    def forward(self, x):
        x = self.mamba(self.norm1(x)) + x
        x = self.head(self.norm2(x)) + x
        return x


class _MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        # self.norm = nn.LayerNorm(dim)
        self.mamba = _MambaTrans(dim)

    @autocast(enabled=False,device_type='cuda')
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
            # x = self.norm(x)
        x_mamba = self.mamba(x)

        return x_mamba


class _MambaFusionLayer(nn.Module):
    def __init__(self, dim, num_tokens_fused_representation=None):
        super().__init__()
        self.dim = dim
        self.num_tokens_fused_representation = num_tokens_fused_representation
        self.fused_tokens = nn.Parameter(torch.randn(1, self.num_tokens_fused_representation, dim))
        self.mamba_layer = _MambaLayer(dim)

    def forward(self, x):  # (B, 2048, 512)
        B = x.size(0)
        fused_tokens = self.fused_tokens.repeat(B, 1, 1)
        x_fused = torch.cat([x, fused_tokens], dim=1)  # (B, 2048+512, 512)
        x_mamba = self.mamba_layer(x_fused)
        x_mamba = x_mamba[:, -self.num_tokens_fused_representation:, :]  # (B, 512, 512)
        return x_mamba
 

class _MambaFusionCatLayer(nn.Module):
    def __init__(self, dim, num_tokens_fused_representation=None):
        super().__init__()
        self.dim = dim
        self.num_tokens_fused_representation = num_tokens_fused_representation
        self.fused_tokens = nn.Parameter(torch.randn(1, self.num_tokens_fused_representation, dim))
        self.mamba_layer = _MambaLayer(dim)

    def forward(self, x):  # [(B, 512, 512)]*4
        B = x[0].size(0)
        fused_tokens = self.fused_tokens.repeat(B, 1, 1)
        x = torch.stack([*x, fused_tokens], dim=2)  # (B, 512, 5, 512)
        x = x.view(B, -1, self.dim)  # (B, 2048+512, 512)
        x = self.mamba_layer(x)
        x = x[:, 4::5, :]
        return x  # (B, 512, 512)


class _Tokenize(nn.Module):
    def __init__(self, dims, num_modals=4):
        super(_Tokenize, self).__init__()
        self.dims = dims
        self.num_modals = num_modals

    def forward(self, x):
        flair_intra_x, t1ce_intra_x, t1_intra_x, t2_intra_x = torch.chunk(x, self.num_modals,
                                                                          dim=1)  # (B, 512, 8, 8, 8)
        multimodal_token_x = torch.cat(
            (flair_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.dims),  # (B, 512, 512)
             t1ce_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.dims),
             t1_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.dims),
             t2_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.dims),
             ), dim=1)
        return multimodal_token_x


class _TokenizeSep(nn.Module):
    def __init__(self, dims, num_modals=4):
        super(_TokenizeSep, self).__init__()
        self.dims = dims
        self.num_modals = num_modals

    def forward(self, x):
        flair_intra_x, t1ce_intra_x, t1_intra_x, t2_intra_x = torch.chunk(x, self.num_modals,
                                                                          dim=1)  # (B, 512, 8, 8, 8)
        return flair_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.dims), \
            t1ce_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.dims), \
            t1_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.dims), \
            t2_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, self.dims)


class _Encoder(nn.Module):
    def __init__(self):
        super(_Encoder, self).__init__()

        self.e1_c1 = nn.Conv3d(in_channels=1, out_channels=_basic_dims, kernel_size=3, stride=1, padding=1,
                               padding_mode='reflect', bias=True)
        self.e1_c2 = _GeneralConv3dPrenorm(_basic_dims, _basic_dims, pad_type='reflect')
        self.e1_c3 = _GeneralConv3dPrenorm(_basic_dims, _basic_dims, pad_type='reflect')

        self.e2_c1 = _GeneralConv3dPrenorm(_basic_dims, _basic_dims * 2, stride=2, pad_type='reflect')
        self.e2_c2 = _GeneralConv3dPrenorm(_basic_dims * 2, _basic_dims * 2, pad_type='reflect')
        self.e2_c3 = _GeneralConv3dPrenorm(_basic_dims * 2, _basic_dims * 2, pad_type='reflect')

        self.e3_c1 = _GeneralConv3dPrenorm(_basic_dims * 2, _basic_dims * 4, stride=2, pad_type='reflect')
        self.e3_c2 = _GeneralConv3dPrenorm(_basic_dims * 4, _basic_dims * 4, pad_type='reflect')
        self.e3_c3 = _GeneralConv3dPrenorm(_basic_dims * 4, _basic_dims * 4, pad_type='reflect')

        self.e4_c1 = _GeneralConv3dPrenorm(_basic_dims * 4, _basic_dims * 8, stride=2, pad_type='reflect')
        self.e4_c2 = _GeneralConv3dPrenorm(_basic_dims * 8, _basic_dims * 8, pad_type='reflect')
        self.e4_c3 = _GeneralConv3dPrenorm(_basic_dims * 8, _basic_dims * 8, pad_type='reflect')

        self.e5_c1 = _GeneralConv3dPrenorm(_basic_dims * 8, _basic_dims * 16, stride=2, pad_type='reflect')
        self.e5_c2 = _GeneralConv3dPrenorm(_basic_dims * 16, _basic_dims * 16, pad_type='reflect')
        self.e5_c3 = _GeneralConv3dPrenorm(_basic_dims * 16, _basic_dims * 16, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))  # (B, 8, 128, 128, 128)

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))  # (B, 16, 64, 64, 64)

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))  # (B, 32, 32, 32, 32)

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))  # (B, 64, 16, 16, 16)

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))  # (B, 128, 8, 8, 8)

        return x1, x2, x3, x4, x5


class _Decoder_sep(nn.Module):
    def __init__(self, num_cls=4):
        super(_Decoder_sep, self).__init__()

        self.d4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d4_c1 = _GeneralConv3dPrenorm(_basic_dims * 16, _basic_dims * 8, pad_type='reflect')
        self.d4_c2 = _GeneralConv3dPrenorm(_basic_dims * 16, _basic_dims * 8, pad_type='reflect')
        self.d4_out = _GeneralConv3dPrenorm(_basic_dims * 8, _basic_dims * 8, k_size=1, padding=0, pad_type='reflect')

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = _GeneralConv3dPrenorm(_basic_dims * 8, _basic_dims * 4, pad_type='reflect')
        self.d3_c2 = _GeneralConv3dPrenorm(_basic_dims * 8, _basic_dims * 4, pad_type='reflect')
        self.d3_out = _GeneralConv3dPrenorm(_basic_dims * 4, _basic_dims * 4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = _GeneralConv3dPrenorm(_basic_dims * 4, _basic_dims * 2, pad_type='reflect')
        self.d2_c2 = _GeneralConv3dPrenorm(_basic_dims * 4, _basic_dims * 2, pad_type='reflect')
        self.d2_out = _GeneralConv3dPrenorm(_basic_dims * 2, _basic_dims * 2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = _GeneralConv3dPrenorm(_basic_dims * 2, _basic_dims, pad_type='reflect')
        self.d1_c2 = _GeneralConv3dPrenorm(_basic_dims * 2, _basic_dims, pad_type='reflect')
        self.d1_out = _GeneralConv3dPrenorm(_basic_dims, _basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=_basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.d4_c1(self.d4(x5))  # (B, 64, 16, 16, 16)

        cat_x4 = torch.cat((de_x5, x4), dim=1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))
        de_x4 = self.d3_c1(self.d3(de_x4))  # (B, 32, 32, 32, 32)

        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))  # (B, 16, 64, 64, 64)

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))  # (B, 8, 128, 128, 128)

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)  # (B, C, 128, 128, 128)

        return pred


class _Decoder_fuse(nn.Module):
    def __init__(self, num_cls=4, mamba_skip=False):
        super(_Decoder_fuse, self).__init__()

        self.d4_c1 = _GeneralConv3dPrenorm(_basic_dims * 16, _basic_dims * 8, pad_type='reflect')
        self.d4_c2 = _GeneralConv3dPrenorm(_basic_dims * 16, _basic_dims * 8, pad_type='reflect')
        self.d4_out = _GeneralConv3dPrenorm(_basic_dims * 8, _basic_dims * 8, k_size=1, padding=0, pad_type='reflect')

        self.d3_c1 = _GeneralConv3dPrenorm(_basic_dims * 8, _basic_dims * 4, pad_type='reflect')
        self.d3_c2 = _GeneralConv3dPrenorm(_basic_dims * 8, _basic_dims * 4, pad_type='reflect')
        self.d3_out = _GeneralConv3dPrenorm(_basic_dims * 4, _basic_dims * 4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = _GeneralConv3dPrenorm(_basic_dims * 4, _basic_dims * 2, pad_type='reflect')
        self.d2_c2 = _GeneralConv3dPrenorm(_basic_dims * 4, _basic_dims * 2, pad_type='reflect')
        self.d2_out = _GeneralConv3dPrenorm(_basic_dims * 2, _basic_dims * 2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = _GeneralConv3dPrenorm(_basic_dims * 2, _basic_dims, pad_type='reflect')
        self.d1_c2 = _GeneralConv3dPrenorm(_basic_dims * 2, _basic_dims, pad_type='reflect')
        self.d1_out = _GeneralConv3dPrenorm(_basic_dims, _basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_d4 = nn.Conv3d(in_channels=_basic_dims * 16, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_d3 = nn.Conv3d(in_channels=_basic_dims * 8, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_d2 = nn.Conv3d(in_channels=_basic_dims * 4, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_d1 = nn.Conv3d(in_channels=_basic_dims * 2, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_layer = nn.Conv3d(in_channels=_basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)

        self.RFM5 = _FusionPrenorm(in_channel=_basic_dims * 16, num_cls=num_cls)
        self.RFM4 = _FusionPrenorm(in_channel=_basic_dims * 8, num_cls=1 if mamba_skip else num_cls)
        self.RFM3 = _FusionPrenorm(in_channel=_basic_dims * 4, num_cls=1 if mamba_skip else num_cls)
        self.RFM2 = _FusionPrenorm(in_channel=_basic_dims * 2, num_cls=1 if mamba_skip else num_cls)
        self.RFM1 = _FusionPrenorm(in_channel=_basic_dims * 1, num_cls=1 if mamba_skip else num_cls)
        self.mamba_skip = mamba_skip

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.RFM5(x5)  # (B, 128, 8, 8, 8)
        pred4 = self.softmax(self.seg_d4(de_x5))
        de_x5 = self.d4_c1(self.up2(de_x5))  # (B, 64, 16, 16, 16)

        de_x4 = self.RFM4(x4)  # (B, 64, 16, 16, 16)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)  # (B, 128, 16, 16, 16)
        de_x4 = self.d4_out(self.d4_c2(de_x4))  # (B, 64, 16, 16, 16)
        pred3 = self.softmax(self.seg_d3(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))  # (B, 32, 32, 32, 32)

        de_x3 = self.RFM3(x3)  # (B, 32, 32, 32, 32)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)  # (B, 64, 32, 32, 32)
        de_x3 = self.d3_out(self.d3_c2(de_x3))  # (B, 32, 32, 32, 32)
        pred2 = self.softmax(self.seg_d2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))  # (B, 16, 64, 64, 64)

        de_x2 = self.RFM2(x2)  # (B, 16, 64, 64, 64)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)  # (B, 32, 64, 64, 64)
        de_x2 = self.d2_out(self.d2_c2(de_x2))  # (B, 16, 64, 64, 64)
        pred1 = self.softmax(self.seg_d1(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))  # (B, 8, 128, 128, 128)

        de_x1 = self.RFM1(x1)  # (B, 8, 128, 128, 128)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)  # (B, 16, 128, 128, 128)
        de_x1 = self.d1_out(self.d1_c2(de_x1))  # (B, 8, 128, 128, 128)

        logits = self.seg_layer(de_x1)  # (B, 4, 128, 128, 128)
        pred = self.softmax(logits)  # (B, 4, 128, 128, 128)

        return pred, (self.up2(pred1), self.up4(pred2), self.up8(pred3), self.up16(pred4))


class _SelfAttention(nn.Module):
    def __init__(
            self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
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
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class _PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class _PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class _GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)


class _FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            _GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class _Transformer(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1, n_levels=1, n_points=4):
        super(_Transformer, self).__init__()
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.depth = depth
        for j in range(self.depth):
            self.cross_attention_list.append(
                _Residual(
                    _PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        _SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )
            self.cross_ffn_list.append(
                _Residual(
                    _PreNorm(embedding_dim, _FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )

        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.cross_ffn_list = nn.ModuleList(self.cross_ffn_list)

    def forward(self, x, pos):
        for j in range(self.depth):
            x = x + pos
            x = self.cross_attention_list[j](x)
            x = self.cross_ffn_list[j](x)
        return x


class _MaskModal(nn.Module):
    def __init__(self):
        super(_MaskModal, self).__init__()

    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        x = y.view(B, -1, H, W, Z)
        return x


class IMFuse(nn.Module):
    def __init__(self, num_cls=4, interleaved_tokenization=False, mamba_skip=False):
        super(IMFuse, self).__init__()
        self.interleaved_tokenization = interleaved_tokenization

        self.flair_encoder = _Encoder()
        self.t1ce_encoder = _Encoder()
        self.t1_encoder = _Encoder()
        self.t2_encoder = _Encoder()

        if self.interleaved_tokenization:
            TokenizerClass = _TokenizeSep
            MambaFusionLayerClass = _MambaFusionCatLayer
        else:
            TokenizerClass = _Tokenize
            MambaFusionLayerClass = _MambaFusionLayer

        ########### IntraFormer
        self.flair_encode_conv = nn.Conv3d(_basic_dims * 16, _transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t1ce_encode_conv = nn.Conv3d(_basic_dims * 16, _transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t1_encode_conv = nn.Conv3d(_basic_dims * 16, _transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t2_encode_conv = nn.Conv3d(_basic_dims * 16, _transformer_basic_dims, kernel_size=1, stride=1, padding=0)

        self.flair_decode_conv = nn.Conv3d(_transformer_basic_dims, _basic_dims * 16, kernel_size=1, stride=1, padding=0)
        self.t1ce_decode_conv = nn.Conv3d(_transformer_basic_dims, _basic_dims * 16, kernel_size=1, stride=1, padding=0)
        self.t1_decode_conv = nn.Conv3d(_transformer_basic_dims, _basic_dims * 16, kernel_size=1, stride=1, padding=0)
        self.t2_decode_conv = nn.Conv3d(_transformer_basic_dims, _basic_dims * 16, kernel_size=1, stride=1, padding=0)

        self.flair_pos = nn.Parameter(torch.zeros(1, _patch_size ** 3, _transformer_basic_dims))
        self.t1ce_pos = nn.Parameter(torch.zeros(1, _patch_size ** 3, _transformer_basic_dims))
        self.t1_pos = nn.Parameter(torch.zeros(1, _patch_size ** 3, _transformer_basic_dims))
        self.t2_pos = nn.Parameter(torch.zeros(1, _patch_size ** 3, _transformer_basic_dims))
        self.fused_pos = nn.Parameter(torch.zeros(1, _patch_size ** 3, _transformer_basic_dims))

        self.flair_transformer = _Transformer(embedding_dim=_transformer_basic_dims, depth=_depth, heads=_num_heads,
                                              mlp_dim=_mlp_dim)
        self.t1ce_transformer = _Transformer(embedding_dim=_transformer_basic_dims, depth=_depth, heads=_num_heads,
                                             mlp_dim=_mlp_dim)
        self.t1_transformer = _Transformer(embedding_dim=_transformer_basic_dims, depth=_depth, heads=_num_heads,
                                           mlp_dim=_mlp_dim)
        self.t2_transformer = _Transformer(embedding_dim=_transformer_basic_dims, depth=_depth, heads=_num_heads,
                                           mlp_dim=_mlp_dim)
        ########### IntraFormer

        ########### InterFormer
        self.mamba_fusion_layer = _MambaFusionLayer(dim=_transformer_basic_dims,
                                                    num_tokens_fused_representation=_patch_size ** 3)
        self.multimodal_transformer = _Transformer(embedding_dim=_transformer_basic_dims, depth=_depth, heads=_num_heads,
                                                   mlp_dim=_mlp_dim, n_levels=_num_modals)
        self.multimodal_decode_conv = nn.Conv3d(_transformer_basic_dims, _basic_dims * 16 * _num_modals, kernel_size=1,
                                                padding=0)
        ########### InterFormer

        self.masker = _MaskModal()

        ######## Skip Connections
        self.tokenize = nn.ModuleList([
            TokenizerClass(dims=8, num_modals=_num_modals),  # (B, 8, 128, 128, 128)->(B, 128**3, 8)
            TokenizerClass(dims=16, num_modals=_num_modals),  # (B, 16, 64, 64, 64)->(B, 64**3, 16)
            TokenizerClass(dims=32, num_modals=_num_modals),  # (B, 32, 32, 32, 32)->(B, 32**3, 32)
            TokenizerClass(dims=64, num_modals=_num_modals),  # (B, 64, 16, 16, 16)->(B, 16**3, 64)
            TokenizerClass(dims=512, num_modals=_num_modals),  # (B, 512, 8, 8, 8)->(B, 8**3, 512)
        ])
        self.mamba_fusion_layers = nn.ModuleList([
            MambaFusionLayerClass(dim=8, num_tokens_fused_representation=128 ** 3),  # (B, 128**3, 8)->(B, 128**3, 8)
            MambaFusionLayerClass(dim=16, num_tokens_fused_representation=64 ** 3),  # (B, 64**3, 16)->(B, 64**3, 16)
            MambaFusionLayerClass(dim=32, num_tokens_fused_representation=32 ** 3),  # (B, 32**3, 32)->(B, 32**3, 32)
            MambaFusionLayerClass(dim=64, num_tokens_fused_representation=16 ** 3),  # (B, 16**3, 64)->(B, 16**3, 64)
            MambaFusionLayerClass(dim=512, num_tokens_fused_representation=8 ** 3),  # (B, 8**3, 512)->(B, 8**3, 512)
        ])
        ########

        self.decoder_fuse = _Decoder_fuse(num_cls=num_cls, mamba_skip=mamba_skip)
        self.decoder_sep = _Decoder_sep(num_cls=num_cls)

        self.is_training = False
        self.mamba_skip = mamba_skip

        self.apply(_InitWeights_He(1e-2))

    def forward(self, x, mask):
        # extract feature from different layers
        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(x[:, 2:3, :, :, :])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(x[:, 0:1, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(x[:, 1:2, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 3:4, :, :, :])

        ########### IntraFormer
        flair_token_x5 = self.flair_encode_conv(flair_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1,
                                                                                                   _transformer_basic_dims)  # (B, 512, 512)
        t1ce_token_x5 = self.t1ce_encode_conv(t1ce_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1,
                                                                                                _transformer_basic_dims)
        t1_token_x5 = self.t1_encode_conv(t1_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1,
                                                                                          _transformer_basic_dims)
        t2_token_x5 = self.t2_encode_conv(t2_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1,
                                                                                          _transformer_basic_dims)

        flair_intra_token_x5 = self.flair_transformer(flair_token_x5, self.flair_pos)
        t1ce_intra_token_x5 = self.t1ce_transformer(t1ce_token_x5, self.t1ce_pos)
        t1_intra_token_x5 = self.t1_transformer(t1_token_x5, self.t1_pos)
        t2_intra_token_x5 = self.t2_transformer(t2_token_x5, self.t2_pos)

        flair_intra_x5 = flair_intra_token_x5.view(x.size(0), _patch_size, _patch_size, _patch_size,
                                                   _transformer_basic_dims).permute(0, 4, 1, 2,
                                                                                    3).contiguous()  # (B, 512, 8, 8, 8)
        t1ce_intra_x5 = t1ce_intra_token_x5.view(x.size(0), _patch_size, _patch_size, _patch_size,
                                                 _transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        t1_intra_x5 = t1_intra_token_x5.view(x.size(0), _patch_size, _patch_size, _patch_size,
                                             _transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        t2_intra_x5 = t2_intra_token_x5.view(x.size(0), _patch_size, _patch_size, _patch_size,
                                             _transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()

        if self.is_training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4, flair_x5)  # (B, C, 128, 128, 128)
            t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5)
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4, t1_x5)
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)
        ########### IntraFormer

        x1 = self.masker(torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1),
                         mask)  # Bx4xCxHWZ = (B, 32, 128, 128, 128)
        x2 = self.masker(torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1), mask)
        x3 = self.masker(torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1), mask)
        x4 = self.masker(torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1), mask)
        x5_intra = self.masker(torch.stack((flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5), dim=1),
                               mask)  # (1, 2048, 8, 8, 8)

        ########### Mamba Skip
        if self.mamba_skip:
            x1 = self.tokenize[-5](x1)  # (B, 128**3, 8)*4
            x1 = self.mamba_fusion_layers[-5](x1)  # (B, 128**3, 8)
            x1 = x1.view(x.size(0), _input_patch_size, _input_patch_size, _input_patch_size, _basic_dims).permute(0, 4, 1,
                                                                                                                  2,
                                                                                                                  3).contiguous()  # (B, 8, 128, 128, 128)

            x2 = self.tokenize[-4](x2)  # (B, 64**3, 16)*4
            x2 = self.mamba_fusion_layers[-4](x2)  # (B, 64**3, 16)
            x2 = x2.view(x.size(0), _input_patch_size // 2, _input_patch_size // 2, _input_patch_size // 2,
                         _basic_dims * 2).permute(0, 4, 1, 2, 3).contiguous()  # (B, 16, 64, 64, 64)

            x3 = self.tokenize[-3](x3)  # (B, 32**3, 32)*4
            x3 = self.mamba_fusion_layers[-3](x3)  # (B, 32**3, 32)
            x3 = x3.view(x.size(0), _input_patch_size // 4, _input_patch_size // 4, _input_patch_size // 4,
                         _basic_dims * 4).permute(0, 4, 1, 2, 3).contiguous()  # (B, 32, 32, 32, 32)

            x4 = self.tokenize[-2](x4)  # (B, 16**3, 64)*4
            x4 = self.mamba_fusion_layers[-2](x4)  # (B, 16**3, 64)
            x4 = x4.view(x.size(0), _input_patch_size // 8, _input_patch_size // 8, _input_patch_size // 8,
                         _basic_dims * 8).permute(0, 4, 1, 2, 3).contiguous()  # (B, 64, 16, 16, 16)
        #######

        ########### MambaFusion + InterFormer
        multimodal_token_x5 = self.tokenize[-1](x5_intra)  # (B, 4*8**3, 128)
        fused_multimodal = self.mamba_fusion_layers[-1](multimodal_token_x5)
        multimodal_pos = self.fused_pos.repeat(x.size(0), 1, 1)
        multimodal_inter_token_x5 = self.multimodal_transformer(fused_multimodal, multimodal_pos)
        multimodal_inter_x5 = self.multimodal_decode_conv(
            multimodal_inter_token_x5.view(multimodal_inter_token_x5.size(0), _patch_size, _patch_size, _patch_size,
                                           _transformer_basic_dims).permute(0, 4, 1, 2,
                                                                            3).contiguous())  # (B, 512, 8, 8, 8) -> (B, 512, 8, 8, 8)
        x5_inter = multimodal_inter_x5

        fuse_pred, preds = self.decoder_fuse(x1, x2, x3, x4, x5_inter)
        ########### InterFormer

        if self.is_training:
            return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), preds
        return fuse_pred
