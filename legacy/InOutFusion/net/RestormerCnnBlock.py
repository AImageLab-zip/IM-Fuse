import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import numbers

class Restormer_CNN_block(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(Restormer_CNN_block, self).__init__()
        self.embed=nn.Conv3d(in_dim, out_dim,kernel_size=3,stride=1, padding=1, bias=False,padding_mode="reflect")
        self.GlobalFeature = GlobalFeatureExtraction(dim=out_dim, num_heads = 8)
        self.LocalFeature = LocalFeatureExtraction(dim=out_dim)
        self.FFN=nn.Conv3d(out_dim*2, out_dim,kernel_size=3,stride=1, padding=1, bias=False,padding_mode="reflect") 
        self.norm = nn.InstanceNorm3d(out_dim)   
        
    def forward(self, x):
        x=self.embed(x)
        x1=self.GlobalFeature(x)
        x2=self.LocalFeature(x)
        out=self.FFN(torch.cat((x1,x2),1))
        return self.norm(out)
    
class GlobalFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,  
                 qkv_bias=False,):
        super(GlobalFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,out_fratures=dim,
                       ffn_expansion_factor=ffn_expansion_factor,)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class LocalFeatureExtraction(nn.Module):
    def __init__(self,
                 dim=64,
                 num_blocks=2,
                 ):
        super(LocalFeatureExtraction, self).__init__()
        self.Extraction = nn.Sequential(*[ResBlock(dim,dim) for i in range(num_blocks)])
    def forward(self, x):
        return self.Extraction(x)
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,padding_mode="reflect"),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,padding_mode="reflect"),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )
    def forward(self, x):
        out = self.conv(x)
        return out+x

class AttentionBase(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=False,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv3d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv3d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv3d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):

        b, c, h, w, n = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w n -> b head c (h w n)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w n -> b head c (h w n)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w n -> b head c (h w n)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w n) -> b (head c) h w n',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out
    
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, 
                 in_features, 
                 out_fratures,
                 ffn_expansion_factor = 2,
                 bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)

        self.project_in = nn.Conv3d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv3d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias,padding_mode="reflect")

        self.project_out = nn.Conv3d(
            hidden_features, out_fratures, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w n -> b (h w n) c')


def to_4d(x, h, w, n):
    return rearrange(x, 'b (h w n) c -> b c h w n', h=h, w=w, n=n)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w, n = x.shape[-3:]
        return to_4d(self.body(to_3d(x)), h, w, n)

class RSEncoder(nn.Module):
    def __init__(self, in_channels, feature_maps):
        super(RSEncoder, self).__init__()
        encoder_outputs = []
        channel=[feature_maps,feature_maps*2,feature_maps*4,feature_maps*8]
        self.V_en_1 = Restormer_CNN_block(in_channels, channel[0])
        self.V_en_2 = Restormer_CNN_block(channel[0], channel[1])
        self.V_en_3 = Restormer_CNN_block(channel[1], channel[2])
        self.V_en_4 = Restormer_CNN_block(channel[2], channel[3])

        self.V_down1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.V_down2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.V_down3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
    def forward(self, v):
        v_1=self.V_en_1(v)
        v_2=self.V_en_2(self.V_down1(v_1))
        v_3=self.V_en_3(self.V_down2(v_2))
        v_4=self.V_en_4(self.V_down3(v_3))
        encoder_outputs = [v_1, v_2, v_3, v_4] 
        return encoder_outputs, v_4