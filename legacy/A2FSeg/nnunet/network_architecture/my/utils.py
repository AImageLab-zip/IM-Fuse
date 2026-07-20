from lib2to3.pgen2 import token
from tkinter.tix import InputOnly
from turtle import forward
import numpy as np
import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F

from einops import repeat




# copy from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            # nn.Sigmoid()
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class MultiModalSELayer(nn.Module):
    def __init__(self, channel, reduction):
        super(MultiModalSELayer, self).__init__()
        self.SENet = []
        for i in range(4):
            self.SENet.append(nn.Sequential(SELayer(channel, reduction)))
        
        self.SENet = nn.ModuleList(self.SENet)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, drop):
        output = []
        t = 0
        for i in range(len(drop)):
            if drop[i]:
                continue
            output.append(self.SENet[i](x[t]))
            t = t + 1
        attention_maps = self.softmax(torch.stack(output, 1))
        
        output = attention_maps[:,0] * x[0]
        for i in range(1, t):
            output += attention_maps[:,i] * x[i]
        return output

class Modality_Aware(nn.Module):
    def __init__(self, channel):
        super(Modality_Aware, self).__init__()

        self.nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}
        self.conv_kwargs['kernel_size'] = 3
        self.conv_kwargs['padding'] = 1

        self.AwareNet = []
        for i in range(4):
            self.AwareNet.append(nn.Sequential(nn.Conv3d(2*channel, channel, **self.conv_kwargs),
                                               nn.BatchNorm3d(channel, **self.norm_op_kwargs),
                                               nn.LeakyReLU(**self.nonlin_kwargs),
                                               nn.Conv3d(channel, channel, **self.conv_kwargs),))
        
        self.AwareNet = nn.ModuleList(self.AwareNet)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, drop):
        x_mean = torch.mean(x, 0)
        output = []
        t = 0
        for i in range(len(drop)):
            if drop[i]:
                continue
            output.append(self.AwareNet[i](torch.cat([x_mean, x[t]], dim=1)))
            t = t + 1
        attention_maps = self.softmax(torch.stack(output, 1))
        
        output = attention_maps[:,0] * x[0]
        for i in range(1, t):
            output += attention_maps[:,i] * x[i]
        return output
'''
def expand_to_batch(tensor, desired_size):
    tile = desired_size // tensor.shape[0]
    return repeat(tensor, 'b ... -> (b tile) ...', tile=tile)

########################################################################zirui
class MultiModalSelfAttention(nn.Module):
    def __init__(self, size, channel, in_channels, blocks=1, heads=1, dim_head=None, dropout=0):
        super(MultiModalSelfAttention, self).__init__()

        # self.attention = MultiHeadSelfAttention(dim=dim, heads=heads)
        self.softmax = nn.Softmax(dim=1)

        # tokens = (img_dim // patch_dim) ** 2
        # self.token_dim = in_channels * (patch_dim ** 2)
        # tokens = 4
        
        self.size = size
        self.channel = channel
        
        self.token_dim = in_channels * (self.size ** 3)

        self.dim = channel * (self.size ** 3)

        self.dim_linear_block = self.dim

        self.dim_head = (int(self.dim / heads)) if dim_head is None else dim_head
        self.project_patches = nn.Linear(self.token_dim, self.dim)

        self.emb_dropout = nn.Dropout(dropout)

        # self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        # self.pos_emb1D = nn.Parameter(torch.randn(tokens + 1, self.dim))

        # self.transformer = TransformerEncoder(self.dim, blocks=blocks, heads=heads, dim_head=self.dim_head, dim_linear_block=dim_linear_block, dropout=dropout)
        self.transformer = TransformerEncoder(dim=self.dim, dim_out=self.token_dim, blocks=blocks, heads=heads, dim_head=self.dim_head, dim_linear_block=self.dim_linear_block, dropout=dropout)


    def forward(self, x, mask=None):
        tokens, b, c, c1, c2, c3 = x.size()
        # print("3", x.size())
        mod_patches = []
        for i in range(tokens):
            mod_patches.append(F.interpolate(x[i], scale_factor=self.size/c1, mode='nearest').view(b, c*(self.size**3)))
        mod_patches = torch.stack(mod_patches, dim=1)
        # print("aa",mod_patches.size())
        mod_patches = self.project_patches(mod_patches) # b, c

        # mod_patches = torch.cat((expand_to_batch(self.cls_token, desired_size=b), mod_patches), dim=1)
        # mod_patches = mod_patches + self.pos_emb1D[:tokens + 1, :]
        patch_embeddings = self.emb_dropout(mod_patches)

        # print("4", patch_embeddings.size())
        # feed patch_embeddings and output of transformer. shape: [batch, tokens, dim]
        y = self.transformer(patch_embeddings, mask)


        ATT = self.softmax(y).view(b, tokens, c, self.size, self.size, self.size)
        # from IPython import embed
        # embed()
        # out = x[0] * ATT[:,0].expand_as(x[0])
        out = x[0] * F.interpolate(ATT[:,0], scale_factor=c1/self.size, mode='nearest')
        for i in range(1, tokens):
            # out += x[i] * ATT[:,i].expand_as(x[i])
            out += x[i] * F.interpolate(ATT[:,i], scale_factor=c1/self.size, mode='nearest')

        return out
###########################################################################################
class TransformerBlock(nn.Module):
    """
    Vanilla transformer block from the original paper "Attention is all you need"
    Detailed analysis: https://theaisummer.com/transformer/
    """

    # def __init__(self, dim, heads=8, dim_head=None,
    def __init__(self, dim, dim_out, heads=8, dim_head=None,
                 dim_linear_block=1024, dropout=0.1, activation=nn.GELU,
                 mhsa=None, prenorm=False):
        """
        Args:
            dim: token's vector length
            heads: number of heads
            dim_head: if none dim/heads is used
            dim_linear_block: the inner projection dim
            dropout: probability of droppping values
            mhsa: if provided you can change the vanilla self-attention block
            prenorm: if the layer norm will be applied before the mhsa or after
        """
        super().__init__()
        self.mhsa = mhsa if mhsa is not None else MultiHeadSelfAttention(dim=dim, heads=heads, dim_head=dim_head)
        self.prenorm = prenorm
        self.drop = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)

        # self.linear = nn.Sequential(
        #     nn.Linear(dim, dim_linear_block),
        #     activation(),  # nn.ReLU or nn.GELU
        #     nn.Dropout(dropout),
        #     # nn.Linear(dim_linear_block, dim),
        #     nn.Linear(dim_linear_block, dim_out),
        #     nn.Dropout(dropout)
        # )

    def forward(self, x, mask=None):
        if self.prenorm:
            y = self.drop(self.mhsa(self.norm_1(x), mask)) + x
            # out = self.linear(self.norm_2(y)) + y
        else:
            y = self.norm_1(self.drop(self.mhsa(x, mask)) + x)
            # out = self.norm_2(self.linear(y) + y)
        # return out
        return y

class TransformerEncoder(nn.Module):
    # def __init__(self, dim, blocks=6, heads=8, dim_head=None, dim_linear_block=1024, dropout=0, prenorm=False):
    def __init__(self, dim, dim_out, blocks=6, heads=8, dim_head=None, dim_linear_block=1024, dropout=0, prenorm=False):

        super().__init__()
        # self.block_list = [TransformerBlock(dim, heads, dim_head,
        self.block_list = [TransformerBlock(dim, dim_out, heads, dim_head,
                                            dim_linear_block, dropout, prenorm=prenorm) for _ in range(blocks)]
        self.layers = nn.ModuleList(self.block_list)

    def forward(self, x, mask=None):
        for layer in self.layers:
            # print(x.size())
            x = layer(x, mask)
        return x

def compute_mhsa(q, k, v, scale_factor=1, mask=None):
    # resulted shape will be: [batch, heads, tokens, tokens]
    scaled_dot_prod = torch.einsum('... i d , ... j d -> ... i j', q, k) * scale_factor

    if mask is not None:
        assert mask.shape == scaled_dot_prod.shape[2:]
        scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

    attention = torch.softmax(scaled_dot_prod, dim=-1)
    # calc result per head
    return torch.einsum('... i j , ... j d -> ... i d', attention, v)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim=8, heads=8, dim_head=None):
        """
        Implementation of multi-head attention layer of the original transformer model.
        einsum and einops.rearrange is used whenever possible
        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head. In general dim_head<dim.
            However, it may not necessary be (dim/heads)
        """
        super().__init__()

        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        self.W_0 = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5

    def forward(self, x, mask=None):
        assert x.dim() == 3
        qkv = self.to_qvk(x)  # [batch, tokens, dim*3*heads ]

        # decomposition to q,v,k and cast to tuple
        # the resulted shape before casting to tuple will be: [3, batch, heads, tokens, dim_head]
        q, k, v = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.heads))

        out = compute_mhsa(q, k, v, mask=mask, scale_factor=self.scale_factor)

        # re-compose: merge heads with dim_head
        out = rearrange(out, "b h t d -> b t (h d)")
        # Apply final linear transformation layer

        return self.W_0(out)


if __name__ == "__main__":

    # network = MultiModalSelfAttention(img_size=128, in_channels=1).cuda()
    # input = torch.zeros((4, 1, 1, 128, 128, 128)).cuda()

    # network = MultiModalSelfAttention(img_size=64, in_channels=16).cuda()
    # input = torch.zeros((4, 1, 16, 64, 64, 64)).cuda()

    network = MultiModalSelfAttention(img_size=32, in_channels=32).cuda()
    input = torch.zeros((2, 1, 32, 32, 32, 32)).cuda()

    print("1",input.size())
    output = network(input)
    print("2",output.size())
'''