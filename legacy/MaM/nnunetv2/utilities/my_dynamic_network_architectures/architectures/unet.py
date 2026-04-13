from typing import Union, Type, List, Tuple
import sys
import os
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'building_blocks')))
# 假设FolderA和FolderB都在同一级目录下
parent_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件的绝对路径，然后获取其父目录
folder_b_path = os.path.join(parent_dir, '../building_blocks')  # 构建FolderB的路径
sys.path.append(folder_b_path)  # 添加FolderB的路径到sys.path
folder_c_path = os.path.join(parent_dir, '../initialization')  # 构建FolderB的路径
sys.path.append(folder_c_path)  # 添加FolderB的路径到sys.path

import torch
from helper import convert_conv_op_to_dim
from plain_conv_encoder import PlainConvEncoder, TransformerConvEncoder
from residual import BasicBlockD, BottleneckD
from residual_encoders import ResidualEncoder
from unet_decoder import UNetDecoder
from unet_residual_decoder import UNetResDecoder
from weight_init import InitWeights_He
from weight_init import init_last_bn_before_add_to_0
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from itertools import chain, combinations
import numpy as np
import random
from einops import rearrange, repeat
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    """
    Args:
        dim (int): dimensionality of the tokens/features
        depth (int): number of transformer blocks
        heads (int): number of attention heads
        dim_head (int): dimension of each attention head
        mlp_dim (int): hidden dimension in the feed-forward network
        dropout (float): dropout rate
        max_seq_len (int): maximum sequence length for positional embeddings
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., max_seq_len=256):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, dim)
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):

        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n] # added positional embeddings
        
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)

        return self.norm(x)

class MultimodalTransformerConvUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.n_stages = n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
                                                                
        self.modality_specific_encoders = nn.ModuleList()
        modality_specific_features_per_stage = [x // input_channels for x in features_per_stage]
        #print(f"modality_specific_features_per_stage: {modality_specific_features_per_stage}")
        self.modality_specific_features_per_stage = modality_specific_features_per_stage
        for i in range(input_channels):
            self.modality_specific_encoders.append(TransformerConvEncoder(1, n_stages, modality_specific_features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first))  
                                                         
        self.encoder = TransformerConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                         n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                         dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                         nonlin_first=nonlin_first)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

    def forward(self, x):
        batch_size, _, depth, height, width = x.size()
        modality_specific_inputs = torch.chunk(x, self.input_channels, dim=1)
        #print ('shape of modality specific input:', modality_specific_inputs.shape)
        combined_skips = [[] for _ in range(self.n_stages)]  # 基于stages数量初始化
        attn_maps = []
        # 计算每个stage的空间尺寸（假设每个stage深度、高度和宽度都下采样两倍）
        spatial_sizes = [(depth // (2 ** stage), height // (2 ** stage), width // (2 ** stage)) for stage in range(self.n_stages)]

        # 处理每个模态
        for i, modality_input in enumerate(modality_specific_inputs):
            if torch.all(modality_input == 0):  # 检查模态是否缺失
                # 生成与该模态encoder输出相匹配的全0特征图
                #print('yes')
                modality_skips = [torch.zeros(batch_size, self.modality_specific_features_per_stage[stage], *spatial_sizes[stage]).to(modality_input.device) for stage in range(self.n_stages)]
            else:
                #处理并获取该模态的skip features
                #modality_skips, modality_attn_maps = self.modality_specific_encoders[i](modality_input)
                #attn_maps.append(modality_attn_maps)
                modality_skips = self.modality_specific_encoders[i](modality_input)
                for i in range(len(modality_skips)):
                    print(f"modality_skips[{i}]: {modality_skips[0][i].shape}")
            # 将当前模态的skips添加到对应分辨率的combined_skips中
            for stage_idx, skip in enumerate(modality_skips):
                if len(combined_skips[stage_idx]) == 0:
                    combined_skips[stage_idx] = skip 
                    
                else:
                    combined_skips[stage_idx] = torch.cat((combined_skips[stage_idx], skip), dim=1)
        
        #for i in range(len(combined_skips)):
        #    print(f"combined_skips[{i}]: {combined_skips[i].shape}")
        # 接下来你可以通过decoder处理combined_skips
        return self.decoder(combined_skips)

class PlainConvUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

    def forward(self, x):
        print('waht  ttsdgsgf')
        skips = self.encoder(x)
        for i in range(len(skips)):
            print(f"skips[{i}]: {skips[i].shape}")
        return self.decoder(skips), skips[-1]

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)

class ResidualEncoderUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None
                 ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)

class ResidualUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None
                 ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)

class PlainConvUNetTest(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

    def forward(self, x):
        skips = self.encoder(x)
        #print("skips", skips[0].shape)
        print('all skips', [s.shape for s in skips])
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)

class MultimodalRecon(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.n_stages = n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
                                                                
        self.modality_specific_encoders = nn.ModuleList()
        modality_specific_features_per_stage = [x // input_channels for x in features_per_stage] #[8, 16, 32, 64, 128, 128]
        #print(f"modality_specific_features_per_stage: {modality_specific_features_per_stage}")
        self.modality_specific_features_per_stage = modality_specific_features_per_stage
        for i in range(input_channels):
            self.modality_specific_encoders.append(PlainConvEncoder(1, n_stages, modality_specific_features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first))  
        #print('(modality_specific_features_per_stage[-1]:', modality_specific_features_per_stage[-1])
        # Added learnable tokens for each modality to replace missing bottleneck_feature_token
        # Out final bottleneck stage has 128 channels and a spatial resolution of 4x4x4 (from your print statements).
        # We'll define learnable tokens for each modality: Shape => (num_modalities, 128, 4, 4, 4)
        self.learned_bottleneck_tokens = nn.Parameter(
            torch.randn(input_channels,
                        modality_specific_features_per_stage[-1],
                        4, 4, 4)
        )
        self.feature_reconstruction = Transformer(128, 1, 4, 32, 256)
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                         n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                         dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                         nonlin_first=nonlin_first)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

    def forward(self, x, selected_mask):
        """
               Args:
                   x: (B, input_channels, D, H, W)
                   selected_mask: List[bool], length == input_channels
                                  True => modality present
                                  False => modality missing
               Returns:
                   decoder_output, final_bottleneck, ground_truth_bottleneck
        """
        batch_size, _, depth, height, width = x.size()
        # Split the input into separate modalities
        modality_specific_inputs = torch.chunk(x, self.input_channels, dim=1)
        combined_skips = [[] for _ in range(self.n_stages)]  # 基于stages数量初始化
        #selected_mask = selected_mask.cpu()
        #selected_mask = selected_mask.tolist()

        # For each stage, compute the expected spatial size (assuming each stage down-samples by factor 2 in each dim)
        spatial_sizes = [(depth // (2 ** stage), height // (2 ** stage), width // (2 ** stage)) for stage in range(self.n_stages)] #[(128, 128, 128), (64, 64, 64), ..., (4, 4, 4)]

        # Run each modality through its own encoder
        for i, modality_input in enumerate(modality_specific_inputs):
            # if torch.all(modality_input == 0):  # 检查模态是否缺失
            #     # 生成与该模态encoder输出相匹配的全0特征
            #     #print('yes')
            #     modality_skips = [torch.zeros(batch_size, self.modality_specific_features_per_stage[stage], *spatial_sizes[stage]).to(modality_input.device) for stage in range(self.n_stages)]
            # else:
            #     #处理并获取该模态的skip features
            modality_skips = self.modality_specific_encoders[i](modality_input)
                
            # 将当前模态的skips添加到对应分辨率的combined_skips中
            for stage_idx, skip in enumerate(modality_skips):
                skip = skip.unsqueeze(1) # shape: (B, C, D, H, W) => (B, 1, C, D, H, W)
                #print('skip:',skip.shape)
                if len(combined_skips[stage_idx]) == 0:
                    combined_skips[stage_idx] = skip
                else:
                    combined_skips[stage_idx] = torch.cat((combined_skips[stage_idx], skip), dim=1)
        
        
        Reconstructed_GT = combined_skips[-1].detach() # shape: (B, input_channels, C, D, H, W) => (1, 4, 128, 4, 4, 4)
        '''for i in range(len(combined_skips)):
            for j, mask in enumerate(selected_mask):
                if not mask:
                    combined_skips[i][:, j, :, :, :, :] = 0
            # combined_skips[i] = combined_skips[i].contiguous().reshape(combined_skips[i].shape[0], -1, combined_skips[i].shape[-3], 
            #                                                            combined_skips[i].shape[-2], combined_skips[i].shape[-1])
        '''

        # Insert learned tokens for missing modalities
        bottleneck_feature = combined_skips[-1].clone()
        for j, modality_present in enumerate(selected_mask):
            if not modality_present:
                # Replace entire feature map for missing modality j with the learned token for that modality
                # learned_bottleneck_tokens[j] => shape: (C, 4, 4, 4). Broadcast sul batch
                bottleneck_feature[:, j] = self.learned_bottleneck_tokens[j]

        #  Flatten to (B, N, dim)
        bottleneck_feature_token = rearrange(bottleneck_feature, 'b m c h w d -> b (m h w d) c') #in our case (1,256,128)
        bottleneck_Rconstructed_Feature_Token = self.feature_reconstruction(bottleneck_feature_token)
        bottleneck_Rconstructed_Feature = rearrange(bottleneck_Rconstructed_Feature_Token, 'b (m h w d) c -> b m c h w d', m=self.input_channels, h=4, w=4, d=4)
        #print('shape:', bottleneck_Rconstructed_Feature.shape)
        # keep real features for present modalities, use reconstructed for missing modalities.
        for j, modality_present in enumerate(selected_mask):
            if not modality_present:
                # For missing modalities, replace with reconstructed else keep the original encoder feature
                bottleneck_feature[:, j] = bottleneck_Rconstructed_Feature[:, j]

        # Update the last skip level with this merged version
        combined_skips[-1] = bottleneck_feature

        for i in range(len(combined_skips) - 1):
            for j, mask in enumerate(selected_mask):
                if not mask:
                    combined_skips[i][:, j, :, :, :, :] = 0
            combined_skips[i] = combined_skips[i].contiguous().reshape(combined_skips[i].shape[0], -1, combined_skips[i].shape[-3], 
                                                                        combined_skips[i].shape[-2], combined_skips[i].shape[-1])
        combined_skips[-1] = combined_skips[-1].contiguous().reshape(combined_skips[-1].shape[0], -1,
                                                                   combined_skips[-1].shape[-3],
                                                                   combined_skips[-1].shape[-2],
                                                                   combined_skips[-1].shape[-1])
        #combined_skips[-1] = bottleneck_Rconstructed_Feature
        #for i in range(len(combined_skips)):
        #    print(f"combined_skips[{i}]: {combined_skips[i].shape}")
        # 接下来你可以通过decoder处理combined_skips
        return self.decoder(combined_skips), combined_skips[-1], Reconstructed_GT

class MultimodalReconBase(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.n_stages = n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
                                                                
        self.modality_specific_encoders = nn.ModuleList()
        modality_specific_features_per_stage = [x // input_channels for x in features_per_stage]
        #print(f"modality_specific_features_per_stage: {modality_specific_features_per_stage}")
        self.modality_specific_features_per_stage = modality_specific_features_per_stage
        for i in range(input_channels):
            self.modality_specific_encoders.append(PlainConvEncoder(1, n_stages, modality_specific_features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first))  
        #print('(modality_specific_features_per_stage[-1]:', modality_specific_features_per_stage[-1])
        #self.feature_reconstruction = Transformer(128, 1, 4, 32, 256)
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                         n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                         dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                         nonlin_first=nonlin_first)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

    def forward(self, x, selected_mask):
        batch_size, _, depth, height, width = x.size()
        modality_specific_inputs = torch.chunk(x, self.input_channels, dim=1)
        combined_skips = [[] for _ in range(self.n_stages)]  # 基于stages数量初始化

        # 计算每个stage的空间尺寸（假设每个stage深度、高度和宽度都下采样两倍）
        spatial_sizes = [(depth // (2 ** stage), height // (2 ** stage), width // (2 ** stage)) for stage in range(self.n_stages)]

        # 处理每个模态
        for i, modality_input in enumerate(modality_specific_inputs):
            # if torch.all(modality_input == 0):  # 检查模态是否缺失
            #     # 生成与该模态encoder输出相匹配的全0特征
            #     #print('yes')
            #     modality_skips = [torch.zeros(batch_size, self.modality_specific_features_per_stage[stage], *spatial_sizes[stage]).to(modality_input.device) for stage in range(self.n_stages)]
            # else:
            #     #处理并获取该模态的skip features
            modality_skips = self.modality_specific_encoders[i](modality_input)
                
            # 将当前模态的skips添加到对应分辨率的combined_skips中
            for stage_idx, skip in enumerate(modality_skips):
                skip = skip.unsqueeze(1)
                #print('skip:',skip.shape)
                if len(combined_skips[stage_idx]) == 0:
                    combined_skips[stage_idx] = skip
                else:
                    combined_skips[stage_idx] = torch.cat((combined_skips[stage_idx], skip), dim=1)
        
        
        Reconstructed_GT = combined_skips[-1].detach()
        for i in range(len(combined_skips)):
            for j, mask in enumerate(selected_mask):
                if not mask:
                    combined_skips[i][:, j, :, :, :, :] = 0
            # combined_skips[i] = combined_skips[i].contiguous().reshape(combined_skips[i].shape[0], -1, combined_skips[i].shape[-3], 
            #                                                            combined_skips[i].shape[-2], combined_skips[i].shape[-1])
        
        # bottleneck_feature = combined_skips[-1]
        # for j, mask in enumerate(selected_mask):
        #         if not mask:
        #             bottleneck_feature[:, j, :, :, :, :] = 0
        # bottleneck_feature_token = rearrange(bottleneck_feature, 'b m c h w d -> b (m h w d) c')
        # bottleneck_Rconstructed_Feature_Token = self.feature_reconstruction(bottleneck_feature_token)
        # bottleneck_Rconstructed_Feature = rearrange(bottleneck_Rconstructed_Feature_Token, 'b (m h w d) c -> b (m c) h w d', m=4, h=4, w=4, d=4)
        #print('shape:', bottleneck_Rconstructed_Feature.shape)
        for i in range(len(combined_skips)):
            for j, mask in enumerate(selected_mask):
                if not mask:
                    combined_skips[i][:, j, :, :, :, :] = 0
            combined_skips[i] = combined_skips[i].contiguous().reshape(combined_skips[i].shape[0], -1, combined_skips[i].shape[-3], 
                                                                        combined_skips[i].shape[-2], combined_skips[i].shape[-1])
        
        #for i in range(len(combined_skips)):
        #    print(f"combined_skips[{i}]: {combined_skips[i].shape}")
        # 接下来你可以通过decoder处理combined_skips
        return self.decoder(combined_skips), combined_skips[-1], Reconstructed_GT
class PlainConvUNetMissing(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

    def forward(self, x):
        
        skips = self.encoder(x)
        #print("skips", skips[0].shape)
        for i in range (len(skips)):
            print(skips[i].shape)
        return self.decoder(skips), skips[-1]

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

if __name__ == '__main__':
    masks_total = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
               [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], 
               [True, False, False, True], [True, True, False, False], [True, True, True, False], [True, False, True, True], 
               [True, True, False, True], [False, True, True, True], [True, True, True, True]]
   
    data = torch.rand((1, 4, 128, 128, 128))
    selected_mask = random.choice(masks_total)
    '''model = MultimodalPlainConvUNet(4, 6, (32, 64, 124, 256, 512, 512), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4,
                                (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True)'''
    model = MultimodalRecon(4, 6, (32, 64, 124, 256, 512, 512), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4,
                                (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True)


    print('shenme1')
    output = model(data, selected_mask)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    #print(model.compute_conv_feature_map_size(data.shape[2:]))

    data = torch.rand((1, 4, 512, 512))

    # model = PlainConvUNet(4, 8, (32, 64, 125, 256, 512, 512, 512, 512), nn.Conv2d, 3, (1, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2, 2, 2), 4,
    #                             (2, 2, 2, 2, 2, 2, 2), False, nn.BatchNorm2d, None, None, None, nn.ReLU, deep_supervision=True)



    # print(model.compute_conv_feature_map_size(data.shape[2:]))
