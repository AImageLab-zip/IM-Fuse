import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import torch
import torch.nn.functional as F

# ****************************************************************************************
# ----------------------------------- Unet   Basic blocks  -------------------------------
# ****************************************************************************************
class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm_type='instance', use_dropout=True, bias=True):
        super(UNetEncoder, self).__init__()

        self.in_channels = in_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.features = nn.Sequential()
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout()

        in_features = in_channels
        for i in range(levels):
            out_features = (2**i) * feature_maps

            conv_block = UNetConvBlock3D(in_features, out_features, norm_type=norm_type, bias=bias)
            self.features.add_module('convblock%d' % (i+1), conv_block)

            pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i+1), pool)

            in_features = out_features


    def forward(self, inputs):
        encoder_outputs = []
        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'convblock%d' % (i+1))(outputs)

            if i == self.levels-1:
                continue
            if self.use_dropout:
                encoder_outputs.append(self.dropout(outputs))
            else:
                encoder_outputs.append(outputs)
            outputs = getattr(self.features, 'pool%d' % (i+1))(outputs)

        return encoder_outputs, outputs

class UNetEncoder_hved(nn.Module):
    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm_type='instance', use_dropout=True, bias=True):
        super(UNetEncoder_hved, self).__init__()

        self.in_channels = in_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.features = nn.Sequential()
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout()

        in_features = in_channels
        for i in range(levels):
            out_features = (2**i) * feature_maps

            conv_block = UNetConvBlock3D_hved(in_features, out_features, norm_type=norm_type, bias=bias)
            self.features.add_module('convblock%d' % (i+1), conv_block)

            pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i+1), pool)

            in_features = out_features


    def forward(self, inputs):
        encoder_outputs = []
        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'convblock%d' % (i+1))(outputs)

            if i == self.levels-1:
                continue
            if self.use_dropout:
                encoder_outputs.append(self.dropout(outputs))
            else:
                encoder_outputs.append(outputs)
            outputs = getattr(self.features, 'pool%d' % (i+1))(outputs)

        return encoder_outputs, outputs

class UNetDecoder(nn.Module):
    def __init__(self, out_channels, feature_maps=64, levels=4, norm_type='instance', bias=True):
        super(UNetDecoder, self).__init__()
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.features = nn.Sequential()

        for i in range(levels-1):
            upconv = UNetUpSamplingBlock3D(2**(levels-i-1) * feature_maps, 2**(levels-i-1) * feature_maps, deconv=False,
                                         bias=bias)
            self.features.add_module('upconv%d' % (i+1), upconv)

            conv_block = UNetConvBlock3D(2**(levels-i-2) * feature_maps * 3, 2**(levels-i-2) * feature_maps,
                                           norm_type=norm_type, bias=bias, flag='decoder')

            self.features.add_module('convblock%d' % (i+1), conv_block)

        self.score = nn.Conv3d(feature_maps, out_channels, kernel_size=1, stride=1, bias=bias)

    def forward(self, inputs, encoder_outputs):
        encoder_outputs.reverse()
        outputs = inputs
        for i in range(self.levels-1):
            outputs = getattr(self.features, 'upconv%d' % (i+1))(outputs)
            outputs = torch.cat([encoder_outputs[i], outputs], dim=1)
            outputs = getattr(self.features, 'convblock%d' % (i+1))(outputs)
        encoder_outputs.reverse()
        return self.score(outputs)

class UNetDecoder_hved(nn.Module):
    def __init__(self, out_channels, feature_maps=64, levels=4, norm_type='instance', bias=True):
        super(UNetDecoder_hved, self).__init__()
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.features = nn.Sequential()

        for i in range(levels-1):
            upconv = UNetUpSamplingBlock3D(2**(levels-i-1) * feature_maps, 2**(levels-i-1) * feature_maps, deconv=False,
                                         bias=bias)
            self.features.add_module('upconv%d' % (i+1), upconv)

            conv_block = UNetConvBlock3D_hved(2**(levels-i-2) * feature_maps * 3, 2**(levels-i-2) * feature_maps,
                                           norm_type=norm_type, bias=bias, flag='decoder')

            self.features.add_module('convblock%d' % (i+1), conv_block)

        self.score = nn.Conv3d(feature_maps, out_channels, kernel_size=1, stride=1, bias=bias)

    def forward(self, inputs, encoder_outputs):
        encoder_outputs.reverse()
        outputs = inputs
        for i in range(self.levels-1):
            outputs = getattr(self.features, 'upconv%d' % (i+1))(outputs)
            outputs = torch.cat([encoder_outputs[i], outputs], dim=1)
            outputs = getattr(self.features, 'convblock%d' % (i+1))(outputs)
        encoder_outputs.reverse()
        return self.score(outputs)

class UNetUpSamplingBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, bias=True):
        super(UNetUpSamplingBlock3D, self).__init__()
        self.deconv = deconv
        if self.deconv:
            self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, *inputs):
        if len(inputs) == 2:
            return self.forward_concat(inputs[0], inputs[1])
        else:
            return self.forward_standard(inputs[0])

    def forward_concat(self, inputs1, inputs2):
        return torch.cat([inputs1, self.up(inputs2)], 1)

    def forward_standard(self, inputs):
        return self.up(inputs)

class UNetConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='SAME', norm_type='instance', bias=True, flag='encoder'):
        super(UNetConvBlock3D, self).__init__()
        if flag=='encoder':
            self.conv1 = ConvNormRelu3D(in_channels, out_channels//2, kernel_size=kernel_size, padding=padding,
                                      norm_type=norm_type, bias=bias)
            self.conv2 = ConvNormRelu3D(out_channels//2, out_channels, kernel_size=kernel_size, padding=padding,
                                      norm_type=norm_type, bias=bias)
        else:
            self.conv1 = ConvNormRelu3D(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                        norm_type=norm_type, bias=bias)
            self.conv2 = ConvNormRelu3D(out_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                        norm_type=norm_type, bias=bias)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class UNetConvBlock3D_hved(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='SAME', norm_type='instance', bias=True, flag='encoder'):
        super(UNetConvBlock3D_hved, self).__init__()
        if flag=='encoder':
            self.conv1 = NormRelu3DConv(in_channels, out_channels//2, kernel_size=kernel_size, padding=padding,
                                      norm_type=norm_type, bias=bias)
            self.conv2 = NormRelu3DConv(out_channels//2, out_channels, kernel_size=kernel_size, padding=padding,
                                      norm_type=norm_type, bias=bias)
        else:
            self.conv1 = NormRelu3DConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                        norm_type=norm_type, bias=bias)
            self.conv2 = NormRelu3DConv(out_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                        norm_type=norm_type, bias=bias)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class ConvNormRelu3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding='SAME', bias=True, dilation=1, norm_type='instance'):

        super(ConvNormRelu3D, self).__init__()
        norm = nn.BatchNorm3d if norm_type == 'batch' else nn.InstanceNorm3d
        if padding == 'same':
            p = padding
        elif padding == 'SAME':
            p = kernel_size // 2
        else:
            p = 0


        self.unit = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                            padding=p, stride=stride, bias=bias, dilation=dilation),
                                  norm(out_channels),
                                  nn.LeakyReLU(0.01, inplace=True))

    def forward(self, inputs):
        return self.unit(inputs)

class NormRelu3DConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding='SAME', bias=True, dilation=1, norm_type='instance'):

        super(NormRelu3DConv, self).__init__()
        norm = nn.BatchNorm3d if norm_type == 'batch' else nn.InstanceNorm3d
        if padding == 'SAME':
            p = kernel_size // 2
        else:
            p = 0

        self.unit = nn.Sequential(
            norm(in_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=p, stride=stride, bias=bias, dilation=dilation)
        )

    def getpre_n(self, pre):
        re = {}
        p = pre.flatten()
        # print(pre.shape)
        for i in range(len(p)):
            n = int(abs(p[i]))
            if n in re.keys():
                re[n] += 1
            else:
                re[n] = 1
        # print(re)

    def forward(self, inputs):
        return self.unit(inputs)

# ****************************************************************************************
# ------------------------------ TFusion Basic blocks  ----------------------------------
# ****************************************************************************************
class AF_3d(nn.Module):
    def __init__(self, embedding_dim, nhead=4):
        super(AF_3d, self).__init__()
        from net.basicTrans import FFTransformerEncoderLayer
        d_model = embedding_dim
        self.fusion_block1 = nn.TransformerEncoder(FFTransformerEncoderLayer(embedding_dim, nhead=nhead, dim_feedforward=d_model * 4), num_layers=2)
        self.fusion_block2 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   batch_first=True, dim_feedforward=d_model * 4), num_layers=2)
        self.conv_map = nn.Conv1d(embedding_dim*2, 1, 1, 1)
    def forward(self, x):
        x1 = self.fusion_block1(x)
        x2 = self.fusion_block2(x)
        _map = self.conv_map(torch.cat([x1,x2],dim=2).transpose(1,2)).transpose(1,2)
        _map = F.sigmoid(_map)
        o = _map*x1+(1-_map)*x2
        return o
        
 #  def forward(self, x):
        x=self.embed(x)
        x1=self.AF_3d(o1)
        x2=self.AF_3d(o2)
        out=self.FFN(torch.cat((o1,o2),1))
        return self.norm(out)
 
    
class TF_3D(nn.Module):
    def __init__(self, embedding_dim=1024, volumn_size=8, nhead=4, num_layers=8, method='TF', trans='basic'):
        super(TF_3D, self).__init__()
        self.embedding_dim = embedding_dim
        self.d_model = self.embedding_dim
        self.patch_dim = 8
        self.method = method
        self.scale_factor = volumn_size // self.patch_dim

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead,
                                                   batch_first=True, dim_feedforward=self.d_model * 4)
        
        if trans == 'basic':        
            self.fusion_block = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        elif trans == 'ff':
            # from net.basicTrans import FFTransformerEncoderLayer
            # self.fusion_block1 = FFTransformerEncoderLayer(embedding_dim, nhead=nhead, dim_feedforward=self.d_model * 4)
            # self.fusion_block2 = nn.TransformerEncoder(encoder_layer, num_layers=2)
            # self.conv_map = nn.Conv3d(embedding_dim, 1, 3, 1)
            self.fusion_block = AF_3d(embedding_dim, nhead)
        elif trans == 'la':
            from net.basicTrans import LTEncoderLayer
            from local_attention.local_attention import LocalAttention
            self.fusion_block = LTEncoderLayer(embedding_dim, nhead=nhead, dim_feedforward=self.d_model * 4)

        self.dropout = nn.Dropout(p=0.1)
        self.avgpool = nn.AdaptiveAvgPool3d((self.patch_dim, self.patch_dim, self.patch_dim))
        self.upsample = DUpsampling3D(self.embedding_dim, self.scale_factor)
        if method=='Token':
            self.fusion_token = nn.parameter.Parameter(torch.zeros((1, self.patch_dim ** 3, self.d_model)))

    def forward(self, all_content):

        n_modality = len(all_content)
        token_content = self.project(all_content)
        position_enc = PositionalEncoding(self.d_model, token_content.size(1))
        
        # print(token_content.size())
        out = self.fusion_block(self.dropout(position_enc(token_content)))
        atten_map = self.reproject(out, n_modality, self.method)
        return self.atten(all_content, atten_map, n_modality)

    def project(self, all_content):
        n_modality = len(all_content)
        token_content_in = None
        for i in range(n_modality):
            content = self.avgpool(all_content[i])
            content = content.permute(0, 2, 3, 4, 1).contiguous()
            content2 = content.view(content.size(0), -1, self.embedding_dim)
            if i == 0:
                token_content_in = content2
            else:
                token_content_in = torch.cat([token_content_in, content2], dim=1)
        return token_content_in



    def reproject(self, atten_map, n_modality, method):
        n_patch = self.patch_dim ** 3
        a_m0 = None
        for i in range(n_modality):
            atten_mapi = atten_map[:, n_patch*i : n_patch*(i+1), :].view(
                    atten_map.size(0),
                    self.patch_dim,
                    self.patch_dim,
                    self.patch_dim,
                    self.embedding_dim,
                )

            atten_mapi = atten_mapi.permute(0, 4, 1, 2, 3).contiguous()
            atten_mapi = self.upsample(atten_mapi).unsqueeze(dim=0)

            if a_m0 == None:
                a_m0 = atten_mapi
            else:
                a_m0 = torch.cat([a_m0, atten_mapi], dim=0)

        a_m = F.softmax(a_m0, dim=0)
        return a_m   
    


    def compute_window_weights(atten_map, kernel_size=3, upsample_size=None):
        pooled_map = F.avg_pool3d(atten_map, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        # 应用softmax来计算权重
        weights = F.softmax(pooled_map, dim=1)
        # 上采样到原始尺寸
        if upsample_size is not None:
            weights = F.interpolate(weights, size=upsample_size, mode='trilinear', align_corners=True)
        return weights 

    
   

    def atten(self, all_content, atten_map, n_modality):
        output = None
        for i in range(n_modality):
            a_m = atten_map[i, :, :, :, :, :]
            assert all_content[i].shape == a_m.shape, 'all_content and a_m cannot match!!'
            if output == None:
                output = all_content[i] * a_m
            else:
                output += all_content[i] * a_m
        return output  

class DUpsampling3D(nn.Module):
    def __init__(self, inplanes, scale):
        super(DUpsampling3D, self).__init__()
        output_channel = inplanes * (scale ** 3)
        self.conv_3d = nn.Conv3d(inplanes, output_channel, kernel_size=1, stride=1, bias=False)
        self.scale = scale

    def forward(self, x):
        x = self.conv_3d(x)
        B, C, D, H, W = x.size()

        x_permuted = x.permute(0, 4, 3, 2, 1)

        x_permuted = x_permuted.contiguous().view((B, W, H, D * self.scale, int(C / (self.scale))))

        x_permuted = x_permuted.permute(0, 3, 1, 2, 4)


        x_permuted = x_permuted.contiguous().view((B, D * self.scale, W, H * self.scale, int(C / (self.scale**2))))

        x_permuted = x_permuted.permute(0, 1, 3, 2, 4)

        x_permuted = x_permuted.contiguous().view(
            (B, D * self.scale, H * self.scale, W * self.scale, int(C / (self.scale **3))))

        x = x_permuted.permute(0, 4, 1, 2, 3)

        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach().to(torch.device('cuda'))


# ****************************************************************************************
# ------------------------------------ rmbts basic blocks  ------------------------------
# ****************************************************************************************
class general_conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding='SAME', bias=True, dilation=1, norm_type=True, dropout=0.0, act=True):

        super(general_conv3d, self).__init__()
        if padding == 'SAME':
            p = kernel_size // 2
        else:
            p = 0
        self.unit = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                            padding=p, stride=stride, bias=bias, dilation=dilation))
        if dropout != 0.0:
            self.unit.add_module('drop', nn.Dropout(p=dropout))
        if norm_type:
            self.unit.add_module('norm', nn.InstanceNorm3d(out_channels))
        if act:
            self.unit.add_module('activation',nn.LeakyReLU(0.01, inplace=True))

    def forward(self, inputs):
        return self.unit(inputs)

class linear(nn.Module):
    def __init__(self, units):

        super(linear, self).__init__()
        self.unit = nn.Sequential(nn.Flatten(),
                                  nn.LazyLinear(units))
    def forward(self, inputs):
        return self.unit(inputs)

class style_encoder(nn.Module):
    def __init__(self, in_channels, n_base_ch_se=32):
        super(style_encoder, self).__init__()

        self.unit = nn.Sequential(
            general_conv3d(in_channels, n_base_ch_se, kernel_size=7, stride=1),
            general_conv3d(n_base_ch_se, n_base_ch_se*2, kernel_size=4, stride=2),
            general_conv3d(n_base_ch_se*2, n_base_ch_se*4, kernel_size=4, stride=2),
            general_conv3d(n_base_ch_se*4, n_base_ch_se*4, kernel_size=4, stride=2),
            general_conv3d(n_base_ch_se*4, n_base_ch_se*4, kernel_size=4, stride=2),
        )
        self.unit2 = general_conv3d(n_base_ch_se*4, 8, kernel_size=1, stride=1, norm_type=False, act=False)

    def forward(self, inputs):
        output = self.unit(inputs)
        output = torch.mean(output, dim=(2, 3, 4), keepdim=True)
        return self.unit2(output)


class content_encoder(nn.Module):
    def __init__(self, in_channels, n_base_filters=16):
        super(content_encoder, self).__init__()
        self.unit1_0 = general_conv3d(in_channels, n_base_filters)
        self.unit1 = nn.Sequential(
            general_conv3d(n_base_filters, n_base_filters, dropout=0.3),
            general_conv3d(n_base_filters, n_base_filters),
        )

        self.unit2_0 = general_conv3d(n_base_filters, n_base_filters*2, stride=2)
        self.unit2 = nn.Sequential(
            general_conv3d(n_base_filters*2, n_base_filters*2, dropout=0.3),
            general_conv3d(n_base_filters*2, n_base_filters*2),
        )

        self.unit3_0 = general_conv3d(n_base_filters*2, n_base_filters*4, stride=2)
        self.unit3 = nn.Sequential(
            general_conv3d(n_base_filters*4, n_base_filters*4, dropout=0.3),
            general_conv3d(n_base_filters*4, n_base_filters*4 ),
        )

        self.unit4_0 = general_conv3d(n_base_filters * 4, n_base_filters*8, stride=2)
        self.unit4 = nn.Sequential(
            general_conv3d(n_base_filters*8, n_base_filters * 8, dropout=0.3),
            general_conv3d(n_base_filters * 8, n_base_filters * 8),
        )

    def forward(self, inputs):
        output1_0 = self.unit1_0(inputs)
        output1 = self.unit1(output1_0) + output1_0

        output2_0 = self.unit2_0(output1)
        output2 = self.unit2(output2_0) + output2_0

        output3_0 = self.unit3_0(output2)
        output3 = self.unit3(output3_0) + output3_0

        output4_0 = self.unit4_0(output3)
        output4 = self.unit4(output4_0) + output4_0

        return {
            's1': output1,
            's2': output2,
            's3': output3,
            's4': output4,
        }

class image_decoder(nn.Module):
    def __init__(self, input_channel, mlp_ch=128, img_ch=1, scale=4):
        super(image_decoder, self).__init__()
        channel = mlp_ch
        self.scale = scale
        self.ar1 = adaptive_resblock(input_channel, channel)
        self.ar2 = adaptive_resblock(channel, channel)
        self.ar3 = adaptive_resblock(channel, channel)
        self.ar4 = adaptive_resblock(channel, channel)

        self.mlp = mlp(channel)
        self.features = nn.Sequential()
        in_channel = channel
        out_channel = channel
        self.lrelu = nn.LeakyReLU(0.01)
        for i in range(scale-1):
            out_channel = in_channel // 2
            up_block = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
            conv_block = general_conv3d(in_channel, out_channel, kernel_size=5, stride=1)
            norm_block = nn.InstanceNorm3d(out_channel)
            self.features.add_module('upblock%d' % (i+1), up_block)
            self.features.add_module('convblock%d' % (i+1), conv_block)
            self.features.add_module('normblock%d' % (i+1), norm_block)

            in_channel = out_channel
        self.conv_final = general_conv3d(out_channel, img_ch, kernel_size=7, stride=1)


    def forward(self, style, content):
        mu, sigma = self.mlp(style)
        x = self.ar1(content, mu, sigma)
        x = self.ar2(x, mu, sigma)
        x = self.ar3(x, mu, sigma)
        x = self.ar4(x, mu, sigma)

        for i in range(self.scale - 1):
            x = getattr(self.features, 'upblock%d' % (i+1))(x)
            x = getattr(self.features, 'convblock%d' % (i+1))(x)
            x = getattr(self.features, 'normblock%d' % (i+1))(x)
            x = self.lrelu(x)

        x = self.conv_final(x)

        return x, mu, sigma

class mask_decoder(nn.Module):
    def __init__(self, input_channel, n_base_filters=16, num_cls=4):
        super(mask_decoder, self).__init__()

        self.features = nn.Sequential()
        in_channel = input_channel
        out_channel = n_base_filters * 4
        for i in range(3):
            up_block = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
            conv_block1 = general_conv3d(in_channel, out_channel)
            conv_block2 = general_conv3d(out_channel*2, out_channel)
            conv_block3 = general_conv3d(out_channel, out_channel, kernel_size=1)

            self.features.add_module('upblock%d' % (i+1), up_block)
            self.features.add_module('convblock1%d' % (i+1), conv_block1)
            self.features.add_module('convblock2%d' % (i+1), conv_block2)
            self.features.add_module('convblock3%d' % (i+1), conv_block3)

            in_channel = out_channel
            out_channel = out_channel // 2

        self.conv_seg = general_conv3d(in_channel, num_cls, kernel_size=1, norm_type=False, act=False)


    def forward(self, inp):
        input = [inp['e4_out'], inp['e3_out'], inp['e2_out'], inp['e1_out']]

        out = input[0]
        for i in range(3):
            out = getattr(self.features, 'upblock%d' % (i + 1))(out)
            out = getattr(self.features, 'convblock1%d' % (i + 1))(out)
            out = torch.cat([out, input[i+1]], dim=1)
            out = getattr(self.features, 'convblock2%d' % (i + 1))(out)
            out = getattr(self.features, 'convblock3%d' % (i + 1))(out)
        seg = self.conv_seg(out)

        return seg




class adaptive_resblock(nn.Module):
    def __init__(self,input_channel, channel):
        super(adaptive_resblock, self).__init__()
        self.conv1 = general_conv3d(input_channel, channel)
        self.lrelu = nn.LeakyReLU(0.01)
        self.conv2 = general_conv3d(channel, channel)


    def forward(self, x_init, mu, sigma):
        x = self.adaptive_instance_norm(self.conv1(x_init), mu, sigma)
        x = self.lrelu(x)
        x = self.adaptive_instance_norm(self.conv2(x), mu, sigma)
        return x + x_init


    def adaptive_instance_norm(self, content, gamma, beta):
        c_mean = torch.mean(content, dim=(2, 3, 4), keepdim=True)
        c_std = torch.std(content, dim=(2, 3, 4), keepdim=True)
        return gamma * ((content - c_mean) / c_std) + beta

class mlp(nn.Module):
    def __init__(self, channel):
        super(mlp, self).__init__()
        self.channel = channel
        self.unit = nn.Sequential(
            linear(channel),
            nn.LeakyReLU(0.01),
            linear(channel),
            nn.LeakyReLU(0.01),
        )
        self.get_mu = linear(channel)
        self.get_sigma = linear(channel)

    def forward(self, style):
        s = self.unit(style)
        mu = self.get_mu(s)
        sigma = self.get_sigma(s)

        mu = mu.view(-1, self.channel, 1, 1, 1)
        sigma = sigma.view(-1, self.channel, 1, 1, 1)

        return mu, sigma


# ****************************************************************************************
# ------------------------------------- lmcr basic blocks  -------------------------------
# ****************************************************************************************
class ResDilBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, norm_type='instance', bias=True, flag='encoder'):
        super(ResDilBlock3D, self).__init__()

        self.conv1 = ConvNormRelu3D(in_channels, out_channels, kernel_size=kernel_size, padding='same',
                                      norm_type=norm_type, dilation=2, bias=bias)
        self.conv2 = ConvNormRelu3D(out_channels, out_channels, kernel_size=kernel_size, padding='same',
                                      norm_type=norm_type, dilation=4, bias=bias)
        self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return self.relu(outputs+inputs)


class LMCREncoder(nn.Module):
    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm_type='instance', use_dropout=True, bias=True):
        super(LMCREncoder, self).__init__()

        self.in_channels = in_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.features = nn.Sequential()
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout()

        in_features = in_channels
        for i in range(levels):
            out_features = (2**i) * feature_maps

            conv_block = ConvNormRelu3D(in_features, out_features, norm_type=norm_type, bias=bias)
            self.features.add_module('convblock%d' % (i+1), conv_block)

            resdil_block = ResDilBlock3D(out_features, out_features, norm_type=norm_type, bias=bias)
            self.features.add_module('resdilblock%d' % (i + 1), resdil_block)

            pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i+1), pool)

            in_features = out_features


    def forward(self, inputs):
        encoder_outputs = []
        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'convblock%d' % (i+1))(outputs)
            outputs = getattr(self.features, 'resdilblock%d' % (i+1))(outputs)

            if i == self.levels-1:
                continue
            if self.use_dropout:
                encoder_outputs.append(self.dropout(outputs))
            else:
                encoder_outputs.append(outputs)
            outputs = getattr(self.features, 'pool%d' % (i+1))(outputs)

        return encoder_outputs, outputs


class LMCRDecoder(nn.Module):
    def __init__(self, out_channels, feature_maps=64, levels=4, norm_type='instance', bias=True, type='seg'):
        super(LMCRDecoder, self).__init__()
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.type = type
        self.features = nn.Sequential()

        for i in range(levels-1):
            upconv = UNetUpSamplingBlock3D(2**(levels-i-1) * feature_maps, 2**(levels-i-1) * feature_maps, deconv=False,
                                         bias=bias)
            self.features.add_module('upconv%d' % (i+1), upconv)

            conv_block = ConvNormRelu3D(2**(levels-i-2) * feature_maps * 3, 2**(levels-i-2) * feature_maps, norm_type=norm_type, bias=bias)
            self.features.add_module('convblock%d' % (i+1), conv_block)

            resdil_block = ResDilBlock3D(2**(levels-i-2) * feature_maps, 2**(levels-i-2) * feature_maps, norm_type=norm_type, bias=bias)
            self.features.add_module('resdilblock%d' % (i+1), resdil_block)

            if self.type=='seg':
                conv = nn.Conv3d(2**(levels-i-2) * feature_maps, out_channels, kernel_size=1, stride=1, bias=bias)
                self.features.add_module('conv%d' % (i + 1), conv)
                up = UNetUpSamplingBlock3D(2 ** (levels - i - 1) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                           deconv=False,
                                           bias=bias)
                self.features.add_module('up%d' % (i + 1), up)
        self.score = nn.Conv3d(feature_maps, out_channels, kernel_size=1, stride=1, bias=bias)

    def forward(self, inputs, encoder_outputs):
        encoder_outputs.reverse()
        outputs = inputs
        deep_outputs = None

        for i in range(self.levels-1):
            outputs = getattr(self.features, 'upconv%d' % (i+1))(outputs)
            outputs = torch.cat([encoder_outputs[i], outputs], dim=1)
            outputs = getattr(self.features, 'convblock%d' % (i+1))(outputs)
            outputs = getattr(self.features, 'resdilblock%d' % (i+1))(outputs)
            if self.type == 'seg' and i != self.levels-2:
                if deep_outputs is None:
                    deep_outputs = getattr(self.features, 'conv%d' % (i+1))(outputs.clone())
                else:
                    deep_outputs += getattr(self.features, 'conv%d' % (i+1))(outputs.clone())
                deep_outputs = getattr(self.features, 'up%d' % (i+1))(deep_outputs)

        encoder_outputs.reverse()
        if type == 'seg':
            return deep_outputs + self.score(outputs)
        else:
            return self.score(outputs)

class MPE(nn.Module):
    def __init__(self, channels):
        super(MPE, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.unit = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(channels, 4)
        )

    def forward(self, data):
        output = self.pool(data).view(data.size(0),-1)
        output = self.unit(output)
        return output


class CR(nn.Module):
    def __init__(self, channels):
        super(CR, self).__init__()
        self.MPE = MPE(channels)

    def forward(self, inputs):
        outputs = []

        f0 = self.MPE(inputs[0])
        outputs.append(f0[0][0]+
                       f0[0][1]*inputs[1]+
                       f0[0][2]*inputs[2]+
                       f0[0][3]*inputs[3])

        f1 = self.MPE(inputs[1])
        outputs.append(f1[0][0] * inputs[0] +
                       f1[0][1] +
                       f1[0][2] * inputs[2] +
                       f1[0][3] * inputs[3])

        f2 = self.MPE(inputs[2])
        outputs.append(f2[0][0] * inputs[0] +
                       f2[0][1] * inputs[1] +
                       f2[0][2] +
                       f2[0][3] * inputs[3])

        f3 = self.MPE(inputs[3])
        outputs.append(f3[0][0] * inputs[0] +
                       f3[0][1] * inputs[1] +
                       f3[0][2] * inputs[2] +
                       f3[0][3] )
        return outputs

class LMCR_Fusion(nn.Module):
    def __init__(self, channels):
        super(LMCR_Fusion, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.ca = nn.Sequential(
            nn.Linear(channels*4, channels*4),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(channels*4, channels*4)
        )
        self.sa = nn.Conv3d(channels*4, 1, kernel_size=1, stride=1)
        self.conv = nn.Conv3d(channels*4, channels, kernel_size=1, stride=1)
    def forward(self, inputs):
        input = None
        for i in range(len(inputs)):
            if input is None:
                input = inputs[i]
            else:
                input = torch.cat([input, inputs[i]],dim=1)

        ca_map = torch.sigmoid(self.ca(self.pool(input).view(input.size(0),-1)))
        ca_map = ca_map.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        sa_map = torch.sigmoid(self.sa(input))
        return self.conv(input*ca_map+input*sa_map)
