import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from layers import normalization
from layers import general_conv3d
from layers import prm_generator_laststage, prm_generator, region_aware_modal_fusion
from layers import *

basic_dims = 16
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

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        return x1, x2, x3, x4

class Decoder_sep(nn.Module):
    def __init__(self, num_cls=4, activation = 'softmax'):
        super(Decoder_sep, self).__init__()

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
        if activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('activation function not supported')

    def forward(self, x1, x2, x3, x4):
        de_x4 = self.d3_c1(self.d3(x4))

        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        pred = self.activation(logits)

        return pred

class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_fuse, self).__init__()

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

        self.RFM4 = region_aware_modal_fusion(in_channel=basic_dims*8, num_cls=num_cls)
        self.RFM3 = region_aware_modal_fusion(in_channel=basic_dims*4, num_cls=num_cls)
        self.RFM2 = region_aware_modal_fusion(in_channel=basic_dims*2, num_cls=num_cls)
        self.RFM1 = region_aware_modal_fusion(in_channel=basic_dims*1, num_cls=num_cls)

        self.prm_generator4 = prm_generator_laststage(in_channel=basic_dims*8, num_cls=num_cls)
        self.prm_generator3 = prm_generator(in_channel=basic_dims*4, num_cls=num_cls)
        self.prm_generator2 = prm_generator(in_channel=basic_dims*2, num_cls=num_cls)
        self.prm_generator1 = prm_generator(in_channel=basic_dims*1, num_cls=num_cls)


    def forward(self, x1, x2, x3, x4, mask):
        prm_pred4 = self.prm_generator4(x4, mask)
        de_x4 = self.RFM4(x4, prm_pred4.detach(), mask)
        fusion_x4 = de_x4
        de_x4 = self.d3_c1(self.up2(de_x4))

        prm_pred3 = self.prm_generator3(de_x4, x3, mask)
        de_x3 = self.RFM3(x3, prm_pred3.detach(), mask)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        prm_pred2 = self.prm_generator2(de_x3, x2, mask)
        de_x2 = self.RFM2(x2, prm_pred2.detach(), mask)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        prm_pred1 = self.prm_generator1(de_x2, x1, mask)
        de_x1 = self.RFM1(x1, prm_pred1.detach(), mask)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred, (prm_pred1, self.up2(prm_pred2), self.up4(prm_pred3), self.up8(prm_pred4)), fusion_x4

class Model(nn.Module):
    def __init__(self, num_cls=4):
        super(Model, self).__init__()
        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)
        self.decoder_sep = Decoder_sep(num_cls=num_cls)

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #

    def forward(self, x, mask):
        #extract feature from different layers
        flair_x1, flair_x2, flair_x3, flair_x4 = self.flair_encoder(x[:, 0:1, :, :, :])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4 = self.t1ce_encoder(x[:, 1:2, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4 = self.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4 = self.t2_encoder(x[:, 3:4, :, :, :])

        x1 = torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1) #Bx4xCxHWZ
        x2 = torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1)
        x3 = torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1)
        x4 = torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1)
        
        fuse_pred, prm_preds = self.decoder_fuse(x1, x2, x3, x4, mask)

        if self.is_training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4)
            t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4)
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4)
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4)
            return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), prm_preds
        return fuse_pred


class Style_encoder(nn.Module):

    def __init__(self, in_channels = 1, n_base_ch_se = 32):
        super(Style_encoder, self).__init__()
        
        layers = [BasicConv(in_channels, n_base_ch_se, 7, 
                                    stride=1, padding=(7-1) // 2, relu=True, norm=False)]
        
        layers.append(BasicConv(n_base_ch_se, n_base_ch_se*2, 4, 
                                stride=2, padding=(4-1) // 2, relu=True, norm=False))
        layers.append(BasicConv(n_base_ch_se*2, n_base_ch_se*4, 4, 
                                stride=2, padding=(4-1) // 2, relu=True, norm=False))
        layers.append(BasicConv(n_base_ch_se*4, n_base_ch_se*4, 4, 
                                stride=2, padding=(4-1) // 2, relu=True, norm=False))
        layers.append(BasicConv(n_base_ch_se*4, n_base_ch_se*4, 4, 
                                stride=2, padding=(4-1) // 2, relu=True, norm=False))
        self.encoder = nn.Sequential(*layers)
        self.final_conv = BasicConv(n_base_ch_se*4, n_base_ch_se*4, 1, 
                                stride=2, padding=(1-1) // 2, relu=False, norm=False)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.mean(x, [2,3,4], keepdim=True)
        x = self.final_conv(x)
        
        return x

class MLP(nn.Module):

    def __init__(self, in_ch = 128, mlp_ch = 128):
        super(MLP, self).__init__()
        
        self.channel = mlp_ch
        self.l1 = nn.Linear(in_ch, mlp_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(mlp_ch, mlp_ch)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.l_mu = nn.Linear(mlp_ch, mlp_ch)
        self.l_sigma = nn.Linear(mlp_ch, mlp_ch)

    def forward(self, style):
        x = style.view(style.size(0), -1)
        x = self.l1(x)
        x = self.relu1(x)
        
        x = self.l2(x)
        x = self.relu2(x)
        
        mu = self.l_mu(x)
        sigma = self.l_sigma(x)
        
        mu = mu.reshape(-1, self.channel, 1, 1, 1)
        sigma = sigma.reshape(-1, self.channel, 1, 1, 1)
        
        return mu, sigma 

class Image_decoder(nn.Module):

    def __init__(self, in_style_ch = 128, in_content_ch = 128, mlp_ch = 128, img_ch=1):
        super(Image_decoder, self).__init__()
        channel = mlp_ch
        self.mlp = MLP(in_style_ch, mlp_ch)
        
        res_blocks = []
        for i in range(4):
            res_blocks.append(Adaptive_resblock(in_content_ch, channel))
        self.res_blocks = nn.ModuleList(res_blocks)
        
        decoder_blocks = []
        for i in range(3):
            level_decoder = []
            level_decoder.append(nn.Upsample(scale_factor=2, mode='trilinear'))
            level_decoder.append(BasicConv(channel, channel // 2, 5, stride=1, padding=(5-1) // 2, relu=False, norm=False))
            decoder_blocks.append(nn.Sequential(*level_decoder))
            channel = channel // 2
            
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        
        self.final_conv = BasicConv(channel, img_ch, 7, stride=1, padding=(7-1) // 2, relu=False, norm=False)

    def forward(self, style, content):
        mu, sigma = self.mlp(style)
        x = content
        
        for i in range(len(self.res_blocks)):
            x = self.res_blocks[i](x, mu, sigma)
            
        for i in range(len(self.decoder_blocks)):
            x = self.decoder_blocks[i](x)
            x = F.layer_norm(x, x.shape[1:])
            x = F.relu(x, inplace=True)
        x = self.final_conv(x)
        
        return x, mu, sigma

class DC_Seg(nn.Module):
    def __init__(self, num_cls=4, fusion_type='RFM'):
        super(DC_Seg, self).__init__()
        self.fusion_type = fusion_type
        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        self.flair_style_encoder = Style_encoder()
        self.t1ce_style_encoder = Style_encoder()
        self.t1_style_encoder = Style_encoder()
        self.t2_style_encoder = Style_encoder()

        self.flair_decoder = Image_decoder()
        self.t1ce_decoder = Image_decoder()
        self.t1_decoder = Image_decoder()
        self.t2_decoder = Image_decoder()
        self.decoders = [self.flair_decoder, self.t1ce_decoder, self.t1_decoder, self.t2_decoder]
        #self.decoders = nn.ModuleList([
        #    self.flair_decoder, self.t1ce_decoder, self.t1_decoder, self.t2_decoder
        #])

        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)
        self.decoder_sep = Decoder_sep(num_cls=num_cls)

        in_out_ch = 128
        self.content_attn = (BasicConv(in_out_ch*4, 4, 3, stride=1, padding=(3-1) // 2, relu=False, norm=True))
        self.content_share_conv_list = (BasicConv(in_out_ch*4, in_out_ch, 1, stride=1, padding=(1-1) // 2, relu=True, norm=True))

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #

    def forward(self, x, mask):
        #extract feature from different layers
        flair_x1, flair_x2, flair_x3, flair_x4 = self.flair_encoder(x[:, 0:1, :, :, :])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4 = self.t1ce_encoder(x[:, 1:2, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4 = self.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4 = self.t2_encoder(x[:, 3:4, :, :, :])

        x1 = torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1) #Bx4xCxHWZ
        x2 = torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1)
        x3 = torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1)
        x4 = torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1)
        
        fuse_pred, prm_preds, fusion_x4 = self.decoder_fuse(x1, x2, x3, x4, mask)

        if self.is_training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4)
            t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4)
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4)
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4)
        
            flair_style, t1ce_style, t1_style, t2_style = self.flair_style_encoder(x[:, 0:1, :, :, :]), self.t1ce_style_encoder(x[:, 1:2, :, :, :]), self.t1_style_encoder(x[:, 2:3, :, :, :]), self.t2_style_encoder(x[:, 3:4, :, :, :])
            
            fusion_type = self.fusion_type
            if fusion_type == 'RFM':
                out = fusion_x4
            elif fusion_type == 'gated': 
                share_concat = torch.cat([flair_x4, t1ce_x4, t1_x4, t2_x4], 1)
                attnmap = self.content_attn(share_concat)
                attnmap = F.sigmoid(attnmap)
                share_content = []
                for i in range(4):
                    share_content.append(x4[:, i] * attnmap[:, i:i+1])
                share_content = torch.cat(share_content, 1)
                share_content = self.content_share_conv_list(share_content)
                out = share_content
            else:
                out = x4.mean(1)

            recon_list = []
            mu_list = []
            sigma_list = []
            for i, style in enumerate([flair_style, t1ce_style, t1_style, t2_style]):
                recon, mu, sigma = self.decoders[i](style, out)
                recon_list.append(recon)
                mu_list.append(mu)
                sigma_list.append(sigma)
                
            recon_out = torch.cat(recon_list, 1)

            # contents = x4.mean([3, 4, 5])
            contents = x4
            styles = torch.stack([flair_style, t1ce_style, t1_style, t2_style], 1).squeeze(-1, -2, -3)
            return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), prm_preds, recon_out, mu_list, sigma_list, contents, styles
        return fuse_pred

class Decoder_fuse_wmh(nn.Module):
    def __init__(self, num_cls=4, num_modal=2, activation='softmax'):
        super(Decoder_fuse_wmh, self).__init__()

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
        if activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('activation function not supported')

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)

        self.RFM4 = region_aware_modal_fusion_wmh(in_channel=basic_dims*8, num_cls=num_cls, num_modal=num_modal)
        self.RFM3 = region_aware_modal_fusion_wmh(in_channel=basic_dims*4, num_cls=num_cls, num_modal=num_modal)
        self.RFM2 = region_aware_modal_fusion_wmh(in_channel=basic_dims*2, num_cls=num_cls, num_modal=num_modal)
        self.RFM1 = region_aware_modal_fusion_wmh(in_channel=basic_dims*1, num_cls=num_cls, num_modal=num_modal)

        self.prm_generator4 = prm_generator_laststage(in_channel=basic_dims*8, num_cls=num_cls, num_modal=num_modal)
        self.prm_generator3 = prm_generator(in_channel=basic_dims*4, num_cls=num_cls, num_modal=num_modal)
        self.prm_generator2 = prm_generator(in_channel=basic_dims*2, num_cls=num_cls, num_modal=num_modal)
        self.prm_generator1 = prm_generator(in_channel=basic_dims*1, num_cls=num_cls, num_modal=num_modal)


    def forward(self, x1, x2, x3, x4, mask):
        prm_pred4 = self.prm_generator4(x4, mask)
        de_x4 = self.RFM4(x4, prm_pred4.detach(), mask)
        fusion_x4 = de_x4
        de_x4 = self.d3_c1(self.up2(de_x4))

        prm_pred3 = self.prm_generator3(de_x4, x3, mask)
        de_x3 = self.RFM3(x3, prm_pred3.detach(), mask)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        prm_pred2 = self.prm_generator2(de_x3, x2, mask)
        de_x2 = self.RFM2(x2, prm_pred2.detach(), mask)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        prm_pred1 = self.prm_generator1(de_x2, x1, mask)
        de_x1 = self.RFM1(x1, prm_pred1.detach(), mask)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.activation(logits)

        return pred, (prm_pred1, self.up2(prm_pred2), self.up4(prm_pred3), self.up8(prm_pred4)), fusion_x4

class DC_Seg_WMH(nn.Module):
    def __init__(self, num_cls=1, fusion_type='RFM', activation='sigmoid'):
        super(DC_Seg_WMH, self).__init__()
        self.fusion_type = fusion_type
        self.flair_encoder = Encoder()
        self.t1_encoder = Encoder()

        self.flair_style_encoder = Style_encoder()
        self.t1_style_encoder = Style_encoder()

        self.flair_decoder = Image_decoder()
        self.t1_decoder = Image_decoder()
        self.decoders = [self.flair_decoder, self.t1_decoder]

        self.decoder_fuse = Decoder_fuse_wmh(num_cls=num_cls,activation=activation)
        self.decoder_sep = Decoder_sep(num_cls=num_cls, activation=activation)

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #

    def forward(self, x, mask):
        #extract feature from different layers
        flair_x1, flair_x2, flair_x3, flair_x4 = self.flair_encoder(x[:, 0:1, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4 = self.t1_encoder(x[:, 1:2, :, :, :])

        x1 = torch.stack((flair_x1, t1_x1), dim=1) #Bx4xCxHWZ
        x2 = torch.stack((flair_x2, t1_x2), dim=1)
        x3 = torch.stack((flair_x3, t1_x3), dim=1)
        x4 = torch.stack((flair_x4, t1_x4), dim=1)
        
        fuse_pred, prm_preds, fusion_x4 = self.decoder_fuse(x1, x2, x3, x4, mask)

        if self.is_training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4)
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4)
        
            flair_style, t1_style = self.flair_style_encoder(x[:, 0:1, :, :, :]), self.t1_style_encoder(x[:, 1:2, :, :, :])
            
            fusion_type = self.fusion_type
            if fusion_type == 'RFM':
                out = fusion_x4
            else:
                out = x4.mean(1)

            recon_list = []
            mu_list = []
            sigma_list = []
            for i, style in enumerate([flair_style, t1_style]):
                recon, mu, sigma = self.decoders[i](style, out)
                recon_list.append(recon)
                mu_list.append(mu)
                sigma_list.append(sigma)
                
            recon_out = torch.cat(recon_list, 1)

            # contents = x4.mean([3, 4, 5])
            contents = x4
            styles = torch.stack([flair_style, t1_style], 1).squeeze(-1, -2, -3)
            return fuse_pred, (flair_pred, t1_pred), prm_preds, recon_out, mu_list, sigma_list, contents, styles
        return fuse_pred


if __name__ == '__main__':
    model = DC_Seg()
    model.is_training = True
    x = torch.randn(3, 4, 112, 112, 112)
    fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), prm_preds, recon_out, mu_list, sigma_list, contents, styles = model(x, None)
    print(fuse_pred.shape, flair_pred.shape, t1ce_pred.shape, t1_pred.shape, t2_pred.shape, recon_out.shape, contents.shape, styles.shape)

    # model = DC_Seg_WMH()
    # model.is_training = True
    # input = torch.randn(3, 2, 128, 128, 128)
    # mask = [[True, False], [True, True], [True, False]]
    # mask = torch.tensor(mask)
    # fuse_pred, (flair_pred, t1_pred), prm_preds, recon_out, mu_list, sigma_list, contents, styles = model(input, mask)
    # print(fuse_pred.shape, flair_pred.shape, t1_pred.shape, recon_out.shape, contents.shape, styles.shape)