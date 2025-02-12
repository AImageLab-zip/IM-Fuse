"""Code for constructing the model and get the outputs from the model."""
import torch.nn as nn
import torch.nn.functional as F
import torch
import layers_robustseg
import numpy as np

n_base_filters = 16
n_base_ch_se = 32
mlp_ch = 128
img_ch = 1
scale = 4
num_cls = 4


class style_encoder(nn.Module):
    def __init__(self):
        super(style_encoder, self).__init__()
        self.c_0 = layers_robustseg.general_conv3d(in_ch=1, out_ch=n_base_ch_se, k_size=7, stride=1, padding=3, pad_type="reflect", norm=None, act_type='relu')
        self.c_1 = layers_robustseg.general_conv3d(in_ch=n_base_ch_se, out_ch=n_base_ch_se*2, k_size=4, stride=2, padding=1, pad_type="reflect", norm=None, act_type='relu')
        self.c_2 = layers_robustseg.general_conv3d(in_ch=n_base_ch_se*2, out_ch=n_base_ch_se*4, k_size=4, stride=2, padding=1, pad_type="reflect", norm=None, act_type='relu')
        self.c_3 = layers_robustseg.general_conv3d(in_ch=n_base_ch_se*4, out_ch=n_base_ch_se*4, k_size=4, stride=2, padding=1, pad_type="reflect", norm=None, act_type='relu')
        self.c_4 = layers_robustseg.general_conv3d(in_ch=n_base_ch_se*4, out_ch=n_base_ch_se*4, k_size=4, stride=2, padding=1, pad_type="reflect", norm=None, act_type='relu')
        self.se_logit = layers_robustseg.general_conv3d(in_ch=n_base_ch_se*4, out_ch=8, k_size=1, stride=1, padding=0, pad_type="reflect", norm=None, act_type=None)
    
    def forward(self,input):    #[B,1,80,80,80]
        x = self.c_0(input)     #[B,32,80,80,80]
        x = self.c_1(x)         #[B,64,40,40,40]
        x = self.c_2(x)         #[B,128,20,20,20]
        x = self.c_3(x)         #[B,128,10,10,10]
        x = self.c_4(x)         #[B,128,5,5,5]
        x = torch.mean(x, dim=(2,3,4), keepdims=True)
        x = self.se_logit(x)    #[B,8,1,1,1]
        return x                #[B,8,1,1,1]

class content_encoder(nn.Module):
    def __init__(self):
        super(content_encoder, self).__init__()
        self.e1_c1 = layers_robustseg.general_conv3d(in_ch=1, out_ch=n_base_filters, k_size=3, stride=1, padding=1, pad_type='reflect')
        self.e1_c2 = layers_robustseg.general_conv3d(in_ch=n_base_filters, out_ch=n_base_filters, k_size=3, stride=1, padding=1, pad_type='reflect', drop_rate=0.3)
        self.e1_c3 = layers_robustseg.general_conv3d(in_ch=n_base_filters, out_ch=n_base_filters, k_size=3, stride=1, padding=1, pad_type='reflect')

        self.e2_c1 = layers_robustseg.general_conv3d(in_ch=n_base_filters, out_ch=n_base_filters*2, k_size=3, stride=2, padding=1, pad_type='reflect')
        self.e2_c2 = layers_robustseg.general_conv3d(in_ch=n_base_filters*2, out_ch=n_base_filters*2, k_size=3, stride=1, padding=1, pad_type='reflect', drop_rate=0.3)
        self.e2_c3 = layers_robustseg.general_conv3d(in_ch=n_base_filters*2, out_ch=n_base_filters*2, k_size=3, stride=1, padding=1, pad_type='reflect')

        self.e3_c1 = layers_robustseg.general_conv3d(in_ch=n_base_filters*2, out_ch=n_base_filters*4, k_size=3, stride=2, padding=1, pad_type='reflect')
        self.e3_c2 = layers_robustseg.general_conv3d(in_ch=n_base_filters*4, out_ch=n_base_filters*4, k_size=3, stride=1, padding=1, pad_type='reflect',drop_rate=0.3)
        self.e3_c3 = layers_robustseg.general_conv3d(in_ch=n_base_filters*4, out_ch=n_base_filters*4, k_size=3, stride=1, padding=1, pad_type='reflect')

        self.e4_c1 = layers_robustseg.general_conv3d(in_ch=n_base_filters*4, out_ch=n_base_filters*8, k_size=3, stride=2, padding=1, pad_type='reflect')
        self.e4_c2 = layers_robustseg.general_conv3d(in_ch=n_base_filters*8, out_ch=n_base_filters*8, k_size=3, stride=1, padding=1, pad_type='reflect')
        self.e4_c3 = layers_robustseg.general_conv3d(in_ch=n_base_filters*8, out_ch=n_base_filters*8, k_size=3, stride=1, padding=1, pad_type='reflect')

    def forward(self,x):            #[B,1,80,80,80]
        e1_c1 = self.e1_c1(x)       #[B,16,80,80,80]
        e1_c2 = self.e1_c2(e1_c1)   #[B,16,80,80,80]
        e1_c3 = self.e1_c3(e1_c2)   #[B,16,80,80,80]
        e1_out = e1_c1 + e1_c3      #[B,16,80,80,80]

        e2_c1 = self.e2_c1(e1_out)  #[B,32,40,40,40]
        e2_c2 = self.e2_c2(e2_c1)   #[B,32,40,40,40]
        e2_c3 = self.e2_c3(e2_c2)   #[B,32,40,40,40]
        e2_out = e2_c1 + e2_c3      #[B,32,40,40,40]

        e3_c1 = self.e3_c1(e2_out)  #[B,64,20,20,20]
        e3_c2 = self.e3_c2(e3_c1)   #[B,64,20,20,20]
        e3_c3 = self.e3_c3(e3_c2)   #[B,64,20,20,20]
        e3_out = e3_c1 + e3_c3      #[B,64,20,20,20]

        e4_c1 = self.e4_c1(e3_out)  #[B,128,10,10,10]
        e4_c2 = self.e4_c2(e4_c1)   #[B,128,10,10,10]
        e4_c3 = self.e4_c3(e4_c2)   #[B,128,10,10,10]
        e4_out = e4_c1 + e4_c3      #[B,128,10,10,10]

        return {
            's1':e1_out,    #[B,16,80,80,80]
            's2':e2_out,    #[B,32,40,40,40]
            's3':e3_out,    #[B,64,20,20,20]
            's4':e4_out,    #[B,128,10,10,10]
        }

class image_decoder(nn.Module):
    def __init__(self):
        super(image_decoder, self).__init__()
        self.channel = mlp_ch
        self.mlp = mlp()
        self.res_0 =layers_robustseg.adaptive_resblock()
        self.res_1 =layers_robustseg.adaptive_resblock()
        self.res_2 =layers_robustseg.adaptive_resblock()
        self.res_3 =layers_robustseg.adaptive_resblock()

        self.conv_0 = layers_robustseg.general_conv3d(in_ch=128, out_ch=64, k_size=5, stride=1, padding=2, norm=None, act_type=None)
        self.conv_1 = layers_robustseg.general_conv3d(in_ch=64, out_ch=32, k_size=5, stride=1, padding=2, norm=None, act_type=None)
        self.conv_2 = layers_robustseg.general_conv3d(in_ch=32, out_ch=16, k_size=5, stride=1, padding=2, norm=None, act_type=None)
        self.layer_norm_1 = nn.LayerNorm([20,20,20], eps=1e-06)##########################################################
        self.layer_norm_2 = nn.LayerNorm([40,40,40], eps=1e-06)##########################################################
        self.layer_norm_3 = nn.LayerNorm([80,80,80], eps=1e-06)##########################################################
        self.G_logit = layers_robustseg.general_conv3d(in_ch=16, out_ch=1, k_size=7, stride=1, padding=3, pad_type='reflect', norm=None, act_type=None)
        self.UpSampling3D = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.relu = nn.ReLU()

    def forward(self, style, content):   #stlye:[B,8,1,1,1]
        mu, sigma = self.mlp(style)      #[B,128,1,1,1],[B,128,1,1,1]
        x = content                      #[B,128,10,10,10]

        x = self.res_0(x, self.channel, mu, sigma)   #[B,128,10,10,10]
        x = self.res_1(x, self.channel, mu, sigma)   #[B,128,10,10,10]
        x = self.res_2(x, self.channel, mu, sigma)   #[B,128,10,10,10]
        x = self.res_3(x, self.channel, mu, sigma)   #[B,128,10,10,10]

        x = self.UpSampling3D(x)        #[B,128,20,20,20]
        x = self.conv_0(x)              #[B,64,20,20,20]
        x = self.layer_norm_1(x)        #[B,64,20,20,20]
        x = self.relu(x)                #[B,64,20,20,20]

        x = self.UpSampling3D(x)        #[B,64,40,40,40]
        x = self.conv_1(x)              #[B,32,40,40,40]
        x = self.layer_norm_2(x)        #[B,32,40,40,40]
        x = self.relu(x)                #[B,32,40,40,40]

        x = self.UpSampling3D(x)        #[B,32,80,80,80]
        x = self.conv_2(x)              #[B,16,80,80,80]
        x = self.layer_norm_3(x)        #[B,16,80,80,80]
        x = self.relu(x)                #[B,16,80,80,80]

        x = self.G_logit(x)             #[B,1,80,80,80]
        return x, mu, sigma               #[B,1,80,80,80],[B,128,1,1,1],[B,128,1,1,1]

class mask_decoder(nn.Module):
    def __init__(self) -> None:
        super(mask_decoder, self).__init__()
        self.UpSampling3D = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = layers_robustseg.general_conv3d(in_ch=128, out_ch=n_base_filters*4, k_size=3, stride=1, padding=1, pad_type='reflect')
        self.d3_c2 = layers_robustseg.general_conv3d(in_ch=128, out_ch=n_base_filters*4, k_size=3, stride=1, padding=1, pad_type='reflect')
        self.d3_out = layers_robustseg.general_conv3d(in_ch=n_base_filters*4, out_ch=n_base_filters*4, k_size=1, stride=1, padding=0, pad_type='reflect')

        self.d2_c1 = layers_robustseg.general_conv3d(in_ch=64, out_ch=n_base_filters*2, k_size=3, stride=1, padding=1, pad_type='reflect')
        self.d2_c2 = layers_robustseg.general_conv3d(in_ch=64, out_ch=n_base_filters*2, k_size=3, stride=1, padding=1, pad_type='reflect')
        self.d2_out = layers_robustseg.general_conv3d(in_ch=n_base_filters*2, out_ch=n_base_filters*2, k_size=1, stride=1, padding=0, pad_type='reflect')

        self.d1_c1 = layers_robustseg.general_conv3d(in_ch=32, out_ch=n_base_filters, k_size=3, stride=1, padding=1, pad_type='reflect')
        self.d1_c2 = layers_robustseg.general_conv3d(in_ch=32, out_ch=n_base_filters, k_size=3, stride=1, padding=1, pad_type='reflect')
        self.d1_out = layers_robustseg.general_conv3d(in_ch=n_base_filters, out_ch=n_base_filters, k_size=1, stride=1, padding=0, pad_type='reflect')

        self.seg_logit = layers_robustseg.general_conv3d(in_ch=n_base_filters, out_ch=num_cls, k_size=1, stride=1, padding=0, norm=None, act_type=None)
        self.seg_pred = nn.Softmax(dim=1)

    def forward(self,input):
        e4_out = input['e4_out']    #[B,128,10,10,10]
        e3_out = input['e3_out']    #[B,64,20,20,20]
        e2_out = input['e2_out']    #[B,32,40,40,40]
        e1_out = input['e1_out']    #[B,16,80,80,80]

        d3 = self.UpSampling3D(e4_out)              #[B,128,20,20,20]
        d3_c1 = self.d3_c1(d3)                      #[B,64,20,20,20]
        d3_cat = torch.cat([d3_c1,e3_out], dim=1)    #[B,128,20,20,20]
        d3_c2 = self.d3_c2(d3_cat)                  #[B,64,20,20,20]
        d3_out = self.d3_out(d3_c2)                 #[B,64,20,20,20]

        d2 = self.UpSampling3D(d3_out)              #[B,64,40,40,40]
        d2_c1 = self.d2_c1(d2)                      #[B,32,40,40,40]
        d2_cat = torch.cat([d2_c1,e2_out], dim=1)    #[B,64,40,40,40]
        d2_c2 = self.d2_c2(d2_cat)                  #[B,32,40,40,40]
        d2_out = self.d2_out(d2_c2)                 #[B,32,40,40,40]

        d1 = self.UpSampling3D(d2_out)              #[B,32,80,80,80]
        d1_c1 = self.d1_c1(d1)                      #[B,16,80,80,80]
        d1_cat = torch.cat([d1_c1,e1_out], dim=1)    #[B,32,80,80,80]
        d1_c2 = self.d1_c2(d1_cat)                  #[B,16,80,80,80]
        d1_out = self.d1_out(d1_c2)                 #[B,16,80,80,80]

        seg_logit = self.seg_logit(d1_out)          #[B,4,80,80,80]
        seg_pred = self.seg_pred(seg_logit)         #[B,4,80,80,80]

        return seg_pred, seg_logit   #[B,4,80,80,80] #[B,4,80,80,80]

class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.channel = mlp_ch
        self.linear_0 = layers_robustseg.linear(8, self.channel)
        self.linear_1 = layers_robustseg.linear(self.channel, self.channel)
        self.mu = layers_robustseg.linear(self.channel, self.channel)
        self.sigma = layers_robustseg.linear(self.channel, self.channel)
        self.relu = nn.ReLU()

    def forward(self,style):        #[B,8,1,1,1]
        x = self.linear_0(style)    #[B,128]
        x = self.relu(x)

        x = self.linear_1(x)        #[B,128]
        x = self.relu(x)

        mu = self.mu(x)             #[B,128]
        sigma = self.sigma(x)       #[B,128]

        mu = mu.view([x.shape[0], self.channel, 1, 1, 1])       #[B,128,1,1,1]
        sigma = sigma.view([x.shape[0], self.channel, 1, 1, 1]) #[B,128,1,1,1]

        return mu, sigma

class MaskModal(nn.Module):
    def __init__(self):
        super(MaskModal, self).__init__()
    
    def forward(self, x, mask):
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        return y
    
class RobustSeg(nn.Module):
    def __init__(self, num_cls=4):
        super(RobustSeg, self).__init__()
        self.se_flair = style_encoder()
        self.se_t1c = style_encoder()
        self.se_t1 = style_encoder()
        self.se_t2 = style_encoder()

        self.ce_flair = content_encoder()
        self.ce_t1c = content_encoder()
        self.ce_t1 = content_encoder()
        self.ce_t2 = content_encoder()

        self.masker = MaskModal()

        self.att_c1 = layers_robustseg.general_conv3d(in_ch=64, out_ch=4, k_size=3, stride=1, padding=1, pad_type='reflect')
        self.att_c2 = layers_robustseg.general_conv3d(in_ch=128, out_ch=4, k_size=3, stride=1, padding=1, pad_type='reflect')
        self.att_c3 = layers_robustseg.general_conv3d(in_ch=256, out_ch=4, k_size=3, stride=1, padding=1, pad_type='reflect')
        self.att_c4 = layers_robustseg.general_conv3d(in_ch=512, out_ch=4, k_size=3, stride=1, padding=1, pad_type='reflect')

        self.fusion_c1 = layers_robustseg.general_conv3d(in_ch=64, out_ch=n_base_filters, k_size=1, stride=1, padding=0, pad_type='reflect')
        self.fusion_c2 = layers_robustseg.general_conv3d(in_ch=128, out_ch=n_base_filters*2, k_size=1, stride=1, padding=0, pad_type='reflect')
        self.fusion_c3 = layers_robustseg.general_conv3d(in_ch=256, out_ch=n_base_filters*4, k_size=1, stride=1, padding=0, pad_type='reflect')
        self.fusion_c4 = layers_robustseg.general_conv3d(in_ch=512, out_ch=n_base_filters*8, k_size=1, stride=1, padding=0, pad_type='reflect')

        self.image_de_flair = image_decoder()
        self.image_de_t1c = image_decoder()
        self.image_de_t1 = image_decoder()
        self.image_de_t2 = image_decoder()

        self.mask_de = mask_decoder()

        self.Sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) 
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight) 
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight) 
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight) 

    def forward(self, x, mask, is_training=True):#x[B,4,80,80,80]
        # appereance code
        style_flair = self.se_flair(x[:,0:1,:,:,:]) #[B,8,1,1,1]
        style_t1c = self.se_t1c(x[:,1:2,:,:,:])     #[B,8,1,1,1]
        style_t1 = self.se_t1(x[:,2:3,:,:,:])       #[B,8,1,1,1]
        style_t2 = self.se_t2(x[:,3:4,:,:,:])       #[B,8,1,1,1]
        
        # content code at 4 scales
        content_flair = self.ce_flair(x[:,0:1,:,:,:])   #([B,16,80,80,80],[B,32,40,40,40],[B,64,20,20,20],[B,128,10,10,10])
        content_t1c = self.ce_t1c(x[:,1:2,:,:,:])       #([B,16,80,80,80],[B,32,40,40,40],[B,64,20,20,20],[B,128,10,10,10])
        content_t1 = self.ce_t1(x[:,2:3,:,:,:])         #([B,16,80,80,80],[B,32,40,40,40],[B,64,20,20,20],[B,128,10,10,10])
        content_t2 = self.ce_t2(x[:,3:4,:,:,:])         #([B,16,80,80,80],[B,32,40,40,40],[B,64,20,20,20],[B,128,10,10,10])

        #MaskModalities
        ones_s1 = torch.ones([x.shape[0], 4, 16, 80, 80, 80]).cuda()             #[B,4,16,80,80,80]
        mask_s1 = torch.zeros([x.shape[0], 4, 16, 80, 80, 80]).cuda()            #[B,4,16,80,80,80]
        mask_s1[mask,...] = ones_s1[mask,...]                               #[B,4,16,80,80,80]
        
        ones_s2 = torch.ones([x.shape[0], 4, 32, 40, 40, 40]).cuda()             #[B,4,32,40,40,40]
        mask_s2 = torch.zeros([x.shape[0], 4, 32, 40, 40, 40]).cuda()            #[B,4,32,40,40,40]
        mask_s2[mask,...] = ones_s2[mask,...]                               #[B,4,32,40,40,40]
        
        ones_s3 = torch.ones([x.shape[0], 4, 64, 20, 20, 20]).cuda()             #[B,4,64,20,20,20]
        mask_s3 = torch.zeros([x.shape[0], 4, 64, 20, 20, 20]).cuda()            #[B,4,64,20,20,20]
        mask_s3[mask,...] = ones_s3[mask,...]                               #[B,4,64,20,20,20]
        
        ones_s4 = torch.ones([x.shape[0], 4, 128, 10, 10, 10]).cuda()            #[B,4,128,10,10,10]
        mask_s4 = torch.zeros([x.shape[0], 4, 128, 10, 10, 10]).cuda()           #[B,4,128,10,10,10]
        mask_s4[mask,...] = ones_s4[mask,...]                               #[B,4,128,10,10,10]

        content_flair_s1 = torch.mul(content_flair['s1'], mask_s1[:,0,...])   #[B,16,80,80,80]
        content_flair_s2 = torch.mul(content_flair['s2'], mask_s2[:,0,...])   #[B,32,40,40,40]
        content_flair_s3 = torch.mul(content_flair['s3'], mask_s3[:,0,...])   #[B,64,20,20,20]
        content_flair_s4 = torch.mul(content_flair['s4'], mask_s4[:,0,...])   #[B,128,10,10,10]

        content_t1c_s1 = torch.mul(content_t1c['s1'], mask_s1[:,1,...])       #[B,16,80,80,80]
        content_t1c_s2 = torch.mul(content_t1c['s2'], mask_s2[:,1,...])       #[B,32,40,40,40]
        content_t1c_s3 = torch.mul(content_t1c['s3'], mask_s3[:,1,...])       #[B,64,20,20,20]
        content_t1c_s4 = torch.mul(content_t1c['s4'], mask_s4[:,1,...])       #[B,128,10,10,10]

        content_t1_s1 = torch.mul(content_t1['s1'], mask_s1[:,2,...])         #[B,16,80,80,80]
        content_t1_s2 = torch.mul(content_t1['s2'], mask_s2[:,2,...])         #[B,32,40,40,40]
        content_t1_s3 = torch.mul(content_t1['s3'], mask_s3[:,2,...])         #[B,64,20,20,20]
        content_t1_s4 = torch.mul(content_t1['s4'], mask_s4[:,2,...])         #[B,128,10,10,10]

        content_t2_s1 = torch.mul(content_t2['s1'], mask_s1[:,3,...])         #[B,16,80,80,80]
        content_t2_s2 = torch.mul(content_t2['s2'], mask_s2[:,3,...])         #[B,32,40,40,40]
        content_t2_s3 = torch.mul(content_t2['s3'], mask_s3[:,3,...])         #[B,64,20,20,20]
        content_t2_s4 = torch.mul(content_t2['s4'], mask_s4[:,3,...])         #[B,128,10,10,10]
        ### 

        #Gated-Fusion at each scale
        content_share_c1_concat = torch.cat([content_flair_s1, content_t1c_s1, content_t1_s1, content_t2_s1], dim=1)    #[B,64,80,80,80]
        content_share_c1_attmap = self.att_c1(content_share_c1_concat)                              #[B,4,80,80,80]
        content_share_c1_attmap = self.Sigmoid(content_share_c1_attmap)                             #[B,4,80,80,80]
        content_share_c1 = torch.cat([
            torch.mul(content_flair_s1, content_share_c1_attmap[:,0:1,:,:,:].repeat(1,16,1,1,1)),    #[B,16,80,80,80]
            torch.mul(content_t1c_s1, content_share_c1_attmap[:,1:2,:,:,:].repeat(1,16,1,1,1)),      #[B,16,80,80,80]
            torch.mul(content_t1_s1, content_share_c1_attmap[:,2:3,:,:,:].repeat(1,16,1,1,1)),       #[B,16,80,80,80]
            torch.mul(content_t2_s1, content_share_c1_attmap[:,3:4,:,:,:].repeat(1,16,1,1,1))        #[B,16,80,80,80]
        ], dim=1)            #[B,64,80,80,80]

        content_share_c2_concat = torch.cat([content_flair_s2, content_t1c_s2, content_t1_s2, content_t2_s2], dim=1)    #[B,128,40,40,40]
        content_share_c2_attmap = self.att_c2(content_share_c2_concat)                              #[B,4,40,40,40]                                        
        content_share_c2_attmap = self.Sigmoid(content_share_c2_attmap)                             #[B,4,40,40,40]
        content_share_c2 = torch.cat([
            torch.mul(content_flair_s2, content_share_c2_attmap[:,0:1,:,:,:].repeat(1,32,1,1,1)),    #[B,32,40,40,40]     
            torch.mul(content_t1c_s2 ,content_share_c2_attmap[:,1:2,:,:,:].repeat(1,32,1,1,1)),      #[B,32,40,40,40]
            torch.mul(content_t1_s2, content_share_c2_attmap[:,2:3,:,:,:].repeat(1,32,1,1,1)),       #[B,32,40,40,40]
            torch.mul(content_t2_s2, content_share_c2_attmap[:,3:4,:,:,:].repeat(1,32,1,1,1))        #[B,32,40,40,40]
        ],dim=1)            #[B,128,40,40,40]

        content_share_c3_concat = torch.cat([content_flair_s3, content_t1c_s3, content_t1_s3, content_t2_s3], dim=1)    #[B,256,20,20,20]
        content_share_c3_attmap = self.att_c3(content_share_c3_concat)        #[B,4,20,20,20]        
        content_share_c3_attmap = self.Sigmoid(content_share_c3_attmap)                   #[B,4,20,20,20] 
        content_share_c3 = torch.cat([
            torch.mul(content_flair_s3, content_share_c3_attmap[:,0:1,:,:,:].repeat(1,64,1,1,1)),    #[B,64,20,20,20]
            torch.mul(content_t1c_s3, content_share_c3_attmap[:,1:2,:,:,:].repeat(1,64,1,1,1)),      #[B,64,20,20,20]
            torch.mul(content_t1_s3, content_share_c3_attmap[:,2:3,:,:,:].repeat(1,64,1,1,1)),       #[B,64,20,20,20]
            torch.mul(content_t2_s3, content_share_c3_attmap[:,3:4,:,:,:].repeat(1,64,1,1,1))        #[B,64,20,20,20]
        ],dim=1)            #[B,256,20,20,20]

        content_share_c4_concat = torch.cat([content_flair_s4, content_t1c_s4, content_t1_s4, content_t2_s4], dim=1)    #[B,512,10,10,10]
        content_share_c4_attmap = self.att_c4(content_share_c4_concat)        #[B,4,10,10,10]
        content_share_c4_attmap = self.Sigmoid(content_share_c4_attmap)                   #[B,4,10,10,10]
        content_share_c4 = torch.cat([
            torch.mul(content_flair_s4, content_share_c4_attmap[:,0:1,:,:,:].repeat(1,128,1,1,1)),   #[B,128,10,10,10]
            torch.mul(content_t1c_s4, content_share_c4_attmap[:,1:2,:,:,:].repeat(1,128,1,1,1)),     #[B,128,10,10,10]
            torch.mul(content_t1_s4, content_share_c4_attmap[:,2:3,:,:,:].repeat(1,128,1,1,1)),      #[B,128,10,10,10]
            torch.mul(content_t2_s4, content_share_c4_attmap[:,3:4,:,:,:].repeat(1,128,1,1,1))       #[B,128,10,10,10]
        ],dim=1)            #[B,512,10,10,10]

        content_share_c1 = self.fusion_c1(content_share_c1)     #[B,16,80,80,80]
        content_share_c2 = self.fusion_c2(content_share_c2)     #[B,32,40,40,40]
        content_share_c3 = self.fusion_c3(content_share_c3)     #[B,64,20,20,20]
        content_share_c4 = self.fusion_c4(content_share_c4)     #[B,128,10,10,10]

        if is_training:
            #reconstruction
            reconstruct_flair, mu_flair, sigma_flair = self.image_de_flair(style_flair, content_share_c4)
            reconstruct_t1c, mu_t1c, sigma_t1c = self.image_de_t1c(style_t1c, content_share_c4)
            reconstruct_t1, mu_t1, sigma_t1 = self.image_de_t1(style_t1, content_share_c4)
            reconstruct_t2, mu_t2, sigma_t2 = self.image_de_t2(style_t2, content_share_c4)
            #[B,1,80,80,80][B,128,1,1,1][B,128,1,1,1]

            #segmentation
            mask_de_input = {
                    'e1_out': content_share_c1, #[B,16,80,80,80]
                    'e2_out': content_share_c2, #[B,32,40,40,40]
                    'e3_out': content_share_c3, #[B,64,20,20,20]
                    'e4_out': content_share_c4, #[B,128,10,10,10]
                }
            
            seg_pred, seg_logit = self.mask_de(mask_de_input)
            #[B,4,80,80,80][B,4,80,80,80]
            
            return {
                'style_flair': style_flair,     #[B,8,1,1,1]
                'style_t1c': style_t1c,         #[B,8,1,1,1]
                'style_t1': style_t1,           #[B,8,1,1,1]
                'style_t2': style_t2,           #[B,8,1,1,1]
                'content_flair': content_flair, #([B,16,80,80,80],[B,32,40,40,40],[B,64,20,20,20],[B,128,10,10,10])
                'content_t1c': content_t1c,     #([B,16,80,80,80],[B,32,40,40,40],[B,64,20,20,20],[B,128,10,10,10]) 
                'content_t1': content_t1,       #([B,16,80,80,80],[B,32,40,40,40],[B,64,20,20,20],[B,128,10,10,10])
                'content_t2': content_t2,       #([B,16,80,80,80],[B,32,40,40,40],[B,64,20,20,20],[B,128,10,10,10])
                'mu_flair': mu_flair,           #[B,128,1,1,1]
                'mu_t1c': mu_t1c,               #[B,128,1,1,1]
                'mu_t1': mu_t1,                 #[B,128,1,1,1]
                'mu_t2': mu_t2,                 #[B,128,1,1,1]
                'sigma_flair': sigma_flair,     #[B,128,1,1,1]
                'sigma_t1c': sigma_t1c,         #[B,128,1,1,1]
                'sigma_t1': sigma_t1,           #[B,128,1,1,1]
                'sigma_t2': sigma_t2,           #[B,128,1,1,1]
                'reconstruct_flair': reconstruct_flair, #[B,1,80,80,80]
                'reconstruct_t1c': reconstruct_t1c,     #[B,1,80,80,80]
                'reconstruct_t1': reconstruct_t1,       #[B,1,80,80,80]
                'reconstruct_t2': reconstruct_t2,       #[B,1,80,80,80]
                'seg_pred': seg_pred,           #[B,4,80,80,80]
                'seg_logit': seg_logit,         #[B,4,80,80,80]
            }
        else:
            #segmentation
            mask_de_input = {
                    'e1_out': content_share_c1, #[B,16,80,80,80]
                    'e2_out': content_share_c2, #[B,32,40,40,40]
                    'e3_out': content_share_c3, #[B,64,20,20,20]
                    'e4_out': content_share_c4, #[B,128,10,10,10]
                }
            seg_pred, seg_logit = self.mask_de(mask_de_input)
            #[B,4,80,80,80][B,4,80,80,80]
            return seg_pred                     #[B,4,80,80,80]