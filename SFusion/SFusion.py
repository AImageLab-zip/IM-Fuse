
import torch
import torch.nn as nn

from layers_sfusion import style_encoder, content_encoder, image_decoder, \
    general_conv3d, mask_decoder, TF_3D

def check_values(output, name):
    #check for NaN of Inf values
    if torch.isnan(output).any():
        print(f"NaN detected in {name}!")
    if torch.isinf(output).any():
        print(f"Inf detected in {name}!")

class TF_RMBTS(nn.Module): #TF_RMBTS = SF_FDGF
    def __init__(self, in_channels=None, out_channels=None, levels=None, feature_maps=None, method=None, phase='train'):
        super(TF_RMBTS, self).__init__()
        n_base_filters = feature_maps
        n_base_ch_se = 32
        mlp_ch = feature_maps * (2 ** (levels-1))
        img_ch = 1
        scale = levels
        self.style_encoder_t1___ = style_encoder(img_ch, n_base_ch_se=n_base_ch_se)
        self.style_encoder_t1ce_ = style_encoder(img_ch, n_base_ch_se=n_base_ch_se)
        self.style_encoder_t2___ = style_encoder(img_ch, n_base_ch_se=n_base_ch_se)
        self.style_encoder_flair = style_encoder(img_ch, n_base_ch_se=n_base_ch_se)

        self.content_encoder_t1___ = content_encoder(img_ch, n_base_filters=n_base_filters)
        self.content_encoder_t1ce_ = content_encoder(img_ch, n_base_filters=n_base_filters)
        self.content_encoder_t2___ = content_encoder(img_ch, n_base_filters=n_base_filters)
        self.content_encoder_flair = content_encoder(img_ch, n_base_filters=n_base_filters)

        self.image_decoder_t1___ = image_decoder(n_base_filters*8, mlp_ch, img_ch, scale)
        self.image_decoder_t1ce_ = image_decoder(n_base_filters*8, mlp_ch, img_ch, scale)
        self.image_decoder_t2___ = image_decoder(n_base_filters*8, mlp_ch, img_ch, scale)
        self.image_decoder_flair = image_decoder(n_base_filters*8, mlp_ch, img_ch, scale)

        self.fusion1 = TF_3D(embedding_dim=n_base_filters, volumn_size=128, method=method)
        self.fusion2 = TF_3D(embedding_dim=n_base_filters*2, volumn_size=64, method=method)
        self.fusion3 = TF_3D(embedding_dim=n_base_filters*4, volumn_size=32, method=method)
        self.fusion4 = TF_3D(embedding_dim=n_base_filters*8, volumn_size=16, method=method)

        self.sigmoid = nn.Sigmoid()
        #self.miss_list = missing_list()

        self.mask_decoder = mask_decoder(input_channel=n_base_filters*8,
                                         n_base_filters=n_base_filters,
                                         num_cls=4)

    def forward(self, input, m_d, is_training=True):
        #flair, t1ce, t1, t2
        image_t1ce_ = input[:,1:2,:,:,:] #input[0] -> (B, 128, 128, 128)
        image_t1___ = input[:,2:3,:,:,:] #input[1] -> (B, 128, 128, 128)
        image_t2___ = input[:,3:4,:,:,:] #input[2] -> (B, 128, 128, 128)
        image_flair = input[:,0:1,:,:,:] #input[3] -> (B, 128, 128, 128)
        check_values(image_t1ce_, name='image_t1ce_')
        check_values(image_t1___, name='image_t1___')
        check_values(image_t2___, name='image_t2___')
        check_values(image_flair, name='image_flair')

        style_t1___ = self.style_encoder_t1___(image_t1___) #(B, 8, 1, 1, 1)
        style_t1ce_ = self.style_encoder_t1ce_(image_t1ce_) #(B, 8, 1, 1, 1)
        style_t2___ = self.style_encoder_t2___(image_t2___) #(B, 8, 1, 1, 1)
        style_flair = self.style_encoder_flair(image_flair) #(B, 8, 1, 1, 1)

        content_t1___ = self.content_encoder_t1___(image_t1___) 
        content_t1ce_ = self.content_encoder_t1ce_(image_t1ce_)
        content_t2___ = self.content_encoder_t2___(image_t2___)
        content_flair = self.content_encoder_flair(image_flair)

        content_share_c1 = []
        content_share_c2 = [] 
        content_share_c3 = [] 
        content_share_c4 = [] 

        if m_d[0][0]: #t1c #m_d in self.miss_list[0]:
            content_share_c1.append(content_t1ce_['s1']) #(B, 8, 128, 128, 128)
            content_share_c2.append(content_t1ce_['s2']) #(B, 16, 64, 64, 64)
            content_share_c3.append(content_t1ce_['s3']) #(B, 32, 32, 32, 32)
            content_share_c4.append(content_t1ce_['s4']) #(B, 64, 16, 16, 16)
            check_values(content_t1ce_['s1'], name='content_t1ce_[s1]')
            check_values(content_t1ce_['s2'], name='content_t1ce_[s2]')
            check_values(content_t1ce_['s3'], name='content_t1ce_[s3]')
            check_values(content_t1ce_['s4'], name='content_t1ce_[s4]')

        if m_d[0][1]: #t1  #m_d in self.miss_list[1]:
            content_share_c1.append(content_t1___['s1'])
            content_share_c2.append(content_t1___['s2'])
            content_share_c3.append(content_t1___['s3'])
            content_share_c4.append(content_t1___['s4'])
            check_values(content_t1___['s1'], name='content_t1___[s1]')
            check_values(content_t1___['s2'], name='content_t1___[s2]')
            check_values(content_t1___['s3'], name='content_t1___[s3]')
            check_values(content_t1___['s4'], name='content_t1___[s4]')

        if m_d[0][2]: #t2 #if m_d in self.miss_list[2]:
            content_share_c1.append(content_t2___['s1'])
            content_share_c2.append(content_t2___['s2'])
            content_share_c3.append(content_t2___['s3'])
            content_share_c4.append(content_t2___['s4'])
            check_values(content_t2___['s1'], name='content_t2___[s1]')
            check_values(content_t2___['s2'], name='content_t2___[s2]')
            check_values(content_t2___['s3'], name='content_t2___[s3]')
            check_values(content_t2___['s4'], name='content_t2___[s4]')

        if m_d[0][3]: #flair #if m_d in self.miss_list[3]:
            content_share_c1.append(content_flair['s1'])
            content_share_c2.append(content_flair['s2'])
            content_share_c3.append(content_flair['s3'])
            content_share_c4.append(content_flair['s4'])
            check_values(content_flair['s1'], name='content_flair[s1]')
            check_values(content_flair['s2'], name='content_flair[s2]')
            check_values(content_flair['s3'], name='content_flair[s3]')
            check_values(content_flair['s4'], name='content_flair[s4]')

        content_share_c1 = self.fusion1(content_share_c1) #(B, 8, 128, 128, 128)
        content_share_c2 = self.fusion2(content_share_c2) #(B, 16, 64, 64, 64)
        content_share_c3 = self.fusion3(content_share_c3) #(B, 32, 32, 32, 32)
        content_share_c4 = self.fusion4(content_share_c4) #(B, 64, 16, 16, 16)
        check_values(content_share_c1, name='content_share_c1')
        check_values(content_share_c2, name='content_share_c2')
        check_values(content_share_c3, name='content_share_c3')
        check_values(content_share_c4, name='content_share_c4')
        
        if is_training:
            #reconstruction
            reconstruct_t1___, mu_t1___, sigma_t1___ = self.image_decoder_t1___(style_t1___, content_share_c4) 
            reconstruct_t1ce_, mu_t1ce_, sigma_t1ce_ = self.image_decoder_t1ce_(style_t1ce_, content_share_c4) 
            reconstruct_t2___, mu_t2___, sigma_t2___ = self.image_decoder_t2___(style_t2___, content_share_c4) 
            reconstruct_flair, mu_flair, sigma_flair = self.image_decoder_flair(style_flair, content_share_c4) 
            #(B, 1, 128, 128, 128),(B, 64, 1, 1, 1),(B, 64, 1, 1, 1)
            mask_de_input = {
                'e1_out': content_share_c1, #(B, 8, 128, 128, 128)
                'e2_out': content_share_c2, #(B, 16, 64, 64, 64)
                'e3_out': content_share_c3, #(B, 32, 32, 32, 32)
                'e4_out': content_share_c4, #(B, 64, 16, 16, 16)
            }

            seg_logit, seg_pred = self.mask_decoder(mask_de_input)
            #(B, 4, 128, 128, 128),(B, 4, 128, 128, 128)
            return {
                'style_flair': style_flair,         #[B,8,1,1,1]
                'style_t1c__': style_t1ce_,         #[B,8,1,1,1]
                'style_t1___': style_t1___,         #[B,8,1,1,1]
                'style_t2___': style_t2___,         #[B,8,1,1,1]
                'content_flair': content_flair,     #[B,8,128,128,128],[B,16,64,64,64],[B,32,32,32,32],[B,64,16,16,16]
                'content_t1c__': content_t1ce_,     #[B,8,128,128,128],[B,16,64,64,64],[B,32,32,32,32],[B,64,16,16,16]
                'content_t1___': content_t1___,     #[B,8,128,128,128],[B,16,64,64,64],[B,32,32,32,32],[B,64,16,16,16]
                'content_t2___': content_t2___,     #[B,8,128,128,128],[B,16,64,64,64],[B,32,32,32,32],[B,64,16,16,16]
                'mu_flair': mu_flair,               #[B,64,1,1,1]
                'mu_t1c__': mu_t1ce_,               #[B,64,1,1,1]
                'mu_t1___': mu_t1___,               #[B,64,1,1,1]
                'mu_t2___': mu_t2___,               #[B,64,1,1,1]
                'sigma_flair': sigma_flair,         #[B,64,1,1,1]
                'sigma_t1c__': sigma_t1ce_,         #[B,64,1,1,1]
                'sigma_t1___': sigma_t1___,         #[B,64,1,1,1]
                'sigma_t2___': sigma_t2___,         #[B,64,1,1,1]
                'reconstruct_flair': reconstruct_flair, #[B,1,128,128,128]
                'reconstruct_t1c__': reconstruct_t1ce_, #[B,1,128,128,128]
                'reconstruct_t1___': reconstruct_t1___, #[B,1,128,128,128]
                'reconstruct_t2___': reconstruct_t2___, #[B,1,128,128,128]
                'seg': seg_pred,                    #[B,4,128,128,128]
            }
    
        else:
            #segmentation
            mask_de_input = {
                'e1_out': content_share_c1, #(B, 8, 128, 128, 128)
                'e2_out': content_share_c2, #(B, 16, 64, 64, 64)
                'e3_out': content_share_c3, #(B, 32, 32, 32, 32)
                'e4_out': content_share_c4, #(B, 64, 16, 16, 16)
            }

            seg_logit, seg_pred = self.mask_decoder(mask_de_input)

            return seg_pred                 
