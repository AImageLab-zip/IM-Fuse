
import torch
import torch.nn as nn

from net.BasicBlock import style_encoder, content_encoder, image_decoder, \
    general_conv3d, mask_decoder, TF_3D
from process.utils import missing_list


class RMBTS(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, levels=None, feature_maps=None, method=None, phase='train'):
        super(RMBTS, self).__init__()
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

        self.attm1 = general_conv3d(n_base_filters*4, 4)
        self.attm2 = general_conv3d(n_base_filters*2*4, 4)
        self.attm3 = general_conv3d(n_base_filters*4*4, 4)
        self.attm4 = general_conv3d(n_base_filters*8*4, 4)

        self.fusion1 = general_conv3d(n_base_filters*4, n_base_filters, kernel_size=1)
        self.fusion2 = general_conv3d(n_base_filters*2*4, n_base_filters*2, kernel_size=1)
        self.fusion3 = general_conv3d(n_base_filters*4*4, n_base_filters*4, kernel_size=1)
        self.fusion4 = general_conv3d(n_base_filters*8*4, n_base_filters*8, kernel_size=1)

        self.sigmoid = nn.Sigmoid()
        self.miss_list = missing_list()

        self.mask_decoder = mask_decoder(input_channel=n_base_filters*8,
                                         n_base_filters=n_base_filters,
                                         num_cls=4)

    def forward(self, input, m_d):
        image_t1ce_ = input[0]
        image_t1___ = input[1]
        image_t2___ = input[2]
        image_flair = input[3]

        style_t1___ = self.style_encoder_t1___(image_t1___)
        style_t1ce_ = self.style_encoder_t1ce_(image_t1ce_)
        style_t2___ = self.style_encoder_t2___(image_t2___)
        style_flair = self.style_encoder_flair(image_flair)

        content_t1___ = self.content_encoder_t1___(image_t1___)
        content_t1ce_ = self.content_encoder_t1ce_(image_t1ce_)
        content_t2___ = self.content_encoder_t2___(image_t2___)
        content_flair = self.content_encoder_flair(image_flair)

        flag_t1ce = (1.0 if m_d in self.miss_list[0] else 0.0)
        content_t1ce__s1 = content_t1ce_['s1'] * flag_t1ce
        content_t1ce__s2 = content_t1ce_['s2'] * flag_t1ce
        content_t1ce__s3 = content_t1ce_['s3'] * flag_t1ce
        content_t1ce__s4 = content_t1ce_['s4'] * flag_t1ce

        flag_t1 = (1.0 if m_d in self.miss_list[1] else 0.0)
        content_t1____s1 = content_t1___['s1'] * flag_t1
        content_t1____s2 = content_t1___['s2'] * flag_t1
        content_t1____s3 = content_t1___['s3'] * flag_t1
        content_t1____s4 = content_t1___['s4'] * flag_t1

        flag_t2 = (1.0 if m_d in self.miss_list[2] else 0.0)
        content_t2____s1 = content_t2___['s1'] * flag_t2
        content_t2____s2 = content_t2___['s2'] * flag_t2
        content_t2____s3 = content_t2___['s3'] * flag_t2
        content_t2____s4 = content_t2___['s4'] * flag_t2

        flag_flair = (1.0 if m_d in self.miss_list[3] else 0.0)
        content_flair_s1 = content_flair['s1'] * flag_flair
        content_flair_s2 = content_flair['s2'] * flag_flair
        content_flair_s3 = content_flair['s3'] * flag_flair
        content_flair_s4 = content_flair['s4'] * flag_flair

        content_share_c1_concat = torch.cat([content_t1ce__s1, content_t1____s1, content_t2____s1, content_flair_s1], dim=1)
        content_share_c1_attmap = self.attm1(content_share_c1_concat)
        content_share_c1_attmap = self.sigmoid(content_share_c1_attmap)
        content_share_c1 = torch.cat([
            content_t1ce__s1 * content_share_c1_attmap[:, 0, :, :, :],
            content_t1____s1 * content_share_c1_attmap[:, 1, :, :, :],
            content_t2____s1 * content_share_c1_attmap[:, 2, :, :, :],
            content_flair_s1 * content_share_c1_attmap[:, 3, :, :, :],
        ], dim=1)
        content_share_c1 = self.fusion1(content_share_c1)

        content_share_c2_concat = torch.cat([content_t1ce__s2, content_t1____s2, content_t2____s2, content_flair_s2],
                                            dim=1)
        content_share_c2_attmap = self.attm2(content_share_c2_concat)
        content_share_c2_attmap = self.sigmoid(content_share_c2_attmap)
        content_share_c2 = torch.cat([
            content_t1ce__s2 * content_share_c2_attmap[:, 0, :, :, :],
            content_t1____s2 * content_share_c2_attmap[:, 1, :, :, :],
            content_t2____s2 * content_share_c2_attmap[:, 2, :, :, :],
            content_flair_s2 * content_share_c2_attmap[:, 3, :, :, :],
        ], dim=1)
        content_share_c2 = self.fusion2(content_share_c2)

        content_share_c3_concat = torch.cat([content_t1ce__s3, content_t1____s3, content_t2____s3, content_flair_s3],
                                            dim=1)
        content_share_c3_attmap = self.attm3(content_share_c3_concat)
        content_share_c3_attmap = self.sigmoid(content_share_c3_attmap)
        content_share_c3 = torch.cat([
            content_t1ce__s3 * content_share_c3_attmap[:, 0, :, :, :],
            content_t1____s3 * content_share_c3_attmap[:, 1, :, :, :],
            content_t2____s3 * content_share_c3_attmap[:, 2, :, :, :],
            content_flair_s3 * content_share_c3_attmap[:, 3, :, :, :],
        ], dim=1)
        content_share_c3 = self.fusion3(content_share_c3)

        content_share_c4_concat = torch.cat([content_t1ce__s4, content_t1____s4, content_t2____s4, content_flair_s4],
                                            dim=1)
        content_share_c4_attmap = self.attm4(content_share_c4_concat)
        content_share_c4_attmap = self.sigmoid(content_share_c4_attmap)
        content_share_c4 = torch.cat([
            content_t1ce__s4 * content_share_c4_attmap[:, 0, :, :, :],
            content_t1____s4 * content_share_c4_attmap[:, 1, :, :, :],
            content_t2____s4 * content_share_c4_attmap[:, 2, :, :, :],
            content_flair_s4 * content_share_c4_attmap[:, 3, :, :, :],
        ], dim=1)
        content_share_c4 = self.fusion4(content_share_c4)

        reconstruct_t1___, mu_t1___, sigma_t1___ = self.image_decoder_t1___(style_t1___, content_share_c4)
        reconstruct_t1ce_, mu_t1ce_, sigma_t1ce_ = self.image_decoder_t1ce_(style_t1ce_, content_share_c4)
        reconstruct_t2___, mu_t2___, sigma_t2___ = self.image_decoder_t2___(style_t2___, content_share_c4)
        reconstruct_flair, mu_flair, sigma_flair = self.image_decoder_flair(style_flair, content_share_c4)

        mask_de_input = {
            'e1_out': content_share_c1,
            'e2_out': content_share_c2,
            'e3_out': content_share_c3,
            'e4_out': content_share_c4,
        }

        seg = self.mask_decoder(mask_de_input)

        return {
            'style_flair': style_flair,
            'style_t1c__': style_t1ce_,
            'style_t1___': style_t1___,
            'style_t2___': style_t2___,
            'content_flair': content_flair,
            'content_t1c__': content_t1ce_,
            'content_t1___': content_t1___,
            'content_t2___': content_t2___,
            'mu_flair': mu_flair,
            'mu_t1c__': mu_t1ce_,
            'mu_t1___': mu_t1___,
            'mu_t2___': mu_t2___,
            'sigma_flair': sigma_flair,
            'sigma_t1c__': sigma_t1ce_,
            'sigma_t1___': sigma_t1___,
            'sigma_t2___': sigma_t2___,
            'reconstruct_flair': reconstruct_flair,
            'reconstruct_t1c__': reconstruct_t1ce_,
            'reconstruct_t1___': reconstruct_t1___,
            'reconstruct_t2___': reconstruct_t2___,
            'seg': seg,
        }




class TF_RMBTS(nn.Module):
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
        self.miss_list = missing_list()

        self.mask_decoder = mask_decoder(input_channel=n_base_filters*8,
                                         n_base_filters=n_base_filters,
                                         num_cls=4)

    def forward(self, input, m_d):
        image_t1ce_ = input[0]
        image_t1___ = input[1]
        image_t2___ = input[2]
        image_flair = input[3]

        style_t1___ = self.style_encoder_t1___(image_t1___)
        style_t1ce_ = self.style_encoder_t1ce_(image_t1ce_)
        style_t2___ = self.style_encoder_t2___(image_t2___)
        style_flair = self.style_encoder_flair(image_flair)

        content_t1___ = self.content_encoder_t1___(image_t1___)
        content_t1ce_ = self.content_encoder_t1ce_(image_t1ce_)
        content_t2___ = self.content_encoder_t2___(image_t2___)
        content_flair = self.content_encoder_flair(image_flair)

        content_share_c1 = []
        content_share_c2 = []
        content_share_c3 = []
        content_share_c4 = []

        if m_d in self.miss_list[0]:
            content_share_c1.append(content_t1ce_['s1'])
            content_share_c2.append(content_t1ce_['s2'])
            content_share_c3.append(content_t1ce_['s3'])
            content_share_c4.append(content_t1ce_['s4'])

        if m_d in self.miss_list[1]:
            content_share_c1.append(content_t1___['s1'])
            content_share_c2.append(content_t1___['s2'])
            content_share_c3.append(content_t1___['s3'])
            content_share_c4.append(content_t1___['s4'])

        if m_d in self.miss_list[2]:
            content_share_c1.append(content_t2___['s1'])
            content_share_c2.append(content_t2___['s2'])
            content_share_c3.append(content_t2___['s3'])
            content_share_c4.append(content_t2___['s4'])

        if m_d in self.miss_list[3]:
            content_share_c1.append(content_flair['s1'])
            content_share_c2.append(content_flair['s2'])
            content_share_c3.append(content_flair['s3'])
            content_share_c4.append(content_flair['s4'])

        content_share_c1 = self.fusion1(content_share_c1)
        content_share_c2 = self.fusion2(content_share_c2)
        content_share_c3 = self.fusion3(content_share_c3)
        content_share_c4 = self.fusion4(content_share_c4)

        reconstruct_t1___, mu_t1___, sigma_t1___ = self.image_decoder_t1___(style_t1___, content_share_c4)
        reconstruct_t1ce_, mu_t1ce_, sigma_t1ce_ = self.image_decoder_t1ce_(style_t1ce_, content_share_c4)
        reconstruct_t2___, mu_t2___, sigma_t2___ = self.image_decoder_t2___(style_t2___, content_share_c4)
        reconstruct_flair, mu_flair, sigma_flair = self.image_decoder_flair(style_flair, content_share_c4)

        mask_de_input = {
            'e1_out': content_share_c1,
            'e2_out': content_share_c2,
            'e3_out': content_share_c3,
            'e4_out': content_share_c4,
        }

        seg = self.mask_decoder(mask_de_input)

        return {
            'style_flair': style_flair,
            'style_t1c__': style_t1ce_,
            'style_t1___': style_t1___,
            'style_t2___': style_t2___,
            'content_flair': content_flair,
            'content_t1c__': content_t1ce_,
            'content_t1___': content_t1___,
            'content_t2___': content_t2___,
            'mu_flair': mu_flair,
            'mu_t1c__': mu_t1ce_,
            'mu_t1___': mu_t1___,
            'mu_t2___': mu_t2___,
            'sigma_flair': sigma_flair,
            'sigma_t1c__': sigma_t1ce_,
            'sigma_t1___': sigma_t1___,
            'sigma_t2___': sigma_t2___,
            'reconstruct_flair': reconstruct_flair,
            'reconstruct_t1c__': reconstruct_t1ce_,
            'reconstruct_t1___': reconstruct_t1___,
            'reconstruct_t2___': reconstruct_t2___,
            'seg': seg,
        }
