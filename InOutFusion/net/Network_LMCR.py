
import torch.nn as nn

from net.BasicBlock import LMCREncoder, LMCRDecoder, LMCR_Fusion, CR, ConvNormRelu3D, TF_3D, UNetEncoder, UNetDecoder
from process.utils import missing_list


class LMCR(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, levels=4, feature_maps=32, method=None, phase='train'):
        super(LMCR, self).__init__()
        self.feature_maps = feature_maps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.levels = levels
        self.miss_list = missing_list()

        self.encoders = nn.Sequential()
        self.decoders = nn.Sequential()
        for i in range(4):
            encoder = LMCREncoder(in_channels=self.in_channels, feature_maps=feature_maps, levels=self.levels)
            self.encoders.add_module('encoder%d' % (i), encoder)
            decoder = LMCRDecoder(out_channels=1, feature_maps=feature_maps, levels=self.levels, type='image')
            self.decoders.add_module('decoder%d' % (i), decoder)

        self.seg_decoder = LMCRDecoder(out_channels=self.out_channels , feature_maps=feature_maps, levels=self.levels, type='seg')

        self.fusion = LMCR_Fusion(feature_maps*(2**(self.levels-1)))
        self.CR = CR(feature_maps*(2**(self.levels-1)))
        self.skipfuion = nn.Sequential()
        for i in range(self.levels-1):
            skip_fusion = LMCR_Fusion(feature_maps*(2**i))
            self.skipfuion.add_module('skip_fusion%d' % (i), skip_fusion)

    def forward(self, inputs, m_d):
        all_encoder_levelputs = []
        all_encoder_outputs = []
        for k in range(len(inputs)):
            e, o = getattr(self.encoders, 'encoder%d' % (k))(inputs[k])
            all_encoder_levelputs.append(e)
            all_encoder_outputs.append(o)
        encoder_outputs = []
        for i in range(self.levels-1):
            e_output = []
            for k in range(len(all_encoder_outputs)):
                e_output.append(all_encoder_levelputs[k][i])
            encoder_outputs.append(getattr(self.skipfuion, 'skip_fusion%d' % (i))(e_output))

        output = self.CR(all_encoder_outputs)
        output = self.fusion(output)

        seg = self.seg_decoder(output, encoder_outputs)
        reconstruct_t1c__,reconstruct_t1___,reconstruct_t2___,reconstruct_flair = None,None,None,None
        if m_d in self.miss_list[0]:
            reconstruct_t1c__ = getattr(self.decoders, 'decoder0')(output, all_encoder_levelputs[0])
        if m_d in self.miss_list[1]:
            reconstruct_t1___ = getattr(self.decoders, 'decoder1')(output, all_encoder_levelputs[1])
        if m_d in self.miss_list[2]:
            reconstruct_t2___ = getattr(self.decoders, 'decoder2')(output, all_encoder_levelputs[2])
        if m_d in self.miss_list[3]:
            reconstruct_flair = getattr(self.decoders, 'decoder3')(output, all_encoder_levelputs[3])

        return {
            'reconstruct_flair': reconstruct_flair,
            'reconstruct_t1c__': reconstruct_t1c__,
            'reconstruct_t1___': reconstruct_t1___,
            'reconstruct_t2___': reconstruct_t2___,
            'seg': seg,
        }


class TF_LMCR(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, levels=4, feature_maps=32, method=None, phase='train'):
        super(TF_LMCR, self).__init__()
        self.feature_maps = feature_maps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.levels = levels
        self.miss_list = missing_list()

        self.encoders = nn.Sequential()
        self.decoders = nn.Sequential()
        for i in range(4):
            encoder = LMCREncoder(in_channels=self.in_channels, feature_maps=feature_maps, levels=self.levels)
            self.encoders.add_module('encoder%d' % (i), encoder)
            decoder = LMCRDecoder(out_channels=1, feature_maps=feature_maps, levels=self.levels, type='image')
            self.decoders.add_module('decoder%d' % (i), decoder)

        self.seg_decoder = LMCRDecoder(out_channels=self.out_channels, feature_maps=feature_maps, levels=self.levels,
                                       type='seg')


        self.fusion = TF_3D(embedding_dim=2 ** (self.levels - 1) * feature_maps,
                            volumn_size=(128 // (2 ** (self.levels - 1))), method=method)

        self.skipfuion = nn.Sequential()
        for i in range(self.levels - 1):
            skip_fusion = TF_3D(embedding_dim=(2 ** i) * feature_maps,
                                volumn_size=(128 // (2 ** i)), method=method)
            self.skipfuion.add_module('skip_fusion%d' % (i), skip_fusion)

    def forward(self, inputs, m_d):
        all_encoder_levelputs = []
        all_encoder_outputs = []
        for k in range(len(inputs)):
            if m_d in self.miss_list[k]:
                e, o = getattr(self.encoders, 'encoder%d' % (k))(inputs[k])
                all_encoder_levelputs.append(e)
                all_encoder_outputs.append(o)
        encoder_outputs = []
        for i in range(self.levels - 1):
            e_output = []
            for k in range(len(all_encoder_outputs)):
                e_output.append(all_encoder_levelputs[k][i])

            encoder_outputs.append(getattr(self.skipfuion, 'skip_fusion%d' % (i))(e_output))

        output = self.fusion(all_encoder_outputs)

        seg = self.seg_decoder(output, encoder_outputs)
        idx = 0
        reconstruct_t1c__, reconstruct_t1___, reconstruct_t2___, reconstruct_flair = None, None, None, None
        if m_d in self.miss_list[0]:
            reconstruct_t1c__ = getattr(self.decoders, 'decoder0')(output, all_encoder_levelputs[idx])
            idx+=1
        if m_d in self.miss_list[1]:
            reconstruct_t1___ = getattr(self.decoders, 'decoder1')(output, all_encoder_levelputs[idx])
            idx+=1
        if m_d in self.miss_list[2]:
            reconstruct_t2___ = getattr(self.decoders, 'decoder2')(output, all_encoder_levelputs[idx])
            idx+=1
        if m_d in self.miss_list[3]:
            reconstruct_flair = getattr(self.decoders, 'decoder3')(output, all_encoder_levelputs[idx])

        return {
            'reconstruct_flair': reconstruct_flair,
            'reconstruct_t1c__': reconstruct_t1c__,
            'reconstruct_t1___': reconstruct_t1___,
            'reconstruct_t2___': reconstruct_t2___,
            'seg': seg,
        }
