
import torch
import torch.nn as nn

from net.BasicBlock import UNetEncoder, UNetDecoder, ConvNormRelu3D, TF_3D
from net.RestormerCnnBlock import Restormer_CNN_block, RSEncoder
from process.utils import missing_list


class U_Hemis3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, levels=4, feature_maps=32, method=None, phase='train'):
        super(U_Hemis3D, self).__init__()
        self.feature_maps = feature_maps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.levels = levels
        self.miss_list = missing_list()

        self.encoders = nn.Sequential()
        for i in range(4):
            encoder = UNetEncoder(in_channels=self.in_channels, feature_maps=feature_maps, levels=self.levels)
            self.encoders.add_module('encoder%d' % (i), encoder)

        self.decoder = UNetDecoder(out_channels=self.out_channels , feature_maps=feature_maps, levels=self.levels)

        self.fusion = ConvNormRelu3D(2**(self.levels)*feature_maps, 2**(self.levels-1)*feature_maps)

        self.skipfuion = nn.Sequential()
        for i in range(self.levels-1):
            skip_fusion = ConvNormRelu3D(2**(self.levels-i-1) * feature_maps, 2**(self.levels-i-2) * feature_maps)
            self.skipfuion.add_module('skip_fusion%d' % (self.levels-1-i), skip_fusion)

    def forward(self, inputs, m_d):
        all_encoder_levelputs = []
        all_encoder_outputs = []
        for k in range(len(inputs)): # len(inputs) --> 4
            if m_d in self.miss_list[k]:
                e, o = getattr(self.encoders, 'encoder%d' % (k))(inputs[k])
                all_encoder_levelputs.append(e)
                all_encoder_outputs.append(o)

        encoder_outputs = []
        for i in range(self.levels-1):
            e_output = None
            for k in range(len(all_encoder_outputs)):
                if e_output == None:
                    e_output = all_encoder_levelputs[k][i].unsqueeze(dim=0)
                else:
                    e_output = torch.cat([e_output,all_encoder_levelputs[k][i].unsqueeze(dim=0)],dim=0)

            vs, ms = torch.var_mean(e_output, dim=0)
            if len(all_encoder_outputs) == 1:
                vs = torch.zeros_like(ms)
            else:
                vs = vs / (len(all_encoder_outputs)-1)
            encoder_outputs.append(getattr(self.skipfuion, 'skip_fusion%d' % (i+1))(torch.cat([ms,vs],dim=1)))

        output = None
        for k in range(len(all_encoder_outputs)):
            if output == None:
                output = all_encoder_outputs[k].unsqueeze(dim=0)
            else:
                output = torch.cat([output, all_encoder_outputs[k].unsqueeze(dim=0)], dim=0)
        v, m = torch.var_mean(output, dim=0)
        if len(all_encoder_outputs) == 1:
            v = torch.zeros_like(m)
        else:
            v = v / (len(all_encoder_outputs) - 1)
        output = self.fusion(torch.cat([m,v],dim=1))
        seg = self.decoder(output, encoder_outputs)

        return seg


class TF_U_Hemis3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, levels=4, feature_maps=32, method='TF', phase='train', trans='basic'):
        super(TF_U_Hemis3D, self).__init__()
        self.feature_maps = feature_maps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.levels = levels
        self.miss_list = missing_list()

        self.encoders = nn.Sequential()
        for i in range(4):
            encoder = UNetEncoder(in_channels=self.in_channels, feature_maps=feature_maps, levels=self.levels)
            self.encoders.add_module('encoder%d' % (i), encoder)

        self.decoder = UNetDecoder(out_channels=self.out_channels , feature_maps=feature_maps, levels=self.levels)

        self.fusion = TF_3D(embedding_dim=2**(self.levels-1)*feature_maps, volumn_size=(128//(2**(self.levels-1))), method=method, trans=trans)

        self.skipfuion = nn.Sequential()
        for i in range(self.levels-1):
            skip_fusion = TF_3D(embedding_dim=2**(self.levels-i-2) * feature_maps, volumn_size=(128//(2**(self.levels-2-i))), method=method, trans=trans)
            self.skipfuion.add_module('skip_fusion%d' % (self.levels-1-i), skip_fusion)

    def forward(self, inputs, m_d):
        all_encoder_levelputs = []
        all_encoder_outputs = []
        for k in range(len(inputs)):
            if m_d in self.miss_list[k]:
                e, o = getattr(self.encoders, 'encoder%d' % (k))(inputs[k])
                all_encoder_levelputs.append(e)
                all_encoder_outputs.append(o)

        encoder_outputs = []
        for i in range(self.levels-1):
            e_output = []
            for k in range(len(all_encoder_outputs)):
                e_output.append(all_encoder_levelputs[k][i])
            if i ==1:
                oput = getattr(self.skipfuion, 'skip_fusion%d' % (i+1))(e_output)
            else:
                oput = getattr(self.skipfuion, 'skip_fusion%d' % (i + 1))(e_output)
            encoder_outputs.append(oput)

        output = []
        for k in range(len(all_encoder_outputs)):
            output.append(all_encoder_outputs[k])
        output = self.fusion(output)

        seg = self.decoder(output, encoder_outputs)
        return seg
    
    
class AF_U_Hemis3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, levels=4, feature_maps=32, method='TF', phase='train'):
        super(AF_U_Hemis3D, self).__init__()
        self.feature_maps = feature_maps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.levels = levels
        self.miss_list = missing_list()

        self.encoders = nn.Sequential()
        for i in range(4):
            encoder = UNetEncoder(in_channels=self.in_channels, feature_maps=feature_maps, levels=self.levels)
            self.encoders.add_module('encoder%d' % (i), encoder)

        self.decoder = UNetDecoder(out_channels=self.out_channels , feature_maps=feature_maps, levels=self.levels)

        self.fusion = TF_3D(embedding_dim=2**(self.levels-1)*feature_maps, volumn_size=(128//(2**(self.levels-1))), method=method, trans='ff')

        self.skipfuion = nn.Sequential()
        for i in range(self.levels-1):
            skip_fusion = TF_3D(embedding_dim=2**(self.levels-i-2) * feature_maps, volumn_size=(128//(2**(self.levels-2-i))), method=method)
            self.skipfuion.add_module('skip_fusion%d' % (self.levels-1-i), skip_fusion)

    def forward(self, inputs, m_d):
        all_encoder_levelputs = []
        all_encoder_outputs = []
        for k in range(len(inputs)):
            if m_d in self.miss_list[k]:
                e, o = getattr(self.encoders, 'encoder%d' % (k))(inputs[k])
                all_encoder_levelputs.append(e)
                all_encoder_outputs.append(o)

        encoder_outputs = []
        for i in range(self.levels-1):
            e_output = []
            for k in range(len(all_encoder_outputs)):
                e_output.append(all_encoder_levelputs[k][i])
            if i ==1:
                oput = getattr(self.skipfuion, 'skip_fusion%d' % (i+1))(e_output)
            else:
                oput = getattr(self.skipfuion, 'skip_fusion%d' % (i + 1))(e_output)
            encoder_outputs.append(oput)

        output = []
        for k in range(len(all_encoder_outputs)):
            output.append(all_encoder_outputs[k])
        output = self.fusion(output)

        seg = self.decoder(output, encoder_outputs)
        return seg
    

class InOut_U_Hemis3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, levels=4, feature_maps=32, method='TF', phase='train'):
        super(InOut_U_Hemis3D, self).__init__()
        self.feature_maps = feature_maps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.levels = levels
        self.miss_list = missing_list()

        self.encoders = nn.Sequential()
        for i in range(4):
            encoder = UNetEncoder(in_channels=self.in_channels, feature_maps=feature_maps, levels=self.levels)
            self.encoders.add_module('encoder%d' % (i), encoder)

        self.decoder = UNetDecoder(out_channels=self.out_channels , feature_maps=feature_maps, levels=self.levels)

        self.fusion = TF_3D(embedding_dim=2**(self.levels-1)*feature_maps, volumn_size=(128//(2**(self.levels-1))), method=method, trans='ff')

        self.skipfuion = nn.Sequential()
        for i in range(self.levels-1):
            skip_fusion = TF_3D(embedding_dim=2**(self.levels-i-2) * feature_maps, volumn_size=(128//(2**(self.levels-2-i))), method='la')
            self.skipfuion.add_module('skip_fusion%d' % (self.levels-1-i), skip_fusion)

    def forward(self, inputs, m_d):
        all_encoder_levelputs = []
        all_encoder_outputs = []
        for k in range(len(inputs)):
            if m_d in self.miss_list[k]:
                e, o = getattr(self.encoders, 'encoder%d' % (k))(inputs[k])
                all_encoder_levelputs.append(e)
                all_encoder_outputs.append(o)

        encoder_outputs = []
        for i in range(self.levels-1):
            e_output = []
            for k in range(len(all_encoder_outputs)):
                e_output.append(all_encoder_levelputs[k][i])
            if i ==1:
                oput = getattr(self.skipfuion, 'skip_fusion%d' % (i + 1))(e_output)
            else:
                oput = getattr(self.skipfuion, 'skip_fusion%d' % (i + 1))(e_output)
            encoder_outputs.append(oput)

        output = []
        for k in range(len(all_encoder_outputs)):
            output.append(all_encoder_outputs[k])
        output = self.fusion(output)

        seg = self.decoder(output, encoder_outputs)
        return seg


class RsInOut_U_Hemis3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, levels=4, feature_maps=32, method='TF', phase='train'):
        super(RsInOut_U_Hemis3D, self).__init__()
        self.feature_maps = feature_maps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.levels = levels
        self.miss_list = missing_list()

        self.encoders = nn.Sequential()
        for i in range(4):
            # InFusion
            encoder = RSEncoder(self.in_channels, feature_maps)
            # encoder = UNetEncoder(in_channels=self.in_channels, feature_maps=feature_maps, levels=self.levels)
            self.encoders.add_module('encoder%d' % (i), encoder)
        self.decoder = UNetDecoder(out_channels=self.out_channels , feature_maps=feature_maps, levels=self.levels)

        # outfusion
        # self.fusion 在最深层，关注目标及患者的高维语义信息，所以使用频、空域融合attention
        # skip fusion 是为了传输浅层文本特征，所以用local attention
        self.fusion = TF_3D(embedding_dim=2**(self.levels-1)*feature_maps, volumn_size=(128//(2**(self.levels-1))), method=method, trans='ff')

        self.skipfuion = nn.Sequential()
        for i in range(self.levels-1):
            skip_fusion = TF_3D(embedding_dim=2**(self.levels-i-2) * feature_maps, volumn_size=(128//(2**(self.levels-2-i))), method='la')
            self.skipfuion.add_module('skip_fusion%d' % (self.levels-1-i), skip_fusion)

    def forward(self, inputs, m_d):
        all_encoder_levelputs = []
        all_encoder_outputs = []
        for k in range(len(inputs)):
            if m_d in self.miss_list[k]:
                e, o = getattr(self.encoders, 'encoder%d' % (k))(inputs[k])
                all_encoder_levelputs.append(e)
                all_encoder_outputs.append(o)

        encoder_outputs = []
        for i in range(self.levels-1):
            e_output = []
            for k in range(len(all_encoder_outputs)):
                e_output.append(all_encoder_levelputs[k][i])
            if i ==1:
                oput = getattr(self.skipfuion, 'skip_fusion%d' % (i + 1))(e_output)
            else:
                oput = getattr(self.skipfuion, 'skip_fusion%d' % (i + 1))(e_output)
            encoder_outputs.append(oput)

        output = []
        for k in range(len(all_encoder_outputs)):
            output.append(all_encoder_outputs[k])
        output = self.fusion(output)

        seg = self.decoder(output, encoder_outputs)
        return seg