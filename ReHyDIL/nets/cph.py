import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from thop import profile
import os


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=1, activation=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class DoubleConv(nn.Module):

    def __init__(self, cin, cout):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(cin, cout, 3, 1, padding=1),
            ConvBNReLU(cout, cout, 3, stride=1, padding=1, activation=False)
        )
        self.conv1 = nn.Conv2d(cout, cout, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.conv(x)
        h = x
        x = self.conv1(x)
        x = self.bn(x)
        x = h + x
        x = self.relu(x)
        return x


class DWCONV(nn.Module):
    """
    Depthwise Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=None):
        super(DWCONV, self).__init__()
        if groups == None:
            groups = in_channels
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=groups, bias=True
                                   )

    def forward(self, x):
        result = self.depthwise(x)
        return result


class UEncoder(nn.Module):

    def __init__(self):
        super(UEncoder, self).__init__()
        self.res1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.res2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.res3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.res4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.res5 = DoubleConv(512, 1024)
        self.pool5 = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        x = self.res1(x)
        features.append(x)
        x = self.pool1(x)

        x = self.res2(x)
        features.append(x)
        x = self.pool2(x)

        x = self.res3(x)
        features.append(x)
        x = self.pool3(x)

        x = self.res4(x)
        features.append(x)
        x = self.pool4(x)

        x = self.res5(x)
        features.append(x)
        x = self.pool5(x)
        features.append(x)
        return features


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class HGNN(nn.Module):
    def __init__(self, in_ch, n_out):
        super(HGNN, self).__init__()
        self.conv = nn.Linear(in_ch, n_out)
        self.bn = nn.BatchNorm1d(n_out)

    def forward(self, x, G):
        residual = x
        x = self.conv(x)
        x = G.matmul(x)
        x = F.relu(self.bn(x.permute(0,2,1).contiguous())).permute(0,2,1).contiguous() + residual
        return x


class G_HGNN_layer(nn.Module):
    def __init__(self, in_ch, node=None, K_neigs=None, kernel_size=5, stride=2):
        super(G_HGNN_layer, self).__init__()
        self.HGNN = HGNN(in_ch, in_ch)
        self.K_neigs = K_neigs
        self.node = node
        self.kernel_size = kernel_size
        self.stride = stride
        self.single_local_H = self.local_kernel(node, kernel_size=kernel_size, stride=stride)


    def forward(self, x):

        B, N, C = x.shape
        x_merged = x.reshape(B*N, C).unsqueeze(0)

        ori_dists = self.pairwise_distance(x_merged)

        k = self.K_neigs[0]
        topk_dists, topk_inds = ori_dists.topk(k+1, dim=2, largest=False, sorted=True)
        avg_dists = ori_dists.mean(-1, keepdim=True)


        H = self.create_incidence_matrix_inter(topk_dists, topk_inds, avg_dists, B, N)

        Dv = torch.sum(H, dim=2, keepdim=True)
        alpha = 1.0
        Dv = Dv * alpha
        max_k = int(Dv.max())


        _topk_dists, _topk_inds = ori_dists.topk(max_k, dim=2, largest=False, sorted=True)  # (1,B*N,max_k)
        _avg_dists = ori_dists.mean(-1, keepdim=True)
        new_H = self.create_incidence_matrix_inter(_topk_dists, _topk_inds, _avg_dists, B, N)


        local_H = self.build_block_diagonal_localH(self.single_local_H, B, x.device)

        _H = torch.cat([new_H, local_H], dim=2)

        _G = self._generate_G_from_H_b(_H)


        x_out = self.HGNN(x_merged, _G)
        x_out = x_out.squeeze(0).view(B,N,C)
        return x_out

    @torch.no_grad()
    def create_incidence_matrix_inter(self, top_dists, inds, avg_dists, B, N, prob=False):

        _, total_nodes, K = top_dists.shape
        weights = self.weights_function(top_dists, avg_dists, prob)

        incidence_matrix = torch.zeros(1, total_nodes, total_nodes, device=inds.device)

        pixel_indices = torch.arange(total_nodes, device=inds.device)[:, None]
        incidence_matrix[0, pixel_indices, inds.squeeze(0)] = weights.squeeze(0)
        return incidence_matrix

    def build_block_diagonal_localH(self, single_local_H, B, device):
        # single_local_H: (N, E)
        # N = self.node * self.node
        N = self.node * self.node
        E = single_local_H.size(1)
        H_local = single_local_H.to(device)


        block_diag = torch.zeros(B * N, B * E, device=device)
        for i in range(B):
            startN = i * N
            endN = startN + N
            startE = i * E
            endE = startE + E

            block_diag[startN:endN, startE:endE] = H_local

        return block_diag.unsqueeze(0)






    @torch.no_grad()
    def _generate_G_from_H_b(self, H, variable_weight=False):

        bs, n_node, n_hyperedge = H.shape



        W = torch.ones([bs, n_hyperedge], requires_grad=False, device=H.device)

        DV = torch.sum(H, dim=2)
        DE = torch.sum(H, dim=1)
        DE = torch.clamp(DE, min=1e-8)

        invDE = torch.diag_embed((torch.pow(DE, -1)))
        DV2 = torch.diag_embed((torch.pow(DV, -0.5)))
        W = torch.diag_embed(W)
        HT = H.transpose(1, 2)



        if variable_weight:
            DV2_H = DV2 @ H
            invDE_HT_DV2 = invDE @ HT @ DV2
            return DV2_H, W, invDE_HT_DV2
        else:

            G = DV2 @ H @ W @ invDE @ HT @ DV2

            return G


    @torch.no_grad()
    def pairwise_distance(self, x):

        with torch.no_grad():
            x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
            x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
            return x_square + x_inner + x_square.transpose(2, 1)



    @torch.no_grad()
    def batched_knn(self, x, k=1):

        ori_dists = self.pairwise_distance(x)
        avg_dists = ori_dists.mean(-1, keepdim=True)
        topk_dists, topk_inds = ori_dists.topk(k + 1, dim=2, largest=False, sorted=True)

        return topk_dists, topk_inds, ori_dists, avg_dists

    @torch.no_grad()
    def create_incidence_matrix(self, top_dists, inds, avg_dists, prob=False):
        B, N, K = top_dists.shape
        weights = self.weights_function(top_dists, avg_dists, prob)
        incidence_matrix = torch.zeros(B, N, N, device=inds.device)

        batch_indices = torch.arange(B)[:, None, None].to(inds.device)
        pixel_indices = torch.arange(N)[None, :, None].to(inds.device)

        incidence_matrix[batch_indices, pixel_indices, inds] = weights

        return incidence_matrix.permute(0,2,1).contiguous()



    @torch.no_grad()
    def weights_function(self, topk_dists, avg_dists, prob):
        if prob:

            topk_dists_sq = topk_dists.pow(2)
            normalized_topk_dists_sq = topk_dists_sq / avg_dists
            weights = torch.exp(-normalized_topk_dists_sq)
        else:
            weights = torch.ones(topk_dists.size(), device=topk_dists.device)
        return weights

    @torch.no_grad()
    def local_kernel(self, size, kernel_size=3, stride=1):
        inp = torch.arange(size * size, dtype=torch.float).reshape(size, size)[None, None, :, :]

        inp_unf = torch.nn.functional.unfold(inp, kernel_size=(kernel_size, kernel_size), stride=stride).squeeze(
            0).transpose(0, 1).long()

        edge, node = inp_unf.shape
        matrix = torch.arange(edge)[:, None].repeat(1, node).long()

        H_local = torch.zeros((size * size, edge))


        H_local[inp_unf, matrix] = 1.

        return H_local




class G_HyperNet(nn.Module):
    def __init__(self, channel, node = 28, kernel_size=3, stride=1, K_neigs = None):
        super(G_HyperNet, self).__init__()
        self.G_HGNN_layer = G_HGNN_layer(channel, node = node, kernel_size=kernel_size, stride=stride, K_neigs=K_neigs)

    def forward(self, x):

        b,c,w,h = x.shape
        x = x.view(b,c,-1).permute(0,2,1).contiguous()
        x = self.G_HGNN_layer(x)
        x = x.permute(0,2,1).contiguous().view(b,c,w,h)

        return x

class L_HyperNet(nn.Module):
    def __init__(self, channel, node = 28, kernel_size=3, stride=1, K_neigs = None):
        super(L_HyperNet, self).__init__()
        self.L_HGNN_layer = L_HGNN_layer(channel, node = node, kernel_size=kernel_size, stride=stride, K_neigs=K_neigs)

    def forward(self, x):

        b,c,w,h = x.shape
        x = x.view(b,c,-1).permute(0,2,1).contiguous()
        x = self.L_HGNN_layer(x)
        x = x.permute(0,2,1).contiguous().view(b,c,w,h)

        return x


class HyperEncoder(nn.Module):
    def __init__(self, channel=[1024, 1024]):
        """
        参数说明：
          channel[0] 对应原 HGNN_layer2 的输入通道数（例如1024）
          channel[1] 对应原 HGNN_layer3 的输入通道数（例如1024）
        """
        super(HyperEncoder, self).__init__()
        kernel_size = 3
        stride = 1
        self.HGNN_layer2 = G_HyperNet(channel[0], node=14, kernel_size=kernel_size, stride=stride, K_neigs=[1])
        self.HGNN_layer3 = G_HyperNet(channel[1], node=7, kernel_size=kernel_size, stride=stride, K_neigs=[1])

    def forward(self, x):

        _, _, _, _, feature2, feature3 = x
        out2 = self.HGNN_layer2(feature2)
        out3 = self.HGNN_layer3(feature3)
        return [out2, out3]



class ParallEncoder(nn.Module):
    def __init__(self):
        super(ParallEncoder, self).__init__()
        self.Encoder1 = UEncoder()
        self.Encoder2 = HyperEncoder(channel=[1024, 1024])
        self.num_module = 2

        self.fusion_list = [1024, 1024]

        self.squeelayers = nn.ModuleList()
        for i in range(self.num_module):
            self.squeelayers.append(
                nn.Conv2d(self.fusion_list[i] * 2, self.fusion_list[i], kernel_size=1, stride=1)
            )

    def forward(self, x):
        skips = []
        features = self.Encoder1(x)

        feature_hyper = self.Encoder2(features)

        skips.extend(features[:4])

        for i in range(self.num_module):
            fused = self.squeelayers[i](torch.cat((feature_hyper[i], features[i + 4]), dim=1))
            skips.append(fused)
        return skips

class CPH(nn.Module):
    def __init__(self, n_classes = 9):
        super(CPH, self).__init__()
        self.p_encoder = ParallEncoder()
        self.encoder_channels = [1024, 512, 256, 128, 64]
        self.decoder1 = DecoderBlock(self.encoder_channels[0] + self.encoder_channels[0], self.encoder_channels[1])
        self.decoder2 = DecoderBlock(self.encoder_channels[1] + self.encoder_channels[1], self.encoder_channels[2])
        self.decoder3 = DecoderBlock(self.encoder_channels[2] + self.encoder_channels[2], self.encoder_channels[3])
        self.decoder4 = DecoderBlock(self.encoder_channels[3] + self.encoder_channels[3], self.encoder_channels[4])


        self.segmentation_head2 = SegmentationHead(
            in_channels=256,
            out_channels=n_classes,
            kernel_size=1,
        )
        self.segmentation_head3 = SegmentationHead(
            in_channels=128,
            out_channels=n_classes,
            kernel_size=1,
        )
        self.segmentation_head4 = SegmentationHead(
            in_channels=64,
            out_channels=n_classes,
            kernel_size=1,
        )
        self.segmentation_head5 = SegmentationHead(
            in_channels=64,
            out_channels=n_classes,
            kernel_size=1,
        )
        self.decoder_final = DecoderBlock(in_channels=64, out_channels=64)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        encoder_skips = self.p_encoder(x)

        x1_up = self.decoder1(encoder_skips[-1], encoder_skips[-2])
        x2_up = self.decoder2(x1_up, encoder_skips[-3])
        x3_up = self.decoder3(x2_up, encoder_skips[-4])
        x4_up = self.decoder4(x3_up, encoder_skips[-5])
        x_final = self.decoder_final(x4_up, None)


        logits = self.segmentation_head5(x_final)

        return logits

if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    model = CPH(n_classes=1).to(device)
    inout = torch.randn((4, 1, 224, 224)).to(device)
    logits = model(inout)
    print(logits.shape)
    print('# generator parameters:', 1.0 * sum(param.numel() for param in model.parameters())/1000000)
    macs, params = profile(model, inputs=(inout,))
    print("FLOPs: {:.2f} GFLOPs".format(macs / 1e9))
    print("Parameters: {:.2f} M".format(params / 1e6))

