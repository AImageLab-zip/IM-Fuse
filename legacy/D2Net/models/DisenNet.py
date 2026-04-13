import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import config
import time
from models.disentangle_modules import AdaINGen, Decoder, MLP, LinearBlock, Target_transform, SELayer
from scipy import signal
from scipy.fftpack import fft
module = 'separable_adapter' # specific module type for universal model: series_adapter, parallel_adapter, separable_adapter
trainMode = 'universal'
#from pylab import *
import torch.nn as nn



class disentangle_trainer(nn.Module):
    def __init__(self, inChans_list=[4], base_outChans=2, style_dim=8, mlp_dim=16, num_class_list=[4], args=None):  ## base_outChans=16, mlp_dim=256
        super().__init__()

        self.inChans_list = inChans_list
        self.num_class_list = num_class_list
        
        style_dim = args.style_dim

        indim2d = int(args.train_transforms.split(',')[1])
        self.shared_enc_model = AdaINGen(input_dim=1, dim=base_outChans, decode_indim=base_outChans*pow(2,max(config.num_pool_per_axis)), style_dim=style_dim, mlp_dim=mlp_dim, num_class_list=[1], in_dim_2d=indim2d, auxdec_dim=args.AuxDec_dim, modal_num=inChans_list[0], args=args) #in_dim_2d=indim2d)
    
        if inChans_list[0] == 4:
            self.gen_flair = self.shared_enc_model
            self.gen_t1 = self.shared_enc_model
            self.gen_t1ce = self.shared_enc_model
            self.gen_t2 = self.shared_enc_model
        elif inChans_list[0] == 2:
            self.m1 = self.shared_enc_model
            self.m2 = self.shared_enc_model

        _channels = base_outChans*(pow(2,max(config.num_pool_per_axis)))  # 32
        _outChans_list = [base_outChans*pow(2,i) for i in range(1+max(config.num_pool_per_axis))] # [8, 16, 32, 64]
        
        if args.use_style_map:
            self.target_gen = Target_transform(n_res=4, dim=_channels, output_dim=_channels, res_norm='adain', activ='LeakyReLU', args=args)  # target style transform
        else:
            self.target_gen = Target_transform(n_res=4, dim=_channels, output_dim=_channels, res_norm='in', activ='LeakyReLU', args=args)
        
        self.seg_main_decoder = Decoder(inChans=_channels, outChans_list=_outChans_list, concatChan_list=_outChans_list, num_class_list=num_class_list, norm=None, use_distill=True, use_kd=args.use_kd, args=args)
        
        _outChans_list_small = [args.AuxDec_dim*pow(2,i) for i in range(1+max(config.num_pool_per_axis))] # [8, 16, 32, 64]
        self.binary_dec = Decoder(inChans=_channels, outChans_list=_outChans_list_small, concatChan_list=_outChans_list, num_class_list=num_class_list, norm=None, use_distill=True, use_kd=args.use_kd, args=args) ###

        self.mlp_s_fusion = MLP(inChans_list[0], num_class_list[0], 16, 3, norm='none', activ='none')
        self.mlp_s_map0 = MLP(style_dim, self.get_num_adain_params(self.target_gen), mlp_dim, 3, norm='none', activ='relu')
        self.mlp_s_map1 = MLP(style_dim, self.get_num_adain_params(self.target_gen), mlp_dim, 3, norm='none', activ='relu')
        self.mlp_s_map2 = MLP(style_dim, self.get_num_adain_params(self.target_gen), mlp_dim, 3, norm='none', activ='relu')
        self.mlp_s_map4 = MLP(style_dim, self.get_num_adain_params(self.target_gen), mlp_dim, 3, norm='none', activ='relu')
        
        self.c_fusion = nn.Conv3d(_channels, _channels, kernel_size=1, stride=1, padding=0) # 64*4
        self.target_fusion1 = nn.Conv3d(_channels*num_class_list[0], _channels*2, kernel_size=1, stride=1, padding=0) # 64*4
        self.se = SELayer(_channels*2)  # self Channel-attention
        self.target_fusion2 = nn.Conv3d(_channels*2, _channels, kernel_size=1, stride=1, padding=0) # 64*4 
        self.args = args

        self.distill_fea_fuse = nn.Conv3d(args.fea_dim*4, args.fea_dim, kernel_size=3, stride=1, padding=1) # To fuse the 4-features of Binary decoder when distilling features
    

    def assign_adain_params(self, adain_params, model):
        '''
        Adopt AdaIN layer to fuse the style-aware information and feature maps
        assign the adain_params to the AdaIN layers in model
        '''
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm3d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]
    

    def get_num_adain_params(self, model):
        '''Return the number of AdaIN parameters needed by the model
        '''
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm3d":
                num_adain_params += 2*m.num_features
        return num_adain_params


    def forward(self, x, complete_x=None, is_test=True):  
        if self.args.miss_modal == True:
            avaliable_list = [0] * x.shape[1]
            for i in range(x.shape[1]):
                if x[:,i,...].sum() != 0:
                    avaliable_list[i] = 1
            avaliable_modality = x.sum(axis=1)/sum(avaliable_list)
            for idx in avaliable_list:
                if idx == 0:
                    x[:,idx,...] = avaliable_modality

        modality_num = self.inChans_list[0]
        t0=time.time()
        if self.args.dataset == 'ProstateDataset':
            x_m1, x_m2 = torch.unsqueeze(x[:,0,...],1),torch.unsqueeze(x[:,1,...],1)
        elif self.args.dataset == 'BraTSDataset':
            x_flair, x_t1, x_t1ce, x_t2 = torch.unsqueeze(x[:,0,...],1),torch.unsqueeze(x[:,1,...],1),torch.unsqueeze(x[:,2,...],1),torch.unsqueeze(x[:,3,...],1)
            if complete_x != None:
                x_flair_complete, x_t1_complete, x_t1ce_complete, x_t2_complete = torch.unsqueeze(complete_x[:,0,...],1),torch.unsqueeze(complete_x[:,1,...],1),torch.unsqueeze(complete_x[:,2,...],1),torch.unsqueeze(complete_x[:,3,...],1)
        bs,w = x.shape[0], x.shape[-1]
        x_parallel = torch.unsqueeze(torch.cat([x[:,i,...] for i in range(x.shape[1])],0), 1)  # [bx4,1,h,w,d]

        c_fusion, style_fake_parallel, style_fea_map, enc_list_parallel = self.shared_enc_model.encode(x_parallel, x) # style_fea_map: [1024, 8, 64, 64] -- freq_fea_map    

        _len = len(enc_list_parallel)
        if self.inChans_list[0] == 2:
            enc_list_m1, enc_list_m2 = [enc_list_parallel[i][0:bs] for i in range(_len)], [enc_list_parallel[i][bs:2*bs] for i in range(_len)]
        elif self.inChans_list[0] == 4:
            enc_list_flair, enc_list_t1, enc_list_t1ce, enc_list_t2 = [enc_list_parallel[i][0:bs] for i in range(_len)], [enc_list_parallel[i][bs:2*bs] for i in range(_len)], [enc_list_parallel[i][2*bs:3*bs] for i in range(_len)], [enc_list_parallel[i][3*bs:4*bs] for i in range(_len)]

        s_piece_bs = torch.cat([torch.unsqueeze(style_fake_parallel[modality_num*bs*i:modality_num*bs*(i+1),...],1) for i in range(w)], 1) # [bx4,128,8,1,1]
        s_piece = torch.cat([torch.unsqueeze(s_piece_bs[bs*i:bs*(i+1),...],1) for i in range(modality_num)], 1) # [b,4,128,8,1,1]

        if self.args.use_freq_map:
            s_fea_piece_bs = torch.cat([torch.unsqueeze(style_fea_map[modality_num*bs*i:modality_num*bs*(i+1),...],1) for i in range(w)], 1) # [bx4,128,8,64,64]
            s_fea_piece = torch.cat([torch.unsqueeze(s_fea_piece_bs[bs*i:bs*(i+1),...],1) for i in range(modality_num)], 1) # [2, 4, 128, 8, 64, 64]
            s_flair_fea, s_t1_fea, s_t1ce_fea, s_t2_fea = s_fea_piece[:,0,...], s_fea_piece[:,1,...], s_fea_piece[:,2,...], s_fea_piece[:,3,...] # [b,128,8,64,64]
        
        if self.inChans_list[0] == 4:
            s_flair, s_t1, s_t1ce, s_t2 = s_piece[:,0,...], s_piece[:,1,...], s_piece[:,2,...], s_piece[:,3,...]  # [b,128,8,1,1]

        elif self.inChans_list[0] == 2:
            s_m1, s_m2 = s
            _piece[:,0,...], s_piece[:,1,...]
       
        c_fusion = self.c_fusion(c_fusion)

        # Fusion for middle-featuremaps (to be concat):
        enc_list = enc_list_parallel


        if self.inChans_list[0] == 2:
            rec_m1 = s_m1.mean(1), enc_list_m1
            rec_m2 = s_m2.mean(1), enc_list_m2
            s_fusion = torch.cat([s_m1.mean(1),s_m2.mean(1)],axis=2)   ###
        elif self.inChans_list[0] == 4:
            rec_flair = s_flair.mean(1), enc_list_flair
            rec_t1 = s_t1.mean(1), enc_list_t1
            rec_t1ce = s_t1ce.mean(1), enc_list_t1ce
            rec_t2 = s_t2.mean(1), enc_list_t2
            s_fusion = torch.cat([s_flair.mean(1),s_t1.mean(1),s_t1ce.mean(1),s_t2.mean(1)],axis=2)   ###

        b,w,h=s_fusion.shape[:3]
        _s_fusion =  s_fusion.new_empty((b,w,self.num_class_list[0])).cuda()
        for i in range(s_fusion.shape[0]):
            _s_fusion[i,:,:] = self.mlp_s_fusion(s_fusion[i,...])
        s_fusion = _s_fusion.permute(0,2,1)

        # Get learned target-style-aware features to be input the final segmentor:
        if self.args.use_style_map:
            if self.inChans_list[0] == 2:
                adain_params = self.mlp_s_map0(s_fusion[:,0,:])
                self.assign_adain_params(adain_params, self.target_gen)
                target_fea0 = self.target_gen(c_fusion)

                adain_params = self.mlp_s_map1(s_fusion[:,1,:])
                self.assign_adain_params(adain_params, self.target_gen)
                target_fea1 = self.target_gen(c_fusion)

                adain_params = self.mlp_s_map2(s_fusion[:,2,:])
                self.assign_adain_params(adain_params, self.target_gen)
                target_fea2 = self.target_gen(c_fusion)

                target_fea = torch.cat([target_fea0, target_fea1, target_fea2],axis=1)

            elif self.inChans_list[0] == 4:
                adain_params = self.mlp_s_map0(s_fusion[:,0,:])
                self.assign_adain_params(adain_params, self.target_gen)
                target_fea0 = self.target_gen(c_fusion)

                adain_params = self.mlp_s_map1(s_fusion[:,1,:])
                self.assign_adain_params(adain_params, self.target_gen)
                target_fea1 = self.target_gen(c_fusion)

                adain_params = self.mlp_s_map2(s_fusion[:,2,:])
                self.assign_adain_params(adain_params, self.target_gen)
                target_fea2 = self.target_gen(c_fusion)

                adain_params = self.mlp_s_map4(s_fusion[:,3,:])
                self.assign_adain_params(adain_params, self.target_gen)
                target_fea4 = self.target_gen(c_fusion)

                target_fea = torch.cat([target_fea0, target_fea1, target_fea2, target_fea4],axis=1)
        else:
            target_fea0 = self.target_gen(c_fusion)
            target_fea1 = self.target_gen(c_fusion)
            target_fea2 = self.target_gen(c_fusion)
            target_fea4 = self.target_gen(c_fusion)
            target_fea = torch.cat([target_fea0, target_fea1, target_fea2, target_fea4],axis=1)

     
        target_fea_tmp = self.target_fusion1(target_fea)
        target_fea = self.se(target_fea_tmp) + target_fea_tmp
        target_fea = self.target_fusion2(target_fea)


        # Get segmentation prediction:
        if self.args.use_kd:
            seg_out, deep_sup_fea_all, distill_kd_fea_all, distill_fea_all = self.seg_main_decoder(target_fea, enc_list)   ####### KD
        else:
            seg_out, deep_sup_fea_all, distill_fea_all = self.seg_main_decoder(target_fea, enc_list)   ####### KD

        if is_test == True:
            return [seg_out]

        bs = target_fea0.shape[0]
        if self.inChans_list[0] == 2:
            fea_parallel = torch.cat([target_fea0,target_fea1,target_fea2],0)  # [b*4,32,8,8,8]
            enc_list_parallel = [torch.cat([enc_list[i],enc_list[i],enc_list[i]],0) for i in range(len(enc_list))]
        elif self.inChans_list[0] == 4:
            fea_parallel = torch.cat([target_fea0,target_fea1,target_fea2,target_fea4],0)  # [b*4,32,8,8,8]
            enc_list_parallel = [torch.cat([enc_list[i],enc_list[i],enc_list[i],enc_list[i]],0) for i in range(len(enc_list))]

        # [bx4,2,128,128,128], [bx4,4,64,64,64], [bx4,8,32,32,32], [bx4,16,16,16,16]

        if self.args.use_kd:
            binary_seg_out, deep_sup_fea_bin, distill_kd_fea_bin, distill_fea = self.binary_dec(fea_parallel, enc_list_parallel, is_binary=True)  ###### KD
        else:
            binary_seg_out, deep_sup_fea_bin, distill_fea = self.binary_dec(fea_parallel, enc_list_parallel, is_binary=True)  ###### KD
        
        if self.inChans_list[0] == 2:
            binary_seg_out0, distill_fea0 = binary_seg_out[0:bs], distill_fea[0:bs]
            binary_seg_out1, distill_fea1 = binary_seg_out[bs:2*bs], distill_fea[bs:2*bs]
            binary_seg_out2, distill_fea2 = binary_seg_out[2*bs:3*bs], distill_fea[2*bs:3*bs]
            binary_seg_out_all = torch.cat([binary_seg_out0,binary_seg_out1,binary_seg_out2],axis=1)
        elif self.inChans_list[0] == 4:
            binary_seg_out0, distill_fea0 = binary_seg_out[0:bs], distill_fea[0:bs]
            binary_seg_out1, distill_fea1 = binary_seg_out[bs:2*bs], distill_fea[bs:2*bs]
            binary_seg_out2, distill_fea2 = binary_seg_out[2*bs:3*bs], distill_fea[2*bs:3*bs]
            binary_seg_out4, distill_fea4 = binary_seg_out[3*bs:4*bs], distill_fea[3*bs:4*bs]
            binary_seg_out_all = torch.cat([binary_seg_out0,binary_seg_out1,binary_seg_out2,binary_seg_out4],axis=1)

        if self.args.use_kd:
            if self.inChans_list[0] == 4:
                # Distill logit&feature of Binary seg decoder:
                distill_bin_logit1 = [deep_sup_fea_bin[i][0:bs] for i in range(len(deep_sup_fea_bin))]   # list: 3*[b,1,...]
                distill_bin_logit2 = [deep_sup_fea_bin[i][bs:2*bs] for i in range(len(deep_sup_fea_bin))]
                distill_bin_logit3 = [deep_sup_fea_bin[i][2*bs:3*bs] for i in range(len(deep_sup_fea_bin))]
                distill_bin_logit4 = [deep_sup_fea_bin[i][3*bs:4*bs] for i in range(len(deep_sup_fea_bin))]

                distill_bin_fea1 = [distill_kd_fea_bin[i][0:bs] for i in range(len(distill_kd_fea_bin))]  # list: 3*[b,dim=8,...]
                distill_bin_fea2 = [distill_kd_fea_bin[i][bs:2*bs] for i in range(len(distill_kd_fea_bin))]
                distill_bin_fea3 = [distill_kd_fea_bin[i][2*bs:3*bs] for i in range(len(distill_kd_fea_bin))]
                distill_bin_fea4 = [distill_kd_fea_bin[i][3*bs:4*bs] for i in range(len(distill_kd_fea_bin))]

                # Distill logit&feature of Main seg decoder:
                distill_main_logit = deep_sup_fea_all   # list: 3*[b,4,...]
                distill_main_fea = distill_kd_fea_all   # list: 3*[b,dim=8,...]

                # Forward to obtain the fused distill_feature according to the 4-binary features:
                distill_bin_fea_total = [self.distill_fea_fuse(torch.cat([distill_bin_fea1[i],distill_bin_fea2[i],distill_bin_fea3[i],distill_bin_fea4[i]],1)) for i in range(len(distill_bin_fea1))]

        
        if self.args.use_distill:  # Compute the logit distillation loss
            if self.inChans_list[0] == 2:
                distill_loss = sum([self.l2_loss(torch.unsqueeze(distill_fea_all[:,i,...],1), j) for i,j in enumerate([distill_fea0,distill_fea1])])/2
            elif self.inChans_list[0] == 4:
                distill_loss = torch.stack([
                    self.l2_loss(torch.unsqueeze(distill_fea_all[:, i, ...], 1), j)
                        for i, j in enumerate([distill_fea0, distill_fea1, distill_fea2, distill_fea4])
                    ]).mean()

        else:
            distill_loss = torch.Tensor([0.0]).cuda()
        
        if self.args.use_kd:   # Compute the knowledge distillatoin loss
            assert self.inChans_list[0] == 4
            kd_logit_loss = torch.Tensor([0.0]).cuda()
            
            kd_logit_loss += torch.stack([
                self.distill_kl(
                    torch.unsqueeze(distill_main_logit[idx][:, i, ...], 1),
                    j
                )
                for i, j in enumerate([
                    distill_bin_logit1[idx],
                    distill_bin_logit2[idx],
                    distill_bin_logit3[idx],
                    distill_bin_logit4[idx]
                ])
            ]).mean()

            kd_fea_loss_spatial = torch.stack([
                self.l2_loss(distill_main_fea[i], distill_bin_fea_total[i], channel_wise=False)
                for i in range(len(distill_bin_fea1))
            ]).sum()

            kd_loss = [kd_logit_loss[0]*self.args.kd_logit_w, kd_fea_loss_spatial]

            if self.args.kd_dense_fea_attn == True:
                kd_fea_attn_loss = torch.stack([
                    self.kd_channel_attn(distill_main_fea[i], distill_bin_fea_total[i])
                    for i in range(len(distill_bin_fea1))
                ]).sum()
                kd_loss[1] = kd_fea_attn_loss

            if self.args.affinity_kd == True: # Compute the affinity-guided knowledge distillatoin loss
                affinity_kd_fea_loss = self.affinity_kd(distill_main_fea, distill_bin_fea_total)  # Compute feature KD loss
                kd_loss[0] = affinity_kd_fea_loss * self.args.self_distill_logit_w
                
                logits = [distill_bin_logit1,distill_bin_logit2,distill_bin_logit3,distill_bin_logit4]
                distill_bin_logit = []
                for i in range(len(distill_bin_logit1)):
                    distill_bin_logit.append(torch.cat([logits[0][0],logits[1][0],logits[2][0],logits[3][0]], 1))
                affinity_kd_logit_loss = self.affinity_kd(distill_main_logit, distill_bin_logit)  # Compute logit KD loss
                kd_loss[1] = affinity_kd_logit_loss * self.args.self_distill_fea_w
            
            if self.args.self_distill == True:   # Compute self-distillation KD loss on feature and logit levels
                kd_self_distill_loss_fea_main = self.l2_loss(distill_main_fea[0], nn.MaxPool3d(2, stride=2)(distill_main_fea[1]), channel_wise=False) + \
                                                self.l2_loss(distill_main_fea[0], nn.MaxPool3d(4, stride=4)(distill_main_fea[2]), channel_wise=False)
                kd_self_distill_loss_fea_bin = self.l2_loss(distill_bin_fea_total[0], nn.MaxPool3d(2, stride=2)(distill_bin_fea_total[1]), channel_wise=False) + \
                                                self.l2_loss(distill_bin_fea_total[0], nn.MaxPool3d(4, stride=4)(distill_bin_fea_total[2]), channel_wise=False)
                kd_self_distill_loss_logit_main = self.distill_kl(distill_main_logit[0], nn.MaxPool3d(2, stride=2)(distill_main_logit[1])) + \
                                                    self.distill_kl(distill_main_logit[0], nn.MaxPool3d(4, stride=4)(distill_main_logit[2]))
                
                bin_logit_layer1 = torch.cat([distill_bin_logit1[0],distill_bin_logit2[0],distill_bin_logit3[0],distill_bin_logit4[0]],1)
                bin_logit_layer2 = torch.cat([distill_bin_logit1[1],distill_bin_logit2[1],distill_bin_logit3[1],distill_bin_logit4[1]],1)
                bin_logit_layer3 = torch.cat([distill_bin_logit1[2],distill_bin_logit2[2],distill_bin_logit3[2],distill_bin_logit4[2]],1)
                kd_self_distill_loss_logit_bin =  self.distill_kl(bin_logit_layer1, nn.MaxPool3d(2, stride=2)(bin_logit_layer2)) + \
                                                    self.distill_kl(bin_logit_layer1, nn.MaxPool3d(4, stride=4)(bin_logit_layer3))
                kd_self_distill_fea = kd_self_distill_loss_fea_main + kd_self_distill_loss_fea_bin
                kd_self_distill_logit = kd_self_distill_loss_logit_main + kd_self_distill_loss_logit_bin
                kd_loss[0] += kd_self_distill_logit * self.args.self_distill_logit_w
                kd_loss[1] += kd_self_distill_fea * self.args.self_distill_fea_w
            
            if self.args.kd_channel_attn == True:  # Compute channel-attention based KD loss
                kd_fea_attn_loss = torch.Tensor([0.0]).cuda()
                kd_fea_loss_channel = torch.stack([
                    self.l2_loss(distill_main_fea[i], distill_bin_fea_total[i], channel_wise=True)
                    for i in range(len(distill_bin_fea1))
                ]).sum()

                kd_loss[1] += self.args.kd_fea_channel_w*kd_fea_loss_channel
        else:
            kd_loss = [torch.Tensor([0.0]).cuda(),torch.Tensor([0.0]).cuda()]

        # Constrastive Loss for style_vec: group conv on one dimension of style_enc to get non-overlap style-vec
        if self.inChans_list[0] == 4:
            tmp_s = torch.cat([torch.unsqueeze(s_flair,1), torch.unsqueeze(s_t1,1), torch.unsqueeze(s_t1ce,1), torch.unsqueeze(s_t2,1)],1)  # [b,4,128,8,1,1]
        elif self.inChans_list[0] == 2:
            tmp_s = torch.cat([torch.unsqueeze(s_m1,1), torch.unsqueeze(s_m2,1)],1)
        tmp_s = torch.squeeze(tmp_s,-1)
        tmp_s = torch.squeeze(tmp_s,-1)  # [b,4,128,8]
        
        if self.args.use_contrast:
            contrastive_loss = self.contrastive_module(tmp_s, t=0.07) * self.args.contrast_w
        else:
            contrastive_loss = torch.Tensor([0.0]).cuda()

        if self.args.use_freq_channel:
            freq_loss = self.freq_filter_loss(tmp_s) * self.args.freq_w
        else:
            freq_loss = torch.Tensor([0.0]).cuda() 
        
        if self.args.use_freq_contrast:
            freq_loss = self.freq_contrast_loss(tmp_s) * self.args.freq_w
        else:
            freq_loss = torch.Tensor([0.0]).cuda() 

        if self.args.use_freq_map:
            torch.cat([torch.unsqueeze(s_flair_fea,1), torch.unsqueeze(s_t1_fea,1), torch.unsqueeze(s_t1ce_fea,1), torch.unsqueeze(s_t2_fea,1)],1) # [b,4,128,8,64,64]

        if self.inChans_list[0] == 4:
            if complete_x != None:
                weight_recon_loss, weight_kl_loss, weight_recon_c_loss, weight_recon_s_loss, seg_aux = self.gen_update(x_flair, x_t1, x_t1ce, x_t2, rec_flair, rec_t1, rec_t1ce, rec_t2, c_fusion, enc_list, x_flair_complete, x_t1_complete, x_t1ce_complete, x_t2_complete, self.args)
            else:
                weight_recon_loss, weight_kl_loss, weight_recon_c_loss, weight_recon_s_loss, seg_aux = self.gen_update(x_flair, x_t1, x_t1ce, x_t2, rec_flair, rec_t1, rec_t1ce, rec_t2, c_fusion, enc_list, self.args)
        elif self.inChans_list[0] == 2:
            weight_recon_loss, weight_kl_loss, weight_recon_c_loss, weight_recon_s_loss, seg_aux = self.gen_update(x_m1, x_m2, None, None, rec_m1, rec_m2, None, None, c_fusion, enc_list, self.args)
        
        return seg_out, binary_seg_out_all, deep_sup_fea_all, weight_recon_loss, weight_kl_loss, weight_recon_c_loss, weight_recon_s_loss, distill_loss, kd_loss, contrastive_loss, freq_loss, seg_aux


    def gen_update(self, x_flair, x_t1, x_t1ce, x_t2, rec_flair, rec_t1, rec_t1ce, rec_t2, c_fusion, enc_list, x_flair_complete=None, x_t1_complete=None, x_t1ce_complete=None, x_t2_complete=None, args=None):         
        # encode
        if self.inChans_list[0] == 4:
            s_flair, enc_list_flair = rec_flair  # [content, style]
            s_t1, enc_list_t1 = rec_t1
            s_t1ce, enc_list_t1ce = rec_t1ce
            s_t2, enc_list_t2 = rec_t2

            c_fusion_parallel = torch.cat([c_fusion,c_fusion,c_fusion,c_fusion],0)  # [bx4,64,4,4,4]
            s_parallel = torch.cat([s_flair,s_t1,s_t1ce,s_t2],0)   # [bx4,8,1,1]
            enc_list_parallel = [torch.cat([enc_list[i],enc_list[i],enc_list[i],enc_list[i]],0) for i in range(len(enc_list))]  #[bx4,2,128,128,128]
            recon_all = self.shared_enc_model.decode(c_fusion_parallel, s_parallel, enc_list_parallel)[0]  # 
            bs = c_fusion.shape[0]
            flair_recon, t1_recon, t1ce_recon, t2_recon = recon_all[0:bs], recon_all[bs:2*bs], recon_all[2*bs:3*bs], recon_all[3*bs:4*bs]
        
        elif self.inChans_list[0] == 2:
            s_m1, enc_list_m1 = rec_flair  # [content, style]
            s_m2, enc_list_m2 = rec_t1

            c_fusion_parallel = torch.cat([c_fusion,c_fusion],0)  # [bx4,64,4,4,4]
            s_parallel = torch.cat([s_m1,s_m2],0)   # [bx4,8,1,1]
            enc_list_parallel = [torch.cat([enc_list[i],enc_list[i]],0) for i in range(len(enc_list))]  #[bx4,2,128,128,128]
            recon_all = self.shared_enc_model.decode(c_fusion_parallel, s_parallel, enc_list_parallel)[0]  # 
            bs = c_fusion.shape[0]
            m1_recon, m2_recon = recon_all[0:bs], recon_all[bs:2*bs]

        # Auxiliary Seg_prediction constrains:
        seg_aux = self.seg_main_decoder(c_fusion, enc_list)[0]

        # reconstruction loss
        if self.inChans_list[0] == 4:
            if x_flair_complete == None:
                self.loss_gen_recon_flair = self.recon_criterion(flair_recon, x_flair)
                self.loss_gen_recon_t1 = self.recon_criterion(t1_recon, x_t1)
                self.loss_gen_recon_t1ce = self.recon_criterion(t1ce_recon, x_t1ce)
                self.loss_gen_recon_t2 = self.recon_criterion(t2_recon, x_t2)
            else:
                self.loss_gen_recon_flair = self.recon_criterion(flair_recon, x_flair_complete)
                self.loss_gen_recon_t1 = self.recon_criterion(t1_recon, x_t1_complete)
                self.loss_gen_recon_t1ce = self.recon_criterion(t1ce_recon, x_t1ce_complete)
                self.loss_gen_recon_t2 = self.recon_criterion(t2_recon, x_t2_complete)

            style_code = torch.cat([s_flair,s_t1,s_t1ce,s_t2],axis=2)  # [b,8,4,1]
        
        elif self.inChans_list[0] == 2:
            self.loss_gen_recon_m1 = self.recon_criterion(m1_recon, x_flair)
            self.loss_gen_recon_m2 = self.recon_criterion(m2_recon, x_t1)

            style_code = torch.cat([s_m1,s_m2],axis=2)  # [b,8,4,1]
        
        mu = torch.mean(style_code.view(-1))  # [8,4], [8], [1](all)
        var = torch.var(style_code.view(-1))  # calculate all dim as the whole var/mean
        self.kl_loss = self.compute_kl(mu, var)
        
        # total loss
        if self.inChans_list[0] == 4:
            self.weight_recon_loss = args.recon_w * (self.loss_gen_recon_flair+self.loss_gen_recon_t1+self.loss_gen_recon_t1ce+self.loss_gen_recon_t2)
        elif self.inChans_list[0] == 2:
            self.weight_recon_loss = args.recon_w * (self.loss_gen_recon_m1+self.loss_gen_recon_m2)
        self.weight_kl_loss = args.kl_w * self.kl_loss
        self.weight_recon_c_loss = torch.Tensor([0.0]).cuda()
        self.weight_recon_s_loss = torch.Tensor([0.0]).cuda()
        
        return self.weight_recon_loss, self.weight_kl_loss, self.weight_recon_c_loss, self.weight_recon_s_loss, seg_aux
    

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))


    def kd_channel_attn(self, s, t):  # [b,c,...], only for feature kd (L2) loss
        '''
        Compute channel-attention based knowledge distillation loss
        '''
        kd_losses = 0
        for i in range(s.shape[1]):
            _s = torch.unsqueeze(s[:,i,...],1)
            _loss = torch.mean(torch.abs(_s - t).pow(2), (2,3,4))
            score = torch.sum(_s * t, (2,3,4)) / (torch.norm(torch.flatten(_s,2), dim=(2)) * torch.norm(torch.flatten(t,2), dim=(2)) + 1e-40)
            score = (score + 1)/2
            _score = score / torch.unsqueeze(torch.sum(score,1),1)
            kd_loss = _score * _loss
            kd_losses += kd_loss.mean()
        
        return torch.mean(kd_losses)
    

    def affinity_kd(self, s, t):
        '''
        Compute affinity-guided knowledge distillation loss
        s: a list of feature maps or logits from student branch
        t: a list of feature maps or logits from teacher branch
        '''
        kd_losses = 0
        for i, s_item in enumerate(s):
            for j, t_item in enumerate(t):
                for k in range(s_item.shape[1]):
                    if s_item.shape != t_item.shape:
                        t_item = F.interpolate(t_item, size=s_item.shape[-3:], mode='trilinear', align_corners=False)                         
                    _loss = torch.mean(torch.abs(s_item - t_item).pow(2), (2,3,4))
                    score = torch.sum(s_item * t_item, (2,3,4)) / (torch.norm(torch.flatten(s_item,2), dim=(2)) * torch.norm(torch.flatten(t_item,2), dim=(2)) + 1e-40)
                    score = (score + 1)/2
                    _score = score / torch.unsqueeze(torch.sum(score,1),1)
                    kd_loss = _score * _loss
                    kd_losses += kd_loss.mean()
        
        return torch.mean(kd_losses)


    def l2_loss(self, input, target, channel_wise=False, T=1):  # input/target: [b,dim=8,...]
        if channel_wise == True:
            bs,dim,d,w,h = input.shape
            input = torch.flatten(input,2) # torch.reshape(input,(bs,dim,-1))
            target = torch.flatten(target,2) # torch.reshape(target,(bs,dim,-1))

            loss = F.kl_div(F.log_softmax(input/T, dim=2), F.softmax(target/T, dim=2), reduction='batchmean') * (T**2)

            return loss
        else:
            return torch.mean(torch.abs(input - target).pow(2))


    def distill_kl(self, y_s, y_t, T=1):
        '''
        vanilla distillation loss with KL divergence
        '''
        if y_s.shape[1] == 1:
            y_s = torch.cat([y_s,torch.zeros_like(y_s)],1)
            y_t = torch.cat([y_t,torch.zeros_like(y_t)],1)

        p_s = F.log_softmax(y_s/T+1e-40, dim=1)
        p_t = F.softmax(y_t/T, dim=1)
      
        loss = F.kl_div((p_s), p_t, reduction='batchmean') * (T**2)
      
        return loss
    

    def compute_kl(self, mu, var):
        '''
        KL divergence loss
        '''
        if var < 1e-40:
            log_var = torch.log(var+1e-40)
        else:
            log_var = torch.log(var)
        
        kl_loss = - 0.5 * torch.mean(1. + log_var - mu**2 - var, axis=-1)  # mu**2 / torch.square(mu)
        
        return kl_loss


    def contrastive_module(self, data, t=0.07):  # T=0.07 data=[b,4,128,8]
        contrast_loss = 0.0
        i1, i2 = 0, 0
        bs,c,piece_num,_len = data.shape  # piece_num=128
        
        for i in range(c):
            pos_vec = data[:,i,...]  # [b,128,8]
            for j in range(4,piece_num//2-1,16):
                i1 += 1
                data_list = []
                data_list.append(pos_vec[:,j,:])
                data_list.append(pos_vec[:,j+piece_num//4,:])
                neg_list = [data[:,k,...][:,j-4:j+5,:] for k in range(c) if k != i]
                data_list.extend(neg_list[idx][:,m,:] for idx in range(c-1) for m in range(neg_list[0].shape[1]))
                contrast_loss += self.contrastive_loss(data_list, t)

        return contrast_loss/(i1+i2)


    def contrastive_loss(self, data_list, t=0.07):  # data_list=[pos1,pos2,neg...]
        '''
        Compute softmax-based contrastive loss with temperature t (like Info-NCE)
        '''
        pos_score = self.score_t(data_list[0], data_list[1], t)
        all_score = 0.0
        
        for i in range(1,len(data_list)):
            all_score += self.score_t(data_list[0], data_list[i], t)
        contrast = - torch.log(pos_score/all_score+1e-5).mean()
        
        return contrast


    def score_t(self, x, y, t=0.07):  # x=[b,8]
        '''
        Compute the similarity score between x and y with temperature t
        '''
        if torch.norm(x,dim=1).mean().item() <=0.001 or torch.norm(y,dim=1).mean().item() <=0.001:
            print (torch.norm(x,dim=1).mean().item(),torch.norm(y,dim=1).mean().item())
        
        return torch.exp((x*y).sum(1)/(t*(torch.norm(x,dim=1)*torch.norm(y,dim=1))+1e-5))
    

    def freq_filter_loss(self, data, b_low=0.1, b_high=0.9):  # data: style vectors.  # data: [b,4,128,8 or 16]
        '''
        Compute DFT in frequency domain
        '''
        bs,num_modal,num_slice,dim = data.shape
        loss = [0]*num_modal
        (b,a) = signal.butter(2, [b_low, b_high], 'bandpass')
        
        for modal_idx in range(num_modal):
            slice_bank = [0]*num_slice
            for i in range(num_slice):
                _freq = 0
                for j in range(bs):
                    _freq += signal.filtfilt(b, a, data[j,modal_idx,i,:].cpu().detach().numpy())
                slice_bank[i] = _freq
            loss[modal_idx] = [(x-np.mean(slice_bank,0))**2 for x in slice_bank]
        loss = np.sum(loss)
        
        return loss


    def freq_contrast_loss(self, data):
        '''
        Compute contrastive loss in frequency domain
        '''
        freq_vec = fft(data.cpu().detach().numpy())
        freq_vec = np.abs(freq_vec)
        
        loss = self.contrastive_module(torch.from_numpy(freq_vec).cuda())
        
        return loss

