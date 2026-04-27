import numpy as np
import torch


def mask_gen_fusion(Batchsize, NumHead, patches, NumClass, mask):
    attn_shape = (patches*(NumClass+1), patches*(NumClass+1)) # (625, 625)
    bs = mask.size(0)
    mask_shape = (bs, NumHead, patches*(NumClass+1), patches*(NumClass+1))  # (4, 8, 625, 625)
    self_mask_batch = torch.zeros(mask_shape)
    for j in range(bs): #4
        self_mask = np.zeros(attn_shape)    #(625, 625)
        for i in range(NumClass):
            self_mask[patches*i:patches*(i+1), patches*i:patches*(i+1)] = 1  #(0:125, 0:125), (125:250, 125:250), (250:375, 250:375), (375:500, 375:500)
        self_mask[patches*NumClass:patches*(NumClass+1),:] = 1  # (500:625, 500:625)
        for i in range(NumClass):
            if mask[j][i] == 0:
                self_mask[patches*NumClass:patches*(NumClass+1),patches*i:patches*(i+1)] = 0 # masking missing modality-fusion attention
        self_mask = torch.from_numpy(self_mask)
        self_mask = torch.unsqueeze(self_mask, 0).repeat(NumHead,1,1)
        self_mask_batch[j] = self_mask

    return self_mask_batch == 1


def mask_gen_cross4(Batchsize, K, C, mask):
    bs = mask.size(0)
    attn_shape = (bs, K, C)
    self_mask = np.ones(attn_shape)
    for j in range(bs):
        for i in range(4):
            if mask[j][i] == 0:
                self_mask[j:j+1,:,(C//4)*i:(C//4)*(i+1)] = 0

    self_mask = torch.from_numpy(self_mask)

    return self_mask == 1
