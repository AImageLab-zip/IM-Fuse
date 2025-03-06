import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random 

def ShuffleIndex_with_MDP(index: list, sample_ratio: float, mdp = 0, mask = True, patch_shape = 128):
    interal = int(np.power(patch_shape / 16, 3)) #512
    temp_index = index.copy()
    ### Modality Masking
    mdp_list = []
    mdp = np.random.randint(0, mdp+1)
    if mdp > 3:
        mdp = 3
        
    for l in range(mdp):
        cindex = np.random.randint(0,4)
        while cindex in mdp_list:
            cindex = np.random.randint(0,4)
        mdp_list.append(cindex)
        
    if len(mdp_list) != 0:
        for l in mdp_list:
            for ls in range(l*interal, (l+1)*interal):
                temp_index.remove(ls)

    ### Patch Masking
    sample_list = []
    if len(index) < 4:
        raise ValueError("ipnuts must be more than 4")
    elif mask: #False
        sample_length = int((1-sample_ratio) * len(index))
        
        while len(sample_list) < sample_length:
            sample = random.choice(temp_index)
            sample_list.append(sample)
            temp_index.remove(sample)
        
        mask_list = [x for x in index if x not in sample_list]  # get the remain index not in cls token and not in sample_index

        assert len(mask_list) == int(len(index) * sample_ratio), "sample length must be same as the ratio!!!"
    else:
        # only with MDP
        sample_list = temp_index
        mask_list = [x for x in index if x not in sample_list]
    
    return sample_list, mask_list 
    
def ShuffleIndex_with_mask_modal(index: list, mask_modal = [], patch_shape = 128):
    interal = int(np.power(patch_shape / 16, 3))
    temp_index = index.copy()
    mdp_list = mask_modal
        
    if len(mdp_list) != 0:
        for l in mdp_list:
            for ls in range(l*interal, (l+1)*interal):
                temp_index.remove(ls)
    #print(mdp, len(temp_index))
    sample_list = []
    
    sample_list = temp_index
    mask_list = [x for x in index if x not in sample_list]
    
    return sample_list, mask_list 
    
def proj(image, patch_size = 16):

    B, C, D, H, W = image.shape
    image_ = image.reshape(B, C, D // patch_size, patch_size, H // patch_size, patch_size, W // patch_size, patch_size)
    
    image_ = image_.permute(0, 1, 2, 4, 6, 3, 5, 7).reshape(B, C * D // patch_size * H // patch_size * W // patch_size,patch_size, patch_size, patch_size)
    
    return image_

def MaskEmbeeding1(B, mask_ratio=0.75, raw_input = None, patch_size = 16, token_index = [x for x in range(0, 1372)], mdp = 0, mask = True, mask_modal = [], patch_shape = 128):
    """get the mask embeeding after patch_emb + pos_emb
    for numpy
    """
    #imageshape:128*128*128
    D, H, W = patch_shape, patch_shape, patch_shape
    token_index = [x for x in range(0, int(np.power(patch_shape / 16, 3)) * 4)]
    
    if len(mask_modal) == 0:
        sample_index, mask_index = ShuffleIndex_with_MDP(token_index, mask_ratio, mdp = mdp, mask = mask, patch_shape = patch_shape)
    else:
        if -1 in mask_modal:
            sample_index, mask_index = ShuffleIndex_with_mask_modal(token_index, mask_modal = [], patch_shape = patch_shape)
        else:
            sample_index, mask_index = ShuffleIndex_with_mask_modal(token_index, mask_modal = mask_modal, patch_shape = patch_shape)
    
    
    decoder_embeeding = np.zeros((B, raw_input.shape[1], patch_size, patch_size, patch_size))
    
    decoder_embeeding[:,  sample_index, :, :, :] = raw_input[:,  sample_index, :, :, :]
    
    decoder_embeeding = decoder_embeeding.reshape(B, 4, D // patch_size, H // patch_size, W // patch_size, patch_size, patch_size, patch_size).transpose(0, 1, 2, 5, 3, 6, 4, 7)
    
    decoder_embeeding = decoder_embeeding.reshape(B, 4, D, H, W)
    
    return decoder_embeeding


def masking(B, mask_ratio = 0.125, patch_size = 16, raw_size = [64, 192, 192]):

    token_count = raw_size[0] // patch_size * raw_size[1] // patch_size * raw_size[2] // patch_size
    token_index = [x for x in range(0, token_count)]
    temp_index = token_index.copy()
    masked_list = []

    sample_length = int(token_count * mask_ratio)

    while len(masked_list) < sample_length:
        sample = random.choice(temp_index)
        masked_list.append(sample)
        temp_index.remove(sample)
    
    decoder_embeeding = np.zeros((B, token_count, patch_size, patch_size, patch_size))
    decoder_embeeding[:,  masked_list, :, :, :] = 1

    decoder_embeeding = decoder_embeeding.reshape(B, 1, raw_size[0] // patch_size, raw_size[1] // patch_size, raw_size[2] // patch_size, patch_size, patch_size, patch_size).transpose(0, 1, 2, 5, 3, 6, 4, 7)
    decoder_embeeding = decoder_embeeding.reshape(B, 1, raw_size[0], raw_size[1], raw_size[2])

    return decoder_embeeding






