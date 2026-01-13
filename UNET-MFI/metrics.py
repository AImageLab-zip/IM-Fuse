import numpy as np

import torch
import torch.nn.functional as F



def dice_coef(output, target):
    smooth = 1e-8
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()


    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def batch_dice_coef(output, target):
    b,h,w,l = output.shape
    ret = []
    for i in range(b):
        ret.append(dice_coef(output[i,...],target[i,...]))
    return ret



if __name__=='__main__':
    pass
