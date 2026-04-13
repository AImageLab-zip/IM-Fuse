import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, inputs, targets):
        loss = 0
        for i in range(3):
            input = inputs[:,i,:,:,:]
            target = targets[:,i,:,:,:]
            bce = F.binary_cross_entropy_with_logits(input, target)
            smooth = 1e-5
            input = torch.sigmoid(input)
            target = target.float()
            num = 2 * (input * target).sum()+smooth
            den = input.sum() + target.sum() + smooth
            dice = 1.0 - num / den
            loss += dice+0.5*bce
        return loss/3


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

if __name__=='__main__':
    a = b = torch.rand(1,3,512,512)
