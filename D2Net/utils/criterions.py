import torch.nn.functional as F
import torch
import logging
import torch.nn as nn
import numpy as np
import time
from torch.autograd import Variable

__all__ = ['sigmoid_dice_loss','softmax_dice_loss','GeneralizedDiceLoss','FocalLoss','dice','CE_loss','bce_loss','IOU_loss','TverskyLoss','SSIM']

cross_entropy = F.cross_entropy

use_class_balance = False


def CE_loss(output, target):
    if use_class_balance:
        mask = torch.zeros([1,4])
        num_total = torch.sum(target)
        num_pos1 = torch.sum((target==1))
        num_pos2 = torch.sum((target==2))
        num_pos4 = torch.sum((target==4))
        num_neg = num_total - num_pos1 - num_pos2 - num_pos4
        mask[0,1] = 1-num_pos1 / num_total
        mask[0,2] = 1-num_pos2 / num_total
        mask[0,3] = 1-num_pos4 / num_total
        mask[0,0] = 1-num_neg / num_total
    
    if output.dim() > 2:
        _output = output.contiguous().view(output.size(0), output.size(1), -1)  # N,C,H,W,D => N,C,H*W*D
        __output = _output.transpose(1, 2)  # N,C,H*W*D => N,H*W*D,C
        output = __output.reshape(-1, __output.size(2))
    if target.dim() == 4:
        _target = target.view(-1) # N*H*W*D
    
    if use_class_balance:
        loss = F.cross_entropy(output, _target, weight=mask.cuda(),ignore_index=-1) #####
    else:
        loss = F.cross_entropy(output, _target,ignore_index=-1)
    
    return loss


def FocalLoss(output, target, alpha=0.25, gamma=2.0):
    if use_class_balance:
        mask = torch.zeros([1,4])
        num_total = torch.sum(target)
        num_pos1 = torch.sum((target==1))
        num_pos2 = torch.sum((target==2))
        num_pos4 = torch.sum((target==4))
        num_neg = num_total - num_pos1 - num_pos2 - num_pos4
        mask[0,1] = 1-num_pos1 / num_total
        mask[0,2] = 1-num_pos2 / num_total
        mask[0,3] = 1-num_pos4 / num_total
        mask[0,0] = 1-num_neg / num_total
        mask = mask/mask.sum()
        ww = torch.Tensor([num_neg.item(),num_pos1.item(),num_pos2.item(),num_pos4.item()])
        mask = mask/(ww+0.00001)

    # target[target == 4] = 3 # label [4] -> [3]
    # target = expand_target(target, n_class=output.size()[1]) # [N,H,W,D] -> [N,4,H,W,D]
    if output.dim() > 2:
        output = output.view(output.size(0), output.size(1), -1)  # N,C,H,W,D => N,C,H*W*D
        output = output.transpose(1, 2)  # N,C,H*W*D => N,H*W*D,C
        output = output.contiguous().view(-1, output.size(2))  # N,H*W*D,C => N*H*W*D,C
    if target.dim() == 5:
        target = target.contiguous().view(target.size(0), target.size(1), -1)
        target = target.transpose(1, 2)
        target = target.contiguous().view(-1, target.size(2))
    if target.dim() == 4:
        target = target.view(-1) # N*H*W*D
    # compute the negative likelyhood
    if use_class_balance:
        _logpt = -F.cross_entropy(output, target, weight=mask.cuda(),reduce=False)   ###
        logpt = -F.cross_entropy(output, target, reduce=False)  ###
        pt = torch.exp(logpt)
        # compute the loss
        loss = -((1 - pt) ** gamma) * _logpt  ###
        return loss.sum()
    else:
        # logpt = -F.cross_entropy(output, target, reduction='none')
        # pt = torch.exp(logpt)
        # loss = -((1 - pt) ** gamma) * logpt  #_logpt  ###   
        # ((1-torch.exp(-F.cross_entropy(output, target, reduction='none')))** gamma) * (-F.cross_entropy(output, target, reduction='none'))

        # focal loss
        loss = F.cross_entropy(output, target, reduction='none')
        logpt = F.log_softmax(output)
        target = target.view(-1, 1)
        logpt = logpt.gather(1, target)
        pt = Variable(logpt.data.exp()).view(-1)
        # pt = logpt.exp().view(-1)
        loss = ((1 - pt)**gamma) * loss

        return loss.mean()  ## .sum()


def dice(output, target,eps =1e-5): # soft dice loss
    target = target
    # num = 2*(output*target).sum() + eps
    num = 2*(output*target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den

def sigmoid_dice_loss(output, target, alpha=1e-5):
    # output: [-1,3,H,W,T]
    # target: [-1,H,W,T] noted that it includes 0,1,2,4 here
    loss1 = dice(F.sigmoid(output[:,1,...]),(target==1),eps=alpha)
    loss2 = dice(F.sigmoid(output[:,2,...]),(target==2),eps=alpha)
    loss3 = dice(F.sigmoid(output[:,3,...]),(target == 3),eps=alpha)
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(1-loss1.data, 1-loss2.data, 1-loss3.data))

    mask = torch.zeros([1,4]).cuda()
    num_total = torch.sum(target)+1e-5
    num_pos1 = torch.sum((target==1))
    num_pos2 = torch.sum((target==2))
    num_pos4 = torch.sum((target==3)) ### 3/4 
    mask[0,1] = 1-num_pos1 / num_total
    mask[0,2] = 1-num_pos2 / num_total
    mask[0,3] = 1-num_pos4 / num_total
    mask[0,1:4] = mask[0,1:4] / mask[0,1:4].sum()
    return (loss1*mask[0,1] + loss2*mask[0,2] + loss3*mask[0,3])  ###




def softmax_dice_loss(output, target,eps=1e-5): # Only for edge-dice-loss calculation without sigmoid
    # output : [bsize,c,H,W,D]
    # target : [bsize,H,W,D]
    loss1 = dice(output[:,1,...],(target==0))
    loss2 = dice(output[:,2,...],(target==1))
    loss3 = dice(output[:,3,...],(target==2))
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(1-loss1.data, 1-loss2.data, 1-loss3.data))

    return (loss1+loss2+loss3)/3


# Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
def GeneralizedDiceLoss(output,target,eps=1e-5,weight_type='square'): # Generalized dice loss
    """
        Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
    """
    # target = target

    if target.dim() == 4:
        target[target == 4] = 3 # label [4] -> [3]
        target = expand_target(target, n_class=output.size()[1]) # [N,H,W,D] -> [N,4，H,W,D]

    output = flatten(output)[1:,...] # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:,...] # [class, N*H*W*D]

    target_sum = target.sum(-1) # sub_class_voxels [3,1] -> 3个voxels
    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :',weight_type)

    intersect = (output * target).sum(-1)
    intersect_sum = (intersect * class_weights).sum()
    denominator = (output + target).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    loss1 = 2*intersect[0] / (denominator[0] + eps)  ## corresponding to class-1
    loss2 = 2*intersect[1] / (denominator[1] + eps)  ## corresponding to class-2
    loss3 = 2*intersect[2] / (denominator[2] + eps)  ## corresponding to class-4

    # cal TC:
    output_tc = torch.cat((output[0],output[2]), 0)
    target_tc = torch.cat((target[0],target[2]), 0)
    dice_TC = 2*(output_tc*target_tc).sum(-1) / ((output_tc+target_tc).sum(-1) + eps)
    # cal WT:
    output_wt = torch.cat((output[0],output[1],output[2]), 0)
    target_wt = torch.cat((target[0],target[1],target[2]), 0)
    dice_WT = 2*(output_wt*target_wt).sum(-1) / ((output_wt+target_wt).sum(-1) + eps)

    dice_ET = loss3
    # cal Sensitivity:
    Sensitivity_TC = (output_tc*target_tc).sum(-1) / target_tc.sum(-1)
    Sensitivity_WT = (output_wt*target_wt).sum(-1) / target_wt.sum(-1)
    Sensitivity_ET = intersect[2] / target[2].sum(-1)

    # cal Specificity:
    Specificity_TC = ((1-output_tc)*(1-target_tc)).sum(-1) / (1-target_tc).sum(-1)
    Specificity_WT = ((1-output_wt)*(1-target_wt)).sum(-1) / (1-target_wt).sum(-1)
    Specificity_ET = ((1-output) * (1-target)).sum(-1)[2] / (1-target[2]).sum(-1)

    logging.info('1: {:.5f} | 2: {:.5f} | 4: {:.5f}'.format(loss1.data, loss2.data, loss3.data))
    logging.info('Dice_ET:{:.5f} | Dice_WT:{:.5f} | Dice_TC:{:.5f} | Sensitivity_ET:{:.5f} | Sensitivity_WT:{:.5f} | Sensitivity_TC:{:.5f}'.format(dice_ET, dice_WT, dice_TC, Sensitivity_ET, Sensitivity_WT, Sensitivity_TC))
    logging.info('Specificity_ET:{:.5f} | Specificity_WT:{:.5f} | Specificity_TC:{:.5f}'.format(Specificity_ET, Specificity_WT, Specificity_TC))

    return 1 - 2. * intersect_sum / denominator_sum


def expand_target(x, n_class,mode='softmax'):
    """
        Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :return: 5D output image (NxCxDxHxW)
        """
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == 'softmax':
        xx[:,1,:,:,:] = (x == 1)
        xx[:,2,:,:,:] = (x == 2)
        xx[:,3,:,:,:] = (x == 3)  ## corresponding to class-4
    if mode.lower() == 'sigmoid':
        xx[:,0,:,:,:] = (x == 1)
        xx[:,1,:,:,:] = (x == 2)
        xx[:,2,:,:,:] = (x == 3)  ## corresponding to class-4
    return xx.to(x.device)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)


# BCE Loss:
def bce_loss(prediction, label, smooth_label=False):
    label = label.clone().long()
    mask = label.clone()
    if smooth_label:
        num_positive = torch.sum((mask!=0))
        num_negative = torch.sum((mask==0))
        mask[mask > 0] = 1.0 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    else:
        num_positive = torch.sum((mask==1))
        num_negative = torch.sum((mask==0))
        mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
        mask[mask == 2] = 0
    cost = torch.nn.functional.binary_cross_entropy(prediction.clone(),label.clone(), weight=mask, reduce=False)
    return torch.sum(cost) / (num_positive+1e-6)


# IOU Loss:
def _iou(pred, target, size_average = True):
    b = pred.shape[0]
    target[target == 4] = 3 # label [4] -> [3]
    IoU = [0.0, 0.0, 0.0]
    for j in range(1,4):
        for i in range(0,b):
            #compute the IoU of the foreground
            target[target == j] = 1
            target[target != j] = 0
            Iand1 = torch.sum(target.clone()[i,:,:,:]*pred.clone()[i,j,:,:,:])
            Ior1 = torch.sum(target.clone()[i,:,:,:]) + torch.sum(pred.clone()[i,j,:,:,:])-Iand1
            IoU1 = Iand1/Ior1
            #IoU loss is (1-IoU1)
            IoU[j-1] = IoU[j-1] + (1-IoU1)
        IoU[j-1] = IoU[j-1]/b
    return sum(IoU) #IoU/b

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average
    def forward(self, pred, target):
        return _iou(pred, target, self.size_average)

def IOU_loss(pred,label):
    iou_loss = IOU(size_average=True)
    iou_out = iou_loss(pred, label)
    return iou_out


# Tversky Loss
def TverskyLoss(output, targets, smooth=1, alpha=0.3, beta=0.7): 
    #comment out if your model contains a sigmoid or equivalent activation layer
    output = F.sigmoid(output.clone())       
    
    #flatten label and prediction tensors
    if output.dim() > 2:
        output = output.view(output.size(0), output.size(1), -1)  # N,C,H,W,D => N,C,H*W*D
        output = output.transpose(1, 2)  # N,C,H*W*D => N,H*W*D,C
        output = output.contiguous().view(-1, output.size(2))  # N,H*W*D,C => N*H*W*D,C
    if targets.dim() == 4:
        targets = targets.view(-1) # N*H*W*D
        targets = torch.unsqueeze(targets,1).expand(targets.size(0), 4)
    
    #True Positives, False Positives & False Negatives
    TP = (output * targets).sum()    
    FP = ((1-targets) * output).sum()
    FN = (targets * (1-output)).sum()
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
    return (1 - Tversky)


# SSIM Loss:
from math import exp
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width, depth) = img1.size()
    if window is None:
        real_size = min(window_size, height, width, depth)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv3d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv3d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        print("sim",sim)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2
    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, normalize=True)


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # torch.nn.functional.linear(input, weight, bias=None)
        # y=x*W^T+b
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # cos(a+b)=cos(a)*cos(b)-size(a)*sin(b)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            # torch.where(condition, x, y) → Tensor
            # condition (ByteTensor) – When True (nonzero), yield x, otherwise yield y
            # x (Tensor) – values selected at indices where condition is True
            # y (Tensor) – values selected at indices where condition is False
            # return:
            # A tensor of shape equal to the broadcasted shape of condition, x, y
            # cosine>0 means two class is similar, thus use the phi which make it
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # scatter_(dim, index, src)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        return output
