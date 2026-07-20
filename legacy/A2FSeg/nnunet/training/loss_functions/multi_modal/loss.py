import torch
import torch.nn as nn
from torch.autograd import Function, Variable
import torch.nn.functional as F
from itertools import repeat, chain, combinations
import numpy as np

# from buildingblocks import ZeroLayerF
"""
3D
"""
# Intersection = dot(A, B)
# Union = dot(A, A) + dot(B, B)
# The Dice loss function is defined as
# 1/2 * intersection / union
#
# The derivative is 2[(union * target - 2 * intersect * input) / union^2]

MODALITIES = [0,1,2,3]
def all_subsets(l):
    #Does not include the empty set
    return list(chain(*map(lambda x: combinations(l, x), range(1, len(l)+1))))

SUBSETS_MODALITIES = all_subsets(MODALITIES)

"""
ref : https://github.com/ReubenDo/U-HVED
"""
def KL_divergence(mu1, logvar1, mu2 = None, logvar2 = None, eps=1e-8):
    " KLD(p1 || p2)"
    if mu2 is None:
        mu2 = mu1.new_zeros(mu1.shape) # prior
        logvar2 = torch.log(mu1.new_ones(mu1.shape))
        eps = 0
    var1 = logvar1.exp()
    var2 = logvar2.exp() # default : 1
#     KLD = 0.5*torch.mean(torch.sum(-1  + logvar2 - logvar1 + (var1 + (mu1 - mu2).pow(2)) / (var2 + eps), axis=1))
    KLD = 0.5*torch.mean(-1 + logvar2 - logvar1 + (var1 + (mu1 - mu2).pow(2)) / (var2 + eps))
    
    return KLD

class ProductOfExperts(nn.Module):
    """ ref : github.com/mhw32/multimodal-vae-public/blob/master/celeba19/model.py
    Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu_list, logvar_list, mod_list, eps=1e-8): # 0 : prior, 1~N : modality
        logvar = [logvar_list[mod+1] for mod in mod_list] + [logvar_list[0]]
        mu = [mu_list[mod+1] for mod in mod_list] + [mu_list[0]]
        logvar = torch.stack(logvar, 0)
        mu = torch.stack(mu, 0)
        
        
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / var
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        
        return pd_mu, pd_logvar

class ProductOfExperts2(nn.Module): # for drop
    """ ref : github.com/mhw32/multimodal-vae-public/blob/master/celeba19/model.py
    Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, drop, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / var
#         for m in range(drop.shape[1]):
#             mu[m+1] = ZeroLayerF.apply(mu[m+1], drop[:, m])
#             T[m+1] = ZeroLayerF.apply(T[m+1], drop[:, m])
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
                
        return pd_mu, pd_logvar
    
def compute_KLD(mu_list, logvar_list, subset_index_list=[14], choices=[0,1,2,3]):
    # (B,5,C,W,H,D)
    # Prior parameters : list[0]
    # _list -> 0 : prior, 1~N : modality
    mu_list = mu_list.transpose(1,0)
    logvar_list = logvar_list.transpose(1,0)
    mu_list = mu_list # (5,B,C,W,H,D)
    logvar_list = logvar_list
    
    mu_prior = mu_list[0]
#     print(mu_prior.shape)
    logvar_prior = logvar_list[0]
    
    experts = ProductOfExperts()
    # Full modalities
#     full_mu, full_logvar = experts(mu_list, logvar_list, choices)
    # Initialization sums
    sum_prior_KLD = 0

    #sum_prior_KLD += KL_divergence(full_mu, full_logvar, mu_prior, logvar_prior)
#     subset_index_list = np.random.choice(range(4,14), 3, replace=False)
    cnt = 0
    for idx, subset in enumerate(SUBSETS_MODALITIES):
        if idx in subset_index_list:
            cnt += 1
            sub_mu, sub_logvar = experts(mu_list, logvar_list, subset)

            # Modality to 0,1
            sub_prior_KLD = KL_divergence(sub_mu, sub_logvar, mu_prior, logvar_prior)
            sum_prior_KLD += sub_prior_KLD
    return 1/cnt*sum_prior_KLD

def compute_KLD_drop(mu_list, logvar_list, drop, choices=[0,1,2,3]):
    mu_prior = mu_list[0]
    logvar_prior = logvar_list[0]
    
    experts = ProductOfExperts()
    experts_drop = ProductOfExperts2()

    # Initialization sums
    sum_inter_KLD = 0
    sum_prior_KLD = 0
    
    drop_mu, drop_logvar = experts_drop(mu_list, logvar_list, drop)
    
    sub_prior_KLD = KL_divergence(drop_mu, drop_logvar, mu_prior, logvar_prior)
    sum_prior_KLD += sub_prior_KLD

    return 1/7*sum_inter_KLD, sum_prior_KLD


def boundary_loss(outputs_soft, gt_sdf):
    """
    compute boundary loss for binary segmentation
    input: outputs_soft: sigmoid results,  shape=(b,3,x,y,z)
           gt_sdf: sdf of ground truth (can be original or normalized sdf); shape=(b,3,x,y,z)
    output: boundary_loss; sclar
    """
    multipled = torch.einsum('bcxyz, bcxyz->bcxyz', outputs_soft, gt_sdf)
    bd_loss = multipled.mean()

    return bd_loss


class BCELoss(nn.Module):
    def __init__(self, index = 0):
        super(BCELoss, self).__init__()
        self.label_index = index
        self.loss = nn.BCELoss()

    def forward(self, input, target):
        assert (input.shape == target.shape)
        
        tot_loss = 0
        for i in range(3):
            pred = input[:,i]
            gt = target[:,i]

            tot_loss += self.loss(pred, gt)

        return tot_loss

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class DiceLoss(nn.Module):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation 'weight' parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)
    
    def forward(self, input, target):
        # get probabilities from logits
#         input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)
    
class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237
    """

    def __init__(self, weight=None, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.weight = weight

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"
        
#         input = F.softmax(input, dim=1)
        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())
    
    def forward(self, input, target):
        # get probabilities from logits
#         input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)
    
def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"
    
#     input = F.softmax(input, dim=1)
    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, H, W) -> (C, N * H * W) / 2D
       (N, C, D, H, W) -> (C, N * D * H * W) / 3D
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)
    
    
class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(input)
        target = torch.argmax(target, 1)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
#         input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights



####################################################################################################
def prediction_map_distillation(y, teacher_scores, T=4) :
    """
    basic KD loss function based on "Distilling the Knowledge in a Neural Network"
    https://arxiv.org/abs/1503.02531
    :param y: student score map
    :param teacher_scores: teacher score map
    :param T:  for softmax
    :return: loss value
    """
    p = F.log_softmax(y / T, dim=1)
    q = F.softmax(teacher_scores / T, dim=1)

    p = p.view(-1, 2)
    q = q.view(-1, 2)

    l_kl = F.kl_div(p, q, reduction='batchmean') * (T ** 2)
    return l_kl


def at(x, exp):
    """
    attention value of a feature map
    :param x: feature
    :return: attention value
    """
    return F.normalize(x.pow(exp).mean(1).view(x.size(0), -1))


def importance_maps_distillation(s, t, exp=4):
    """
    importance_maps_distillation KD loss, based on "Paying More Attention to Attention:
    Improving the Performance of Convolutional Neural Networks via Attention Transfer"
    https://arxiv.org/abs/1612.03928
    :param exp: exponent
    :param s: student feature maps
    :param t: teacher feature maps
    :return: imd loss value
    """
    if s.shape[2] != t.shape[2]:
        s = F.interpolate(s, t.size()[-2:], mode='bilinear')
    return torch.sum((at(s, exp) - at(t, exp)).pow(2), dim=1).mean()


def region_contrast(x, gt):
    """
    calculate region contrast value
    :param x: feature
    :param gt: mask
    :return: value
    """
    smooth = 1.0
    mask0 = gt[:, 0].unsqueeze(1)
    mask1 = gt[:, 1].unsqueeze(1)

    region0 = torch.sum(x * mask0, dim=(2, 3)) / torch.sum(mask0, dim=(2, 3))
    region1 = torch.sum(x * mask1, dim=(2, 3)) / (torch.sum(mask1, dim=(2, 3)) + smooth)
    return F.cosine_similarity(region0, region1, dim=1)


def region_affinity_distillation(s, t, gt):
    """
    region affinity distillation KD loss
    :param s: student feature
    :param t: teacher feature
    :return: loss value
    """
    gt = F.interpolate(gt, s.size()[2:])
    return (region_contrast(s, gt) - region_contrast(t, gt)).pow(2).mean()