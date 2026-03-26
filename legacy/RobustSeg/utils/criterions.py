
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain, combinations

MODALITIES_img = ['T1', 'T1c', 'T2', 'Flair']
def all_subsets(l):
    #Does not include the empty set and l
    return list(chain(*map(lambda x: combinations(l, x), range(1, len(l)+1))))

SUBSETS_MODALITIES = all_subsets(MODALITIES_img)

def masker(x, mask):
    y = torch.zeros_like(x)
    y[mask, ...] = x[mask, ...]
    return y

def kl_loss(mu, logvar) : #[B,128,1,1,1]
    logvar = torch.clamp(logvar, min=-50, max=50)  
    loss = 0.5 * torch.sum(torch.square(mu) + torch.exp(logvar) - 1 - logvar, axis=-1)
    loss = torch.mean(loss)
    return loss

def KL_divergence(mu_1, logvar_1, mu_2, logvar_2):
    var_1 = torch.exp(logvar_1)
    var_2 = torch.exp(logvar_2)
    return 1/2 * torch.mean(-1 + logvar_2 - logvar_1 + (var_1 + torch.square(mu_1-mu_2))/(var_2+1e-2))

def Product_Gaussian(means, logvars, list_mod):
    mu_prior = torch.zeros_like(means[MODALITIES_img[0]])
    log_prior = torch.zeros_like(means[MODALITIES_img[0]])

    eps=1e-5
    T = torch.stack([1 / (torch.exp(logvars[mod]) + eps) for mod in list_mod], dim=0)   #(1-4, B, 4, 112, 112, 112)/(4, B, 8, 56, 56, 56)/(4, B, 16, 28, 28, 28)/(4, B, 32, 14, 14, 14)
    mu = torch.stack([means[mod] / (torch.exp(logvars[mod]) + eps) for mod in list_mod], dim=0)
    T = torch.cat([T, (1 + log_prior).unsqueeze(0)], dim=0)   
    mu = torch.cat([mu, mu_prior.unsqueeze(0)], dim=0)                                #(2-5, B, 4, 112, 112, 112)/(5, B, 8, 56, 56, 56)/(5, B, 16, 28, 28, 28)/(5, B, 32, 14, 14, 14)

    posterior_means = torch.sum(mu, dim=0) / torch.sum(T, dim=0)
    var = 1 / torch.sum(T, dim=0)  
    posterior_logvars = torch.log(var + eps)

    return posterior_means, posterior_logvars
"""
def Product_Gaussian(means, logvars, list_mod):
    mu_prior = torch.zeros_like(means[list_mod[0]])
    log_prior = torch.zeros_like(means[list_mod[0]])

    eps = 1e-2  # Match TensorFlow's epsilon
    T = [1 / (torch.exp(logvars[mod]) + eps) for mod in list_mod] + [1 + log_prior]
    mu = [means[mod] / (torch.exp(logvars[mod]) + eps) for mod in list_mod] + [mu_prior]

    posterior_means = sum(mu) / sum(T)
    var = 1 / sum(T)
    posterior_logvars = torch.log(var + eps)

    return posterior_means, posterior_logvars
"""
def Product_Gaussian_main(means, logvars, list_mod, choices):
    eps=1e-5
    mu_prior = torch.zeros_like(means[MODALITIES_img[0]])
    log_prior = torch.zeros_like(means[MODALITIES_img[0]])

    T = masker(torch.stack([1 / (torch.exp(logvars[mod]) + eps) for mod in list_mod], dim=0), choices.permute(1, 0)) #(0-4, B, C, 112, 112, 112)
    mu = masker(torch.stack([means[mod] / (torch.exp(logvars[mod]) + eps) for mod in list_mod], dim=0), choices.permute(1, 0))

    #filtered_T = torch.zeros_like(T)
    #filtered_mu = torch.zeros_like(mu)
    #mask_to_select = choices.permute(1, 0) #(4, B)
    #filtered_T[mask_to_select, ...] = T[mask_to_select, ...] 
    #filtered_mu[mask_to_select, ...] = mu[mask_to_select, ...]
    #a = [1/(torch.exp(logvars[mod]) + eps)  for mod in list_mod]
    #b = [means[mod]/(torch.exp(logvars[mod]) + eps) for mod in list_mod]
    #T = torch.zeros(4,mu_prior.shape[0],mu_prior.shape[1],mu_prior.shape[2],mu_prior.shape[3],mu_prior.shape[4]).cuda()
    #mu = torch.zeros(4,log_prior.shape[0],log_prior.shape[1],log_prior.shape[2],log_prior.shape[3],log_prior.shape[4]).cuda()
    #for i in range(4):
    #    if choices[0][i]:
    #        T[i,...] = a[i]
    #        mu[i,...] = b[i]
    T = torch.cat([T, (1 + log_prior).unsqueeze(0)], dim=0)   
    mu = torch.cat([mu, mu_prior.unsqueeze(0)], dim=0)       #(2-5, B, 4, 112, 112, 112)/(5, B, 8, 56, 56, 56)/(5, B, 16, 28, 28, 28)/(5, B, 32, 14, 14, 14)
        
    posterior_means = torch.sum(mu, dim=0) / torch.sum(T, dim=0)
    var = 1 / torch.sum(T, dim=0)
    posterior_logvars = torch.log(var + eps)

    return posterior_means, posterior_logvars

def compute_KLD(means, logvars, choices):
    # Prior parameters
    mu_prior = torch.zeros_like(means[MODALITIES_img[0]])
    log_prior = torch.zeros_like(means[MODALITIES_img[0]])

    # Full modalities
    full_means, full_logvars = Product_Gaussian_main(means, logvars, MODALITIES_img, choices)   #(B, 4, 112, 112, 112)

    sum_inter_KLD = 0
    sum_prior_KLD = 0

    for subset in SUBSETS_MODALITIES:
        sub_means, sub_logvars = Product_Gaussian(means, logvars, subset)                       #(B, 4, 112, 112, 112)

        #Inter modality KLD
        sub_inter_KLD = KL_divergence(full_means, full_logvars, sub_means, sub_logvars)
        sum_inter_KLD += sub_inter_KLD

        #Modality to 0,1
        sub_prior_KLD = KL_divergence(sub_means, sub_logvars, mu_prior, log_prior)
        sum_prior_KLD += sub_prior_KLD

    return 1/15*sum_inter_KLD, 1/15*sum_prior_KLD

def cross_loss(output, target, num_cls=4):
    criterion = nn.BCEWithLogitsLoss()
    assert output.shape == target.shape, 'predict & target shape do not match'
    total_loss = 0
    for i in range(num_cls):
        bce_loss = criterion(output[:, i], target[:, i])
        total_loss += bce_loss
    return torch.mean(total_loss) 

def softmax_dice_loss(output, target, num_cls=4, eps=0.00001):
    output = F.softmax(output, dim=1)
    target = target.float()
    for i in range(num_cls):
        num = torch.sum(output[:,i,:,:,:] * target[:,i,:,:,:])
        l = torch.sum(output[:,i,:,:,:])
        r = torch.sum(target[:,i,:,:,:])
        if i == 0:
            dice = (2.0*num+eps) / (l+r+eps)
        else:
            dice += (2.0*num+eps) / (l+r+eps)
    return 1.0 - 1.0 * dice / num_cls

def dice_loss(output, target, num_cls=4, eps=1e-7):
    target = target.float()
    for i in range(num_cls):
        num = torch.sum(output[:,i,:,:,:] * target[:,i,:,:,:])
        l = torch.sum(output[:,i,:,:,:])
        r = torch.sum(target[:,i,:,:,:])
        if i == 0:
            dice = 2.0 * num / (l+r+eps)
        else:
            dice += 2.0 * num / (l+r+eps)
    return 1.0 - 1.0 * dice / num_cls

def dice(output, target,eps =1e-5): # soft dice loss
    target = target.float()
    # num = 2*(output*target).sum() + eps
    num = 2*(output*target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den


def softmax_loss(output, target, num_cls=5):
    target = target.float()
    _, _, H, W, Z = output.size()
    for i in range(num_cls):
        outputi = output[:, i, :, :, :]
        targeti = target[:, i, :, :, :]
        if i == 0:
            cross_loss = -1.0 * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        else:
            cross_loss += -1.0 * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
    cross_loss = torch.mean(cross_loss)
    return cross_loss


def softmax_weighted_loss(output, target, num_cls=5):
    target = target.float() #(B, 4, 128, 128, 128)
    B, _, H, W, Z = output.size() #(B, C, 128, 128, 128)
    for i in range(num_cls):
        outputi = output[:, i, :, :, :] #(B, 128, 128, 128)
        targeti = target[:, i, :, :, :]
        weighted = 1.0 - (torch.sum(targeti, (1,2,3)) / (torch.sum(target, (1,2,3,4)) + 1e-8))
        weighted = torch.reshape(weighted, (-1,1,1,1)).repeat(1,H,W,Z)
        if i == 0:
            cross_loss = -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        else:
            cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
    cross_loss = torch.mean(cross_loss)
    return cross_loss