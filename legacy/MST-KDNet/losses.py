#!/usr/bin/env python3
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.evd import EVD
from models.slkd import DistillKL_logit_stand


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch, consistency=10, consistency_rampup=20.0):
    return consistency * sigmoid_rampup(epoch, consistency_rampup)


def bce_loss(y_pred, y_label):
    """
    BCEWithLogits loss against a scalar label (0 or 1), on the same device/dtype as y_pred.
    """
    y_truth_tensor = torch.full_like(y_pred, fill_value=float(y_label), device=y_pred.device)
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)


def dice_loss(input, target, eps=1e-7):
    """
    Soft dice loss.
    Assumes input and target have same shape.
    """
    input = input.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (input * target).sum()
    denom = (input.pow(2).sum() + target.pow(2).sum() + eps)
    return 1.0 - (2.0 * intersection / denom)


def gram_matrix(input):
    a, b, c, d, e = input.size()
    features = input.view(a * b, c * d * e)
    G = torch.mm(features, features.t())  # compute the gram product
    return G.div(a * b * c * d * e)

# Correlation between features
def mix_matrix(input1, input2):
    a, b, c, d, e = input1.size()
    input1 = input1.view(a * b, c * d * e)
    input2 = input2.view(a * b, c * d * e)
    G = torch.mm(input1, input2.t())  # compute the gram product
    return G.div(a * b * c * d * e) # B * C * (H * W * D)

def mix_matrix_new(input1, input2):
    a, b, c, d, e = input1.size()
    input1 = input1.view(a * b, c * d * e)
    input2 = input2.view(a * b, c * d * e)
    return torch.mm(input1, input2.t())

def get_style_loss(sf, sm):
    g_f = gram_matrix(sf)
    g_m = gram_matrix(sm)

    channels = sf.size(1)
    size     = sf.size(2)*sf.size(3) 

    sloss = torch.sum(torch.square(g_f-g_m)) / (4.0 * (channels ** 2) * (size ** 2))
    return sloss * 0.0001

def get_GS_loss(Gs_f, Gs_m):
    Ms_f = []
    Ms_m = []

    for i in range(len(Gs_f)):
        Ms_f.append(mix_matrix(Gs_f[i], Gs_f[i-1])) # (C, C) = (128, 128)
        Ms_m.append(mix_matrix(Gs_m[i], Gs_m[i-1]))
    
    print(f"Gs_f size: {Gs_f[0].shape}")    # (B, C, 20, 24, 16)

    channels = Gs_f[0].size(1)                   # C =128
    size     = Gs_f[0].size(2) * Gs_f[0].size(3)

    sloss = 0.0
    for M_f, M_m in zip(Ms_f, Ms_m): # 3
        sloss += torch.sum(torch.square(M_f-M_m)) / (4.0 * (channels ** 2) * (size ** 2))
    
    return sloss * 0.0001

def get_GS_loss_new(Gs_f, Gs_m):
    assert len(Gs_f) == 3 and len(Gs_m) == 3

    Genc_f, Gt_f, Gdec_f = Gs_f
    Genc_m, Gt_m, Gdec_m = Gs_m

    M1_f = mix_matrix(Genc_f, Gdec_f)
    M2_f = mix_matrix(Genc_f, Gt_f)
    M3_f = mix_matrix(Gdec_f, Gt_f)

    M1_m = mix_matrix(Genc_m, Gdec_m)
    M2_m = mix_matrix(Genc_m, Gt_m)
    M3_m = mix_matrix(Gdec_m, Gt_m)

    channels = Genc_f.size(1)
    size = Genc_f.size(2) * Genc_f.size(3) * Genc_f.size(4)

    sloss = 0.0
    for M_f, M_m in zip([M1_f, M2_f, M3_f], [M1_m, M2_m, M3_m]):
        sloss += torch.sum((M_f - M_m) ** 2) / (4.0 * (channels ** 2) * (size ** 2))

    return sloss * 1e-4

def unet_Co_loss(config, batch_pred_full, content_full, batch_y,
                 batch_pred_missing, content_missing, sf, sm, epoch):
    loss_dict = {}

    # Dice loss: 3-class version
    loss_dict['wt_dc_loss'] = dice_loss(batch_pred_full[:, 0], batch_y[:, 0])
    loss_dict['tc_dc_loss'] = dice_loss(batch_pred_full[:, 1], batch_y[:, 1])
    loss_dict['et_dc_loss'] = dice_loss(batch_pred_full[:, 2], batch_y[:, 2])

    loss_dict['wt_miss_dc_loss'] = dice_loss(batch_pred_missing[:, 0], batch_y[:, 0])
    loss_dict['tc_miss_dc_loss'] = dice_loss(batch_pred_missing[:, 1], batch_y[:, 1])
    loss_dict['et_miss_dc_loss'] = dice_loss(batch_pred_missing[:, 2], batch_y[:, 2])

    loss_dict['loss_dc'] = (
        loss_dict['wt_dc_loss'] +
        loss_dict['tc_dc_loss'] +
        loss_dict['et_dc_loss']
    )
    loss_dict['loss_miss_dc'] = (
        loss_dict['wt_miss_dc_loss'] +
        loss_dict['tc_miss_dc_loss'] +
        loss_dict['et_miss_dc_loss']
    )

    # Consistency loss
    loss_dict['wt_mse_loss'] = F.mse_loss(batch_pred_full[:, 0], batch_pred_missing[:, 0], reduction='mean')
    loss_dict['tc_mse_loss'] = F.mse_loss(batch_pred_full[:, 1], batch_pred_missing[:, 1], reduction='mean')
    loss_dict['et_mse_loss'] = F.mse_loss(batch_pred_full[:, 2], batch_pred_missing[:, 2], reduction='mean')
    loss_dict['consistency_loss'] = (
        loss_dict['wt_mse_loss'] +
        loss_dict['tc_mse_loss'] +
        loss_dict['et_mse_loss']
    )

    # Logit discrepancy / content alignment
    loss_dict['content_loss'] = F.mse_loss(content_full, content_missing, reduction='mean')

    # Optional style loss
    if config['use_style_loss']:
        sloss = get_style_loss(sf, sm)

    weight_content = float(config['weight_content'])
    weight_missing = float(config['weight_mispath'])
    weight_full = 1.0 - weight_missing
    weight_consistency = get_current_consistency_weight(epoch)

    loss_dict['loss_Co'] = (
        weight_full * loss_dict['loss_dc'] +
        weight_missing * loss_dict['loss_miss_dc'] +
        weight_consistency * loss_dict['consistency_loss'] +
        weight_content * loss_dict['content_loss'] +
        sloss
    )

    return loss_dict

def unet_Co_loss_divide(config, batch_pred_full, content_full, batch_y,
                        batch_pred_missing, content_missing, sf, sm, epoch):
    loss_dict = {}

    # 4-class version
    loss_dict['net_dc_loss'] = dice_loss(batch_pred_full[:, 0], batch_y[:, 0])
    loss_dict['snfh_dc_loss'] = dice_loss(batch_pred_full[:, 1], batch_y[:, 1])
    loss_dict['et_dc_loss'] = dice_loss(batch_pred_full[:, 2], batch_y[:, 2])
    loss_dict['bg_dc_loss'] = dice_loss(batch_pred_full[:, 3], batch_y[:, 3])

    loss_dict['net_miss_dc_loss'] = dice_loss(batch_pred_missing[:, 0], batch_y[:, 0])
    loss_dict['snfh_miss_dc_loss'] = dice_loss(batch_pred_missing[:, 1], batch_y[:, 1])
    loss_dict['et_miss_dc_loss'] = dice_loss(batch_pred_missing[:, 2], batch_y[:, 2])
    loss_dict['bg_miss_dc_loss'] = dice_loss(batch_pred_missing[:, 3], batch_y[:, 3])

    loss_dict['loss_dc'] = (
        loss_dict['net_dc_loss'] +
        loss_dict['snfh_dc_loss'] +
        loss_dict['et_dc_loss'] +
        loss_dict['bg_dc_loss']
    )
    loss_dict['loss_miss_dc'] = (
        loss_dict['net_miss_dc_loss'] +
        loss_dict['snfh_miss_dc_loss'] +
        loss_dict['et_miss_dc_loss'] +
        loss_dict['bg_miss_dc_loss']
    )

    # Consistency loss
    loss_dict['net_mse_loss'] = F.mse_loss(batch_pred_full[:, 0], batch_pred_missing[:, 0], reduction='mean')
    loss_dict['snfh_mse_loss'] = F.mse_loss(batch_pred_full[:, 1], batch_pred_missing[:, 1], reduction='mean')
    loss_dict['et_mse_loss'] = F.mse_loss(batch_pred_full[:, 2], batch_pred_missing[:, 2], reduction='mean')
    loss_dict['bg_mse_loss'] = F.mse_loss(batch_pred_full[:, 3], batch_pred_missing[:, 3], reduction='mean')
    loss_dict['consistency_loss'] = (
        loss_dict['net_mse_loss'] +
        loss_dict['snfh_mse_loss'] +
        loss_dict['et_mse_loss'] +
        loss_dict['bg_mse_loss']
    )

    # DMLD logit discrepancy term
    loss_dict['content_loss'] = F.mse_loss(content_full, content_missing, reduction='mean')

    ## Style loss
    sloss = get_style_loss(sf, sm)

    weight_content = float(config['weight_content'])
    weight_missing = float(config['weight_mispath'])
    weight_full = 1.0 - weight_missing
    weight_consistency = get_current_consistency_weight(epoch)

    loss_dict['loss_Co'] = (
        weight_full * loss_dict['loss_dc'] +
        weight_missing * loss_dict['loss_miss_dc'] +
        weight_consistency * loss_dict['consistency_loss'] +
        weight_content * loss_dict['content_loss']  + 
        sloss
    )

    return loss_dict


def get_losses(config):
    losses = {
        'co_loss': unet_Co_loss,
        'unetr_loss': Unetr_Loss(),
        'evd_loss': EVD,  # keep as-is if EVD is already callable in your repo
        'slkd_loss': DistillKL_logit_stand(),
        'gsm_loss': get_GS_loss,
    }
    return losses

def get_losses_divide(config):
    losses = {
        'co_loss': unet_Co_loss_divide,
        'unetr_loss': Unetr_Loss(),
        'evd_loss': EVD,  # keep as-is if EVD is already callable in your repo
        'slkd_loss': DistillKL_logit_stand(),
        'gsm_loss': get_GS_loss,
    }
    return losses


class Dice(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super(Dice, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = torch.Tensor(prediction)
        target = torch.Tensor(target)
        iflat = prediction.reshape(-1)
        tflat = target.reshape(-1)
        intersection = (iflat * tflat).sum()

        return ((2.0 * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth)).numpy()



class Unetr_Loss(nn.Module):
    def __init__(self, weight_factor=None):
        super().__init__()
        self.weight_factor = weight_factor

    def forward(self, teacher_fs, student_fs):
        assert len(teacher_fs) == len(student_fs), (
            f"teacher/student feature list length mismatch: {len(teacher_fs)} vs {len(student_fs)}"
        )

        if self.weight_factor is None:
            weights = [1.0] * len(teacher_fs)
        else:
            assert len(self.weight_factor) >= len(teacher_fs), (
                f"weight_factor length ({len(self.weight_factor)}) must be >= number of features ({len(teacher_fs)})"
            )
            weights = self.weight_factor

        total_loss = 0.0
        for i, (t, s) in enumerate(zip(teacher_fs, student_fs)):
            assert t.shape == s.shape, f"Feature map shapes must match: {t.shape} vs {s.shape}"
            total_loss = total_loss + F.mse_loss(s, t, reduction='mean') * weights[i]

        return total_loss