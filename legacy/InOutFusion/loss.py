import torch
import torch.nn.functional as F
from utils.util import dice_score

def get_dice(seg, label):
    s = F.softmax(seg, dim=1)
    _, pre = torch.max(s, dim=1, keepdim=True)

    p_temp, l_temp = pre.clone(), label.clone()
    dice_wt = dice_score(p_temp, l_temp, 'wt')

    p_temp2, l_temp2 = pre.clone(), label.clone()
    dice_tc = dice_score(p_temp2, l_temp2, 'tc')

    p_temp3, l_temp3 = pre.clone(), label.clone()
    dice_et = dice_score(p_temp3, l_temp3, 'et')

    return dice_wt, dice_tc, dice_et

def get_label(label):
    extent_label = None
    for k in [0, 1, 2, 3]:
        la = label.clone()
        la[la == k] = -1
        la[la != -1] = 0
        la[la != 0] = 1
        if extent_label is None:
            extent_label = la
        else:
            extent_label = torch.cat([extent_label, la], dim=1)
    return extent_label

def general_dice_loss(seg, label, mdice):
    s = F.softmax(seg, dim=1)
    la = get_label(label).detach()
    loss = mdice(s, la, [0.1, 0.2, 0.3, 0.4])
    return loss

def U_Hemis_loss(seg, label, mdice):
    return general_dice_loss(seg, label, mdice)


def RMBTS_loss(re_dic, label, inputs, mdice, m_d, miss_list, device):
    la = get_label(label).detach()
    seg = re_dic['seg']
    dice_loss = general_dice_loss(seg, label, mdice)

    s = F.softmax(seg, dim=1)

    ce_loss=None
    for i in range(la.shape[1]):
        labeli = la[:, i, :, :, :]
        predi = s[:, i, :, :, :]
        weighted = 1.0-(torch.sum(labeli).item() / torch.sum(la).item())
        if i == 0:
            ce_loss = -1.0 * weighted * labeli * torch.log(torch.clamp(predi, 0.005, 1))
        else:
            ce_loss += -1.0 * weighted * labeli * torch.log(torch.clamp(predi, 0.005, 1))
    ce_loss = torch.mean(ce_loss)

    seg_loss = dice_loss + ce_loss

    rec_name = ['reconstruct_t1c__','reconstruct_t1___','reconstruct_t2___','reconstruct_flair']
    mu_name = ['mu_t1c__','mu_t1___','mu_t2___','mu_flair']
    sigma_name = ['sigma_t1c__','sigma_t1___','sigma_t2___','sigma_flair']


    rec_loss = None
    for i in range(len(rec_name)):
        if m_d in miss_list[i]:
            if rec_loss == None:
                rec_loss = torch.mean(torch.abs(re_dic[rec_name[i]] - inputs[i]))
            else:
                rec_loss += torch.mean(torch.abs(re_dic[rec_name[i]] - inputs[i]))

    KL_loss = None
    for i in range(len(mu_name)):
        if m_d in miss_list[i]:
            if KL_loss == None:
                KL_loss = kl_loss(re_dic[mu_name[i]], torch.log(torch.pow(re_dic[sigma_name[i]], 2)))
            else:
                KL_loss += kl_loss(re_dic[mu_name[i]], torch.log(torch.pow(re_dic[sigma_name[i]], 2)))

    return seg_loss + 0.1 * rec_loss + 0.1 * KL_loss, dice_loss, ce_loss, rec_loss, KL_loss

def kl_loss(mu, logvar) :
    loss = 0.5 * torch.sum(torch.pow(mu, 2) + torch.exp(logvar) - 1 - logvar, dim=1)
    loss = torch.mean(loss)
    return loss


def LMCR_loss(re_dic, label, inputs, mdice, m_d, miss_list, device):

    seg = re_dic['seg']
    dice_loss = general_dice_loss(seg, label, mdice)
    seg_loss = dice_loss

    rec_name = ['reconstruct_t1c__','reconstruct_t1___','reconstruct_t2___','reconstruct_flair']

    rec_loss = None
    for i in range(len(rec_name)):
        if m_d in miss_list[i]:
            if rec_loss == None:
                rec_loss = torch.mean(torch.abs(re_dic[rec_name[i]] - inputs[i]))
            else:
                rec_loss += torch.mean(torch.abs(re_dic[rec_name[i]] - inputs[i]))

    return seg_loss + rec_loss, dice_loss, rec_loss,













