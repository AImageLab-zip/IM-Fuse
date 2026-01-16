import os
import torch
import torch.nn as nn

def parse_image_name(name):
    n = name.split('.')[0]
    mod, pid, index, pn = n.split('_')
    return mod, pid, index, pn, 'modality'+name[len(mod):]

def print_net(model, model_name):
    #print(model)
    if model_name not in ['RMBTS', 'TF_RMBTS']:
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n = n / 1000000
        print('[*] has {:.4f}M parameters!'.format(n))


def check_dirs(path):
    if type(path) not in (tuple, list):
        path = [path]
    for p in path:
        if not os.path.exists(p):
            os.makedirs(p)
    return

def split_test_data(volumn):
    p = volumn[0][0][0][0][0]
    volumns = []

    v1 = torch.zeros_like(volumn[:, :, :, :, 0:16]) + p
    volumn = torch.cat([v1, volumn], dim=4)

    v2 = torch.zeros_like(volumn[:, :, :, 0:16, :]) + p
    volumn = torch.cat([v2, volumn], dim=3)

    v3 = torch.zeros_like(volumn[:, :, 0:5, :, :]) + p
    volumn = torch.cat([v3, volumn], dim=2)

    for i in [0, 128]:
        for j in [0, 128]:
            volumns.append(volumn[:, :, :, i:i + 128, j:j + 128])

    return volumns

def concat_test_data(volumns):
    volumn = torch.cat([torch.cat([volumns[0], volumns[1]], dim=4), torch.cat([volumns[2], volumns[3]], dim=4)], dim=3)
    return volumn[:, :, 5:, 16:, 16:]

def dice_score(pred, masks, type='wt', eps=1e-10):
    eps = 1e-10

    if type == 'tc':
        pred[pred==2] = 0
        masks[masks==2] = 0
    elif type == 'et':
        pred[pred==1] = 0
        pred[pred==2] = 0
        masks[masks == 1] = 0
        masks[masks == 2] = 0
    pred[pred!=0] = 1
    masks[masks!=0] = 1

    inter = (pred * masks)

    inter = inter.sum().float()
    pred = pred.sum().float()
    masks = masks.sum().float()

    dices = (2 * inter + eps) / (pred + masks + eps)

    return  dices

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pre, target):
        N = target.size(0)
        smooth = 1e-7

        assert pre.size() == target.size(), 'size of pre and target are can not match'

        pre_flat = pre.view(N,-1)

        target_flat = target.view(N,-1)

        intersection = pre_flat * target_flat

        # loss = (2 * intersection.sum(1) + smooth) / (pre_flat.sum(1) + target_flat.sum(1) + smooth)
        # loss = 1 - loss.sum() / N
        loss = 1- (2 * intersection.sum() + smooth) / (pre_flat.sum() + target_flat.sum() + smooth)
        
        return loss

class MulticlassDiceLoss(nn.Module):
    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()
        self.dice = DiceLoss()

    def forward(self, pre, target, weights=None):
        C = target.shape[1]
        totalLoss = 0

        for i in range(C):

            diceLoss = self.dice(pre[:, i], target[:, i])
            if weights != None:
                diceLoss *= weights[i]
            totalLoss+=diceLoss
        return totalLoss



def re_crop(volumn, crop_size):
    z = 152
    y = 240
    x = 240
    min_z = crop_size[0].item()
    max_z = crop_size[1].item()
    min_y = crop_size[2].item()
    max_y = crop_size[3].item()
    min_x = crop_size[4].item()
    max_x = crop_size[5].item()

    if min_z != 0:
        volumn = torch.cat([torch.zeros_like(volumn[:min_z, :, :]), volumn], dim=0)
    if (z - max_z) != 0:
        volumn = torch.cat([volumn, torch.zeros_like(volumn[:(z - max_z), :, :])], dim=0)

    if min_y != 0:
        volumn = torch.cat([torch.zeros_like(volumn[:, :min_y, :]), volumn], dim=1)
    if (y - max_y) != 0:
        volumn = torch.cat([volumn, torch.zeros_like(volumn[:, :(y - max_y), :])], dim=1)

    if min_x != 0:
        volumn = torch.cat([torch.zeros_like(volumn[:, :, :min_x]), volumn], dim=2)
    if (x - max_x) != 0:
        volumn = torch.cat([volumn, torch.zeros_like(volumn[:, :, :(x - max_x)])], dim=2)

    volumn = torch.cat([torch.zeros_like(volumn[:3, :, :]), volumn], dim=0)

    return volumn






