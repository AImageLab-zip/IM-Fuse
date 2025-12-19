import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import sys
import nibabel as nib
import scipy.misc
import re

import torchvision
from medpy.metric import dc, hd95
from tqdm import tqdm

cudnn.benchmark = True
path = os.path.dirname(__file__)

def compute_BraTS_HD95(ref, pred):
    """
    ref and gt are binary integer numpy.ndarray s
    spacing is assumed to be (1, 1, 1)
    :param ref:
    :param pred:
    :return:
    """
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)
    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 373.12866
    elif num_pred == 0 and num_ref != 0:
        return 373.12866
    else:
        return hd95(pred, ref, (1, 1, 1))

def cal_hd95(output, target):
     # whole tumor
    mask_gt = (target != 0).astype(int)
    mask_pred = (output != 0).astype(int)
    hd95_whole = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # tumor core
    mask_gt = (target > 1).astype(int)
    mask_pred = (output > 1).astype(int)
    hd95_core = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # enhancing
    mask_gt = (target == 4).astype(int)
    mask_pred = (output == 3).astype(int)
    hd95_enh = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    return hd95_whole, hd95_core, hd95_enh

# dice socre is equal to f1 score
def dice_score(o, t,eps = 1e-8):
    num = 2*(o*t).sum() + eps #
    den = o.sum() + t.sum() + eps # eps
    return num/den

def prostate_dice(output, target, ignore_pixel=255):
    ret = []
    # whole
    if (target==ignore_pixel).sum() > 0:
        target[target==ignore_pixel] = 0
    o = output > 0; t = (target > 0)
    ret += dice_score(o, t),
    # 1
    o = (output==1) 
    t = (target==1)
    ret += dice_score(o , t),
    # 2
    o = (output==2); t = (target==2)
    ret += dice_score(o , t),
    return ret

def softmax_output_dice(output, target):
    ret = []
    # whole
    o = output > 0; t = target > 0 # ce
    ret += dice_score(o, t),
    # core
    o = (output==1) | (output==3)
    t = (target==1) | (target==4)
    ret += dice_score(o , t),
    # active
    o = (output==3); t = (target==4)
    ret += dice_score(o , t),
    return ret

keys = 'WT', 'TC', 'ET', 'loss'
keys_hd95 = 'WT', 'TC', 'ET'
import torch
import torch.nn.functional as F
import numpy as np
import time
import wandb

def validate_softmax(
        train_valid_loader,
        model,
        cpu_only=False,
        epoch_id=0,
        use_wandb=False,   # optional wandb logging
        args = None
    ):
    """
    Simplified validation loop (no test phase).
    Computes Dice and HD95 metrics on validation set.
    Optionally logs mean metrics to Weights & Biases (W&B).
    """
    H, W, T = 240, 240, 155
    model.eval()
    runtimes = []
    vals = AverageMeter()
    vals_hd95 = AverageMeter()

    for data in tqdm(train_valid_loader,total=len(train_valid_loader)):

        target_cpu = data['label'][0, :H, :W, :T].numpy()
        x = data['image']
        target = data['label']
        if not cpu_only:
            x = x.cuda()
            target = target.cuda()

        # --- forward pass ---
        logit = model(x)[0]

        output = F.softmax(logit, dim=1)
        output = output[0, :, :H, :W, :T].cpu().numpy().argmax(0)
        del x, logit
        torch.cuda.empty_cache()

        # --- optional postprocess ---
        if getattr(args, "postprocess", False):
            ET_voxels = (output == 3).sum()
            if ET_voxels < 500:
                output[np.where(output == 3)] = 1

        # --- compute metrics ---
        if args.dataset == 'BraTSDataset':
            dice_scores = softmax_output_dice(output, target_cpu)
            hd95_scores = cal_hd95(output, target_cpu)
            vals.update(np.array(dice_scores))
            vals_hd95.update(np.array(hd95_scores))

    # --- compute mean metrics ---
    dice_mean = vals.avg
    hd95_mean = vals_hd95.avg
    runtime_mean = np.mean(runtimes)

    if use_wandb:
        wandb.log({
            "epoch": epoch_id,
            "val/WT_dice": dice_mean[0],
            "val/TC_dice": dice_mean[1],
            "val/ET_dice": dice_mean[2],
            "val/WT_hd95": hd95_mean[0],
            "val/TC_hd95": hd95_mean[1],
            "val/ET_hd95": hd95_mean[2],
            "val/dice_mean": dice_mean.mean(),
            "val/hd95_mean": hd95_mean.mean(),
            "val/runtime_avg": runtime_mean,
        })

    model.train()
    return dice_mean, hd95_mean

def computational_runtime(runtimes):
    # remove the maximal value and minimal value
    runtimes = np.array(runtimes)
    maxvalue = np.max(runtimes)
    minvalue = np.min(runtimes)
    nums = runtimes.shape[0] - 2
    meanTime = (np.sum(runtimes) - maxvalue - minvalue ) / nums
    fps = 1 / meanTime
    print('mean runtime:',meanTime,'fps:',fps)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
