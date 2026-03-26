import torch
import numpy as np
import random
from math import ceil
import torch.nn as nn
import torch.nn.functional as F

class AverageMeter():
    """
    AverageMeter

    A utility class for computing and tracking running averages of numeric values.
    It can work with both scalars and arrays
    
    Attributes:
        val (float): The most recently updated value.
        avg (float): The cumulative average of all values.
        sum (float): The cumulative sum of all values.
        count (int): The total count of values added.

    Methods:
        reset(): Reinitialize all tracking variables to zero.
        update(val, n=1): Add a new value to the meter, optionally with a weight.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        val = np.array(val)   # ensure array-like

        if self.sum is None:
            self.sum = val * n
        else:
            self.sum += val * n

        self.count += n
        self.avg = self.sum / self.count
        self.val = val


def softmax_output_dice_class4(output, target):
    eps = 1e-8
    assert len(output.size()) == 3, f'Wrong shape for network output: {output.shape} instead of (155, 240, 240)'
    assert len(target.size()) == 3, f'Wrong shape for segmentation target: {target.shape} instead of (155, 240, 240)'
    output = output.unsqueeze(0)
    target = target.unsqueeze(0)
    # label 1 --> Non Enhancing Tumor Core / Necrotic (NET / NCR)
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    union1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    net_ncr_dice = intersect1 / union1
    
    # label 2 --> Edema (ED)
    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    union2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    edema_dice = intersect2 / union2
    
    # label 3 --> Enhancing Tumor Core (ET)
    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
    denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
    enhancing_dice = intersect3 / denominator3

    # Enhancing Tumor with post processing (ETpp)
    if torch.sum(o3) < 500:
       o4 = o3 * 0.0
    else:
       o4 = o3
    t4 = t3
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    union4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
    enhancing_dice_postpro = intersect4 / union4

    # Whole Tumor (WT)
    o_whole = o1 + o2 + o3 
    t_whole = t1 + t2 + t3 
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
    dice_whole = intersect_whole / denominator_whole

    # Tumor Core (TC)
    o_core = o1 + o3
    t_core = t1 + t3
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
    denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
    dice_core = intersect_core / denominator_core

    # Dice for the 4 separate structures
    dice_separate = torch.cat((torch.unsqueeze(net_ncr_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    
    # Dice with the labels aggregated using the BraTS Convention
    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

import torch
import torch.nn.functional as F
from math import ceil
from torch import nn

def predict_sliding(mode, net, img_list, tile_size, classes):
    image, image_res = img_list
    interp = nn.Upsample(size=tile_size, mode='trilinear', align_corners=True)
    image_size = image.shape
    overlap = 1/3

    strideHW = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)
    
    # Create on same device as input
    full_probs = torch.zeros((classes, image_size[2], image_size[3], image_size[4]), 
                             device=image.device)
    count_predictions = torch.zeros((classes, image_size[2], image_size[3], image_size[4]),
                                    device=image.device)

    for dep in range(tile_deps):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                y1 = int(row * strideHW)
                x1 = int(col * strideHW)
                d2 = min(d1 + tile_size[0], image_size[2])
                y2 = min(y1 + tile_size[1], image_size[3])
                x2 = min(x1 + tile_size[2], image_size[4])
                d1 = max(int(d2 - tile_size[0]), 0)
                y1 = max(int(y2 - tile_size[1]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)

                img = image[:, :, d1:d2, y1:y2, x1:x2]
                img_res = image_res[:, :, d1:d2, y1:y2, x1:x2]
                
                # Store actual crop size BEFORE padding
                actual_d = img.shape[2]
                actual_h = img.shape[3]
                actual_w = img.shape[4]
                
                padded_img = pad_image(img, tile_size)
                padded_img_res = pad_image(img_res, tile_size)
                
                with torch.no_grad():
                    padded_prediction, _, _ = net(padded_img, val=True, mode=mode)
                    padded_prediction = F.sigmoid(padded_prediction)
                    padded_prediction = interp(padded_prediction)  # Shape: (1, classes, D, H, W)
                    padded_prediction = padded_prediction[0]  # Shape: (classes, D, H, W)
                    
                    # Crop using ACTUAL dimensions, with channels FIRST
                    prediction = padded_prediction[:, 0:actual_d, 0:actual_h, 0:actual_w]
                
                count_predictions[:, d1:d2, y1:y2, x1:x2] += 1
                full_probs[:, d1:d2, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions.clamp(min=1)
    return full_probs
def pad_image(img, target_size):

    """Pad an image up to the target size."""
    deps_missing = target_size[0] - img.shape[2]
    rows_missing = target_size[1] - img.shape[3]
    cols_missing = target_size[2] - img.shape[4]
    padded_img = F.pad(img, (0, cols_missing, 0, rows_missing, 0, deps_missing), mode='constant', value=0)
    return padded_img


def mask_to_mode(mask: list[bool]):
    return ",".join(str(i) for i, flag in enumerate(mask) if flag)


def fix_segmentation(segmentation):
    assert len(segmentation.shape) == 4, f'Wrong shape for segmentation: {segmentation.shape} instead of (3, 155, 240, 240)'
    print(f'sum --> {segmentation[0].sum()} numel --> {(segmentation[0][segmentation[0]!=0]).numel()}')
    print(f'sum --> {segmentation[1].sum()} numel --> {(segmentation[1][segmentation[0]!=0]).numel()}')
    print(f'sum --> {segmentation[2].sum()} numel --> {(segmentation[2][segmentation[0]!=0]).numel()}')
    segmentation_rounded = torch.round(segmentation)
    
    out_segmentation = torch.zeros((segmentation_rounded.shape[1], segmentation_rounded.shape[2], segmentation_rounded.shape[3]),
                            device=segmentation_rounded.device,
                            dtype=segmentation_rounded.dtype)
    out_segmentation[segmentation_rounded[1]==1]=2
    out_segmentation[segmentation_rounded[2]==1]=1
    out_segmentation[segmentation_rounded[0]==1]=3
    return out_segmentation

