import torch
import numpy as np
import random
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
    """
    Compute Dice coefficients for multi-class segmentation evaluation following BraTS convention.
    This function calculates Dice similarity coefficients for brain tumor segmentation,
    computing metrics for individual tumor regions and their aggregations.

    Args:
        output (torch.Tensor): Model predictions with shape (batch_size, height, width, depth).
            Expected to contain class labels (0-3) representing:
            - 0: Background
            - 1: Non-Enhancing Tumor Core / Necrotic (NET/NCR)
            - 2: Edema (ED)
            - 3: Enhancing Tumor Core (ET)
        target (torch.Tensor): Ground truth segmentation with same shape as output.
            Contains same class labels as output.
    Returns:
        tuple: A tuple containing two numpy arrays:
            - dice_separate (np.ndarray): Shape (batch_size, 3). Dice coefficients for individual 
              tumor regions (NET/NCR, Edema, Enhancing Tumor).
            - dice_evaluate (np.ndarray): Shape (batch_size, 4). Dice coefficients following BraTS 
              convention (Whole Tumor, Tumor Core, Enhancing Tumor, Enhancing Tumor with post-processing).
    Notes:
        - A small epsilon (1e-8) is added to prevent division by zero
        - Post-processing for ET applies thresholding based on volume (500 voxels)
        - All computations are performed over spatial dimensions (1,2,3)
    """
    eps = 1e-8
    assert len(output.size()) == 4, f'Wrong shape for network output: {output.shape} instead of (1,240,240,155)'
    assert len(target.size()) == 4, f'Wrong shape for segmentation target: {target.shape} instead of (1,240,240,155)'

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
