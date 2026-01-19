import torch
import numpy as np
import random
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
from nets.cph import CPH
from einops import rearrange
import SimpleITK as sitk
def read_nii(path, sitk_type=None):
    if sitk_type is None:
        img = sitk.ReadImage(path)
    else:
        img = sitk.ReadImage(path, sitk_type)
    arr = sitk.GetArrayFromImage(img)  # (Z, H, W)
    return arr

def robust_norm(volume):
    v = volume.astype(np.float32)
    nz = v[v > 0]
    if nz.size > 0:
        lo, hi = np.percentile(nz, [1, 99])
        v = np.clip(v, lo, hi)
        nz = v[v > 0]
        if nz.size > 0 and nz.std() > 0:
            v = (v - nz.mean()) / (nz.std() + 1e-8)
    v_min, v_max = v.min(), v.max()
    if v_max > v_min:
        v = (v - v_min) / (v_max - v_min + 1e-8)
    else:
        v = np.zeros_like(v, dtype=np.float32)
    return v

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
    #o1 = (output[:,3]).float()
    t1 = (target == 1).float()
    #intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    #union1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    #net_ncr_dice = intersect1 / union1
    
    # label 2 --> Edema (ED)
    #o2 = (output == 2).float()
    t2 = (target == 2).float()
    #intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    #union2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    #edema_dice = intersect2 / union2
    

    # label 3 --> Enhancing Tumor Core (ET)
    o3 = (output[:,2]).float()
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
    o_whole = (output[:,0]).float()
    t_whole = t1 + t2 + t3 
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
    dice_whole = intersect_whole / denominator_whole

    # Tumor Core (TC)
    o_core = (output[:,1]).float()
    t_core = t1 + t3
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
    denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
    dice_core = intersect_core / denominator_core

    
    # Dice with the labels aggregated using the BraTS Convention
    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

    return 0, dice_evaluate.cpu().numpy()


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

class BaseDataSets_3D(Dataset):
    def __init__(self, root_dir:Path, split_file):
        self.root_dir = root_dir
        
        self.sub_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.sub_list.append(line)

        #self.images = [root_dir / f'imgs_{modality}/{sub}_{str(num).zfill(3)}.npz' for sub in sub_list for num in range(1,156)]
        #self.masks = [root_dir / f'masks_all/{sub}_{str(num).zfill(3)}.npz' for sub in sub_list for num in range(1,156)]
        

    def __len__(self):
        return len(self.sub_list)
        
    def __getitem__(self, idx):
        sub = self.sub_list[idx]
        t1c = self.root_dir / sub / f'{sub}-t1c.nii.gz'
        t1n = self.root_dir / sub / f'{sub}-t1n.nii.gz'
        t2f = self.root_dir / sub / f'{sub}-t2f.nii.gz'
        t2w = self.root_dir / sub / f'{sub}-t2w.nii.gz'
        mask = self.root_dir / sub / f'{sub}-seg.nii.gz'

        t1c = read_nii(t1c, sitk.sitkInt16) # shape --> [155,240,240]
        t1n = read_nii(t1n, sitk.sitkInt16)
        t2f = read_nii(t2f, sitk.sitkInt16)
        t2w = read_nii(t2w, sitk.sitkInt16)
        mask = read_nii(mask, sitk.sitkUInt8)

        t1c  = torch.from_numpy(robust_norm(t1c).transpose(1, 2, 0))
        t1n  = torch.from_numpy(robust_norm(t1n).transpose(1, 2, 0))
        t2f  = torch.from_numpy(robust_norm(t2f).transpose(1, 2, 0))
        t2w  = torch.from_numpy(robust_norm(t2w).transpose(1, 2, 0))

        mask = torch.from_numpy(mask.transpose(1,2,0))
        img = torch.stack([t1c,t1n,t2f,t2w],dim=0)
        # Center crop
        img = img[:,8:232,8:232]
        sample = {'image': img, 'target': mask}
        return sample
    
class CPH_3d(nn.Module):
    def __init__(self,batch_size:int):
        super().__init__()
        class InputAdapter(torch.nn.Module):
            def __init__(self, k: int):
                super().__init__()
                self.conv = torch.nn.Conv2d(k, 1, kernel_size=1, bias=False)
                with torch.no_grad():
                    self.conv.weight[:] = 1.0 / k
            def forward(self, x):
                return self.conv(x)
        net = torch.nn.Sequential(InputAdapter(4), CPH(n_classes=3)).to('cuda')
        net = torch.compile(net, mode="reduce-overhead")
        self.net = net.to(memory_format=torch.channels_last) # type:ignore
        
        self.batch_size = batch_size
        

    def forward(self, input):
        assert tuple(input.shape) == (1,4,240,240,155), f'Wrong shape for the input: {tuple(input.shape)} instead of (1,4,240,240,155)'
        input = input.squeeze(0)
        predictions = []
        for slice_idx in range(0,155,self.batch_size):
            start = slice_idx
            stop = min(slice_idx + self.batch_size, 155)

            sliced_input = input[:,:,:,start:stop]
            sliced_input = rearrange(sliced_input, 'C,H,W,D -> D,C,H,W').contiguous(memory_format=torch.channels_last)
            prediction = self.net(sliced_input)
            predictions.append(prediction)
        predictions = torch.cat(predictions,dim=0)
        predictions = rearrange(predictions,'D,C,H,W -> C,H,W,D').unsqueeze(0).contiguous()

        return predictions
            

            


