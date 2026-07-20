import importlib
import io
import logging
import os
import shutil
import sys
import uuid
import math

import random
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim import Optimizer
from PIL import Image
from sklearn.decomposition import PCA
from itertools import chain, combinations
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

plt.ioff()
plt.switch_backend('agg')


MODALITIES = [0,1,2,3]
def all_subsets(l):
    #Does not include the empty set
    return list(chain(*map(lambda x: combinations(l, x), range(1, len(l)+1))))

SUBSETS_MODALITIES = all_subsets(MODALITIES)

# print(SUBSETS_MODALITIES)
# [(0,), (1,), (2,), (3,), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (0, 1, 2, 3)]

def subset_idx(subset_size=[4]):
    idx_list = []
    for size in subset_size:
        if size == 1:
            start, end = 0, 4
        elif size == 2:
            start, end = 4, 10
        elif size == 3:
            start, end = 10, 14
        elif size == 4:
            start, end = 14, 15
        subset_index = np.random.choice(range(start, end), replace=False).tolist()
        if not subset_index in idx_list:
            idx_list.append(subset_index)
            
    return idx_list
####################################################################
# subset_size = np.random.choice(range(1,4), 1)
# subset_index_list = subset_idx(subset_size)
# for i in range (15):
#     print(SUBSETS_MODALITIES[i])
#     drop = np.array([True if k in SUBSETS_MODALITIES[i] else False for k in range(4)]) == False
#     print(drop)
####################################################################



def get_drop(subset_idx_list):
    drop_mod = (SUBSETS_MODALITIES == False)
    return drop_mod[subset_idx_list]

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def init_weights(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            #init.zeros_(m.bias.data)
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ModuleList):
        for l in m:
            init_weights(l)
            
def compute_sdm(seg_gt):
    """
    args:
        seg_gt: seg ground thruth (b, c, x, y, z)
    """
    sdm_gt = np.zeros(seg_gt.shape)
    for c in range(seg_gt.shape[1]):
        sdm_gt[:,c] = compute_per_channel_sdm(seg_gt[:,c])
    sdm_gt = torch.from_numpy(sdm_gt).float()
    
    return sdm_gt

def compute_per_channel_sdm(seg_gt):
    """
    compute the signed distance map of binary mask
    args:
        seg_gt: segmentation ground thruth (b, x, y, z)
    output: the Signed Distance Map (SDM)
    sdm(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    seg_gt = seg_gt.astype(np.uint8)
    normalized_sdf = np.zeros(seg_gt.shape)

    for b in range(seg_gt.shape[0]): # batch size
        posmask = seg_gt[b].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf[boundary==1] = 0
            normalized_sdf[b] = sdf
            # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
            # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf

class BaseWarmup(object):
    """Base class for all warmup schedules
    Arguments:
        optimizer (Optimizer): an instance of a subclass of Optimizer
        warmup_params (list): warmup paramters
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, warmup_params, last_step=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.warmup_params = warmup_params
        self.last_step = last_step
        self.dampen()

    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def dampen(self, step=None):
        """Dampen the learning rates.
        Arguments:
            step (int): The index of current step. (Default: None)
        """
        if step is None:
            step = self.last_step + 1
        self.last_step = step

        for group, params in zip(self.optimizer.param_groups, self.warmup_params):
            omega = self.warmup_factor(step, **params)
            group['lr'] *= omega

    def warmup_factor(self, step, **params):
        raise NotImplementedError


def get_warmup_params(warmup_period, group_count):
    if type(warmup_period) == list:
        if len(warmup_period) != group_count:
            raise ValueError(
                'size of warmup_period does not equal {}.'.format(group_count))
        for x in warmup_period:
            if type(x) != int:
                raise ValueError(
                    'An element in warmup_period, {}, is not an int.'.format(
                        type(x).__name__))
        warmup_params = [dict(warmup_period=x) for x in warmup_period]
    elif type(warmup_period) == int:
        warmup_params = [dict(warmup_period=warmup_period)
                         for _ in range(group_count)]
    else:
        raise TypeError('{} is not a list nor an int.'.format(
            type(warmup_period).__name__))
    return warmup_params


class LinearWarmup(BaseWarmup):
    """Linear warmup schedule.
    Arguments:
        optimizer (Optimizer): an instance of a subclass of Optimizer
        warmup_period (int or list): Warmup period
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, warmup_period, last_step=-1):
        group_count = len(optimizer.param_groups)
        warmup_params = get_warmup_params(warmup_period, group_count)
        super(LinearWarmup, self).__init__(optimizer, warmup_params, last_step)

    def warmup_factor(self, step, warmup_period):
        return min(1.0, (step+1) / warmup_period)

def prepare_validation(cutted_image, patch_size, overlap_stepsize):
    """Determine patches for validation."""

    patch_ids = []

    D, H, W, _ = cutted_image.shape

    drange = list(range(0, D-patch_size+1, overlap_stepsize))
    hrange = list(range(0, H-patch_size+1, overlap_stepsize))
    wrange = list(range(0, W-patch_size+1, overlap_stepsize))

    if (D-patch_size) % overlap_stepsize != 0:
        drange.append(D-patch_size)
    if (H-patch_size) % overlap_stepsize != 0:
        hrange.append(H-patch_size)
    if (W-patch_size) % overlap_stepsize != 0:
        wrange.append(W-patch_size)

    for d in drange:
        for h in hrange:
            for w in wrange:
                patch_ids.append((d, h, w))

    return patch_ids


def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.
    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    log_info(f"Saving last checkpoint to '{last_file_path}'")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        log_info(f"Saving best checkpoint to '{best_file_path}'")
        shutil.copyfile(last_file_path, best_file_path)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.
    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied
    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])

    return state


def save_network_output(output_path, output, logger=None):
    if logger is not None:
        logger.info(f'Saving network output to: {output_path}...')
    output = output.detach().cpu()[0]
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('predictions', data=output, compression='gzip')


loggers = {}


def get_logger(name, level=logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        loggers[name] = logger

        return logger


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


def find_maximum_patch_size(model, device):
    """Tries to find the biggest patch size that can be send to GPU for inference
    without throwing CUDA out of memory"""
    logger = get_logger('PatchFinder')
    in_channels = model.in_channels

    patch_shapes = [(64, 128, 128), (96, 128, 128),
                    (64, 160, 160), (96, 160, 160),
                    (64, 192, 192), (96, 192, 192)]

    for shape in patch_shapes:
        # generate random patch of a given size
        patch = np.random.randn(*shape).astype('float32')

        patch = torch \
            .from_numpy(patch) \
            .view((1, in_channels) + patch.shape) \
            .to(device)

        logger.info(f"Current patch size: {shape}")
        model(patch)


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]



def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    assert input.dim() == 4

    # expand the input tensor to Nx1xDxHxW before scattering
    input = input.unsqueeze(1)
    # create result tensor shape (NxCxDxHxW)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


def plot_segm(segm, ground_truth, plots_dir='.'):
    """
    Saves predicted and ground truth segmentation into a PNG files (one per channel).
    :param segm: 4D ndarray (CDHW)
    :param ground_truth: 4D ndarray (CDHW)
    :param plots_dir: directory where to save the plots
    """
    assert segm.ndim == 4
    if ground_truth.ndim == 3:
        stacked = [ground_truth for _ in range(segm.shape[0])]
        ground_truth = np.stack(stacked)

    assert ground_truth.ndim == 4

    f, axarr = plt.subplots(1, 2)

    for seg, gt in zip(segm, ground_truth):
        mid_z = seg.shape[0] // 2

        axarr[0].imshow(seg[mid_z], cmap='prism')
        axarr[0].set_title('Predicted segmentation')

        axarr[1].imshow(gt[mid_z], cmap='prism')
        axarr[1].set_title('Ground truth segmentation')

        file_name = f'segm_{str(uuid.uuid4())[:8]}.png'
        plt.savefig(os.path.join(plots_dir, file_name))


def convert_to_numpy(input, target):
    """
    Coverts input and target torch tensors to numpy ndarrays
    Args:
        input (torch.Tensor): 5D torch tensor
        target (torch.Tensor): 5D torch tensor
    Returns:
        tuple (input, target) tensors
    """
    assert isinstance(input, torch.Tensor), "Expected input to be torch.Tensor"
    assert isinstance(target, torch.Tensor), "Expected target to be torch.Tensor"

    input = input.detach().cpu().numpy()  # 5D
    target = target.detach().cpu().numpy()  # 5D

    return input, target