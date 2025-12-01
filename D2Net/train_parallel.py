#coding=utf-8
import argparse
import os
import random
import torch
import torch.optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import datasets, transforms
from pathlib import Path

import numpy as npa
import models
from data.sampler import CycleSampler
from utils import Parser,criterions
from utils import lovasz_loss as lovasz
from predict import validate_softmax, AverageMeter

import sys
import ast
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from models import config
from loss import Losses, Loss_Region
import math
import gc
from data.transforms import *
from data.datasets import BraTS_TrainCache, Cache
import wandb
import traceback
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import pickle

#TODO mettere a posto tutti gli argomenti inutili
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='DMFNet_GDL_all', type=str,
                    help='Your detailed configuration of the network')
parser.add_argument('--gpu', default='0', type=str,
                    help='Supprot one GPU & multiple GPUs.')
parser.add_argument('--seed', default='1024', type=int)
parser.add_argument('--restore', default='model_last.pth', type=str)

parser.add_argument('--train_data_dir', default='./data/MICCAI_BraTS2018_TrainingData_gz', type=str)
parser.add_argument('--valid_data_dir', default='./data/MICCAI_BraTS2018_ValidationData_gz', type=str)
parser.add_argument('--test_data_dir', default='./data/MICCAI_BraTS2018_TestingData_gz', type=str)
parser.add_argument('--train_list', default=['train_0.txt', 'train_1.txt', 'train_2.txt'], type=list)
parser.add_argument('--train_valid_list', default=['valid_0.txt', 'valid_1.txt', 'valid_2.txt'], type=list)
parser.add_argument('--valid_list', default=['valid.txt'], type=list)

# Training hyper-parameters:
parser.add_argument('--criterion', choices=['sigmoid_dice_loss', 'softmax_dice_loss', 'FocalLoss'], default='sigmoid_dice_loss', type=str) 

parser.add_argument('--save_freq', default='20', type=float)
parser.add_argument('--workers', default=8, type=int)

parser.add_argument('--output_set', choices=['train_val','val','test'], default='val', type=str) # [train_val,val,test] as output of submission
parser.add_argument('--batch-size', default=2, type=int, help='Batch size')
parser.add_argument('--opt', default='Adam', type=str)
parser.add_argument('--lr', default='3e-4', type=float) # 1e-3
parser.add_argument('--warmup_epoch', default='20', type=float) # Warm-up for learning rate
parser.add_argument('--weight_decay', default='1e-5', type=float)
parser.add_argument('--amsgrad', default=True, type=bool)
parser.add_argument('--weight_type', default='square', type=str)
parser.add_argument('--eps', default=1e-5, type=float)
parser.add_argument('--dataset', choices=['BraTSDataset'], default='BraTSDataset', type=str)   # RandCrop3D((128,128,128)), \  112,112,112
parser.add_argument('--train_transforms', default='Compose([ \
                                                    RandCrop3D((128,128,128)), \
                                                    RandomRotion(10),  \
                                                    RandomIntensityChange((0.1,0.1)), \
                                                    RandomFlip(0), \
                                                    NumpyType((np.float32, np.int64)), \
                                                    ])', type=str)
                                                    
parser.add_argument('--test_transforms', default='Compose([ \
                                                    Pad((0, 16, 16, 5, 0)), \
                                                    NumpyType((np.float32, np.int64)), \
                                                    ])', type=str)    ### Only if args.net=='U2net' and config.num_pool_per_axis=[5,5,5], Pad-16

parser.add_argument('--setting', default='D2Net', type=str) # summary of all setting you want to record, to save as the name of logs

# Hyper-parameters of the Networks:
parser.add_argument('--net', default='DisenNet', choices=['DMFNet','Unet','U2net3d','DisenNet'], type=str) # name of the used networks
parser.add_argument('--in_channels', default='4', type=int)
parser.add_argument('--channels1', default='32', type=int)
parser.add_argument('--channels2', default='128', type=int) # 128
parser.add_argument('--groups', default='16', type=int) # 16
parser.add_argument('--norm', default='sync_bn', type=str)
parser.add_argument('--num_classes', default='4', type=int)

parser.add_argument('--unet_filter_num_list', default=[8,16,32,48,64], type=list) #ori:[16,32,48,64,96] or small:[8,16,32,48,64]

# Hyper-parameters for U2Net:
parser.add_argument('--use_lovasz', default=False, type=ast.literal_eval) #bool
parser.add_argument('--use_snapshot_ensemble', default=False, type=ast.literal_eval) #bool
parser.add_argument('--smooth_label', default=False, type=ast.literal_eval) #bool
parser.add_argument('--use_focal_loss', default=False, type=ast.literal_eval) #bool
parser.add_argument('--vis_badcase', default=False, type=ast.literal_eval) #bool
parser.add_argument('--R_loss', default=False, type=ast.literal_eval) #bool
parser.add_argument('--loss_balance', default=False, type=ast.literal_eval) #bool
parser.add_argument('--duration', default='20', type=int) #bool
parser.add_argument('--u2net_inchann', default=8, type=int) # Only for U2net3d, the in_channel_num

parser.add_argument('--valid_submission_only', default=False, type=ast.literal_eval) #bool

# Hyper-paremeters for DisenNet:
parser.add_argument('--DisenNet_indim', default=8, type=int) # 2/4
parser.add_argument('--AuxDec_dim', default=2, type=int)     # 2/1
parser.add_argument('--recon_w', default=1, type=float) # [2, 1, 0.5, 0.1]
parser.add_argument('--kl_w', default=1, type=float)    # [2, 1, 0.5, 0.1]
parser.add_argument('--use_distill', default=True, type=ast.literal_eval) #bool
parser.add_argument('--use_contrast', default=True, type=ast.literal_eval) #bool
parser.add_argument('--contrast_w', default=1, type=float)   # [2, 1, 0.5, 0.1]
parser.add_argument('--use_style_map', default=True, type=ast.literal_eval) #bool
parser.add_argument('--style_dim', default=16, type=int) # 8, dim of style vector

# Missing Modality Setting:
parser.add_argument('--miss_modal', default=True, type=ast.literal_eval) #bool
parser.add_argument('--use_Bernoulli_train', default=True, type=ast.literal_eval) #bool
parser.add_argument('--use_kd', default=True, type=ast.literal_eval) # Use knowledge distillation (Fea+Logit KD)
parser.add_argument('--fea_dim', default='8', type=int) #  dim of feature of Decoder to be distilled
parser.add_argument('--kd_logit_w', default=1, type=float)  # [30, 10, 1, 0.1]
parser.add_argument('--kd_fea_w', default=1, type=float)    # [2, 1, 0.5, 0.1]
parser.add_argument('--kd_fea_channel_w', default=1, type=float)   # # [2, 1, 0.5, 0.1]
parser.add_argument('--kd_channel_attn', default=False, type=ast.literal_eval) #bool  Like the paper by chunhua shen
parser.add_argument('--kd_dense_fea_attn', default=False, type=ast.literal_eval) #bool  our novel calculation on each feature map's dense channel
parser.add_argument('--affinity_kd', default=True, type=ast.literal_eval) #bool the affinity kd loss
parser.add_argument('--self_distill', default=False, type=ast.literal_eval) #bool
parser.add_argument('--self_distill_logit_w', default=1, type=float)  # [5, 1, 0.5, 0.1]
parser.add_argument('--self_distill_fea_w', default=1, type=float)  # [2, 1, 0.5, 0.1]

parser.add_argument('--use_freq_map', default=False, type=ast.literal_eval) # Frequency
parser.add_argument('--use_freq_channel', default=False, type=ast.literal_eval) # Frequency, band-pass filter
parser.add_argument('--freq_w', default=1, type=float)  # # [2, 1, 0.5, 0.1]
parser.add_argument('--use_freq_contrast', default=True, type=ast.literal_eval) # Frequency (simple), as part of constrastive loss

parser.add_argument('--saveroot', default='./fig/seg_result_save', type=str)# root_path of saving logs

# Newly added arguments
parser.add_argument('--datapath',type=str, required=True,help='Path to the preprocessed dataset')
parser.add_argument('--num-epochs', type=int, default=229,help='Number of epochs')
parser.add_argument('--wandb-project-name',type=str,default=None,help='If provided, the script will log on wandb')
parser.add_argument('--resume-checkpoint',type=str, default=None,help='If provided, the training will restart from the given checkpoint')
parser.add_argument('--num-workers',type=int, default=8,help='Number of workers for the dataloader')
parser.add_argument('--checkpoint-path',type=str, required=True ,help='Path for checkpoint saving')

# Optional: allow explicit backend override if desired
parser.add_argument('--dist-backend', type=str, default='nccl', help='Distributed backend')

args = parser.parse_args()

# Preparing the arguments
datapath = Path(args.datapath)
num_epochs = args.num_epochs
wandb_project_name = args.wandb_project_name
resume_checkpoint = Path(args.resume_checkpoint) if args.resume_checkpoint is not None else None
num_workers = args.num_workers
checkpoint_path = Path(args.checkpoint_path)

os.makedirs(checkpoint_path,exist_ok=True)

# -------------------------
# Distributed helpers
# -------------------------
def setup_distributed(backend='nccl'):
    """
    Initializes distributed process group using environment variables
    (compatible with torchrun).
    """
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method='env://')
        distributed = True
    else:
        world_size = 1
        rank = 0
        local_rank = 0
        distributed = False

    return distributed, world_size, rank, local_rank


def is_main_process(rank):
    return rank == 0


# wandb should only be initialized on the main process
if wandb_project_name is not None:
    # Temporary init; will be re-used only if rank=0 (see main).
    # This avoids failing before main runs in simple cases.
    pass
def broadcast_object(obj, src=0):
    """
    Broadcast a Python object from the source rank to all ranks.
    """
    if dist.get_rank() == src:
        data = pickle.dumps(obj)
        tensor = torch.tensor(list(data), dtype=torch.uint8, device='cuda')
        length = torch.tensor([tensor.numel()], dtype=torch.int64, device='cuda')
    else:
        tensor = torch.tensor([], dtype=torch.uint8, device='cuda')
        length = torch.tensor([0], dtype=torch.int64, device='cuda')

    # Broadcast the length first
    dist.broadcast(length, src)

    # Resize receive buffer on other ranks
    if dist.get_rank() != src:
        tensor = torch.empty(length.item(), dtype=torch.uint8, device='cuda')

    # Broadcast the data
    dist.broadcast(tensor, src)

    # Deserialize
    data = bytes(tensor.tolist())
    return pickle.loads(data)

def main():
    assert torch.cuda.is_available(), "Currently, only the CUDA version is supported"

    # -------------------------
    # 0. Distributed setup
    # -------------------------
    distributed, world_size, rank, local_rank = setup_distributed(args.dist-backend if hasattr(args, 'dist-backend') else 'nccl')

    # -------------------------
    # 1. Seeds (per-rank)
    # -------------------------
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # -------------------------
    # 2. Initialize model, optimizer, and loss
    # -------------------------
    model = models.DisenNet(
        inChans_list=[4],
        base_outChans=args.DisenNet_indim,
        num_class_list=[4],
        args=args
    ).cuda()

    # Wrap with DDP if distributed
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr * args.batch_size,
        weight_decay=args.weight_decay
    )
    criterion = criterions.sigmoid_dice_loss

    # -------------------------
    # 3. Resume from checkpoint if available
    # -------------------------
    start_epoch = 0
    if args.resume_checkpoint is not None:
        if os.path.exists(args.resume_checkpoint):
            if is_main_process(rank):
                print(f"Resuming checkpoint {Path(args.resume_checkpoint).name}")
            checkpoint = torch.load(args.resume_checkpoint, map_location='cuda')
            # For DDP, model may be wrapped; access .module if needed
            if isinstance(model, DDP):
                model.module.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"] + 1

    # -------------------------
    # 4. Initialize wandb (main process only)
    # -------------------------
    if wandb_project_name is not None and is_main_process(rank):
        wandb.init(
            project=wandb_project_name,
            name='training'
        )

    # -------------------------
    # 5. Load datasets
    # -------------------------
    if is_main_process(rank):
        print("Building datasets...")
        train_cache = Cache(data_file_path=Path(args.datapath), splitfile=Path(__file__).parent / 'datalist' / 'train15splits.csv')
    else:
        train_cache=None
    dist.barrier()
    train_cache = broadcast_object(train_cache,src=0)
    dist.barrier()
    train_set = BraTS_TrainCache(cache=train_cache, testing=False)

    # Distributed sampler
    if distributed:
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
    else:
        train_sampler = None

    try:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        if is_main_process(rank):
            print(f"Training samples: {len(train_set)}")

        # -------------------------
        # 6. Training loop
        # -------------------------
        num_epochs = args.num_epochs
        best_dice_mean = 0.0

        for epoch in tqdm(range(start_epoch, num_epochs), desc='Training the network', position=0, disable=not is_main_process(rank)):
            model.train()
            
            running_loss = 0.0
            num_batches = len(train_loader)

            # For DistributedSampler, set the epoch for proper shuffling
            if distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            total_loss_list = []
            seg_loss_list = []
            recon_loss_list = []
            kl_loss_list = []
            distill_loss_list = []
            kd_logit_loss_list = []
            kd_feature_loss_list = []
            contrast_loss_list = []
            freq_loss_list = []

            for batch_idx, data in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                position=1,
                disable=not is_main_process(rank)
            ):
                x = data['image'].cuda(non_blocking=True)
                target = data['label'].cuda(non_blocking=True)
                complete_x = x.clone()

                # Random missing modality simulation
                if args.miss_modal and args.use_Bernoulli_train:
                    random_miss = (torch.rand(4, device=x.device) > 0.5).float()
                    while random_miss.sum() == 0:
                        random_miss = (torch.rand(4, device=x.device) > 0.5).float()
                    miss_list = [x[:, i, ...] * random_miss[i] for i in range(4)]
                    x = torch.stack(miss_list, dim=1)

                # Forward pass
                outputs = model(x, complete_x, is_test=False)
                seg_out, binary_seg_out_all, deep_sup_fea, weight_recon_loss, weight_kl_loss, \
                    weight_recon_c_loss, weight_recon_s_loss, distill_loss, kd_loss, \
                    contrastive_loss, freq_loss, seg_aux = outputs

                kd_logit_loss, kd_fea_loss = kd_loss

                # Deep supervision aggregation
                total_seg_loss = torch.zeros(1, device='cuda')
                if config.deep_supervision:
                    feature_maps = [deep_sup_fea[i] for i in range(config.num_pool_per_axis[0] - 2)]
                    feature_maps += [binary_seg_out_all, seg_aux, seg_out]
                    weights = np.array([1, 0.5, 0.5])
                    weights = np.append(weights, [1 / (2 ** (i + 1)) for i in range(config.num_pool_per_axis[0] - 3, -1, -1)])
                    weights /= weights.sum()
                else:
                    feature_maps = [seg_aux, binary_seg_out_all, seg_out]
                    weights = np.array([1, 0.5, 0.5])
                    weights /= weights.sum()

                # Segmentation losses
                _loss_fn = Losses()
                for iii, feature_map in enumerate(feature_maps):
                    b, c, h, w, d = feature_map.shape
                    _, h1, w1, d1 = target.shape
                    if d != d1:
                        _target = nn.MaxPool3d(int(h1 / h), stride=int(h1 / h))(target.float()).long()
                    else:
                        _target = target
                    seg_loss = _loss_fn(feature_map, _target, datasets=args.dataset, use_dice=True, ce=True)
                    total_seg_loss += seg_loss * weights[-1 - iii]

                # Total loss
                loss = (
                    total_seg_loss
                    + weight_recon_loss.mean()
                    + weight_kl_loss.mean()
                    + distill_loss.mean()
                    + kd_logit_loss.mean()
                    + kd_fea_loss.mean()
                    + contrastive_loss.mean()
                    + freq_loss.mean()
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total_loss_list.append(loss.detach())
                seg_loss_list.append(total_seg_loss.detach())
                recon_loss_list.append(weight_recon_loss.mean().detach())
                kl_loss_list.append(weight_kl_loss.mean().detach())
                distill_loss_list.append(distill_loss.mean().detach())
                kd_logit_loss_list.append(kd_logit_loss.mean().detach())
                kd_feature_loss_list.append(kd_fea_loss.mean().detach())
                contrast_loss_list.append(contrastive_loss.mean().detach())
                freq_loss_list.append(freq_loss.mean().detach())

            # WandB logging (only main process)
            if args.wandb_project_name and is_main_process(rank):
                wandb.log({
                    "train/total_loss": torch.stack(total_loss_list).mean().cpu().item(),
                    "train/seg_loss": torch.stack(seg_loss_list).mean().cpu().item(),
                    "train/recon_loss": torch.stack(recon_loss_list).mean().cpu().item(),
                    "train/KL_loss": torch.stack(kl_loss_list).mean().cpu().item(),
                    "train/distill_loss": torch.stack(distill_loss_list).mean().cpu().item(),
                    "train/KD_logit_loss": torch.stack(kd_logit_loss_list).mean().cpu().item(),
                    "train/KD_feature_loss": torch.stack(kd_feature_loss_list).mean().cpu().item(),
                    "train/contrast_loss": torch.stack(contrast_loss_list).mean().cpu().item(),
                    "train/freq_loss": torch.stack(freq_loss_list).mean().cpu().item(),
                    "epoch": epoch,
                })

            epoch_loss = running_loss / num_batches

            if is_main_process(rank):
                print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

            # Validation code (if re-enabled) should also be run only on rank 0
            # to avoid redundant evaluation and log duplication.

            # Save every 25 epochs or at the final epoch (main process only)
            if is_main_process(rank):
                if (epoch + 1) % 25 == 0 or (epoch + 1) == num_epochs:
                    checkpoint_name = f"checkpoint_epoch_{epoch+1}.pth.tar"
                    # If model is DDP-wrapped, save the underlying module
                    state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                    torch.save({
                        'epoch': epoch,
                        'state_dict': state_dict,
                        'optimizer': optimizer.state_dict(),
                    }, checkpoint_path / checkpoint_name)
                    print(f"Checkpoint saved: {checkpoint_name}")   

        if is_main_process(rank):
            print("Training completed.")
    except:
        traceback.print_exc()
        train_cache.close()
        # valid_cache.close()  # If re-enabled above

    finally:
        train_cache.close()
        if 'dist' in globals() and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9, warmup_epoch=20, lr_min=1e-8, _type='exp', use_warmup=False): 
    lr_max = INIT_LR
    
    if use_warmup:
        if _type == 'exp':
            if epoch >= warmup_epoch:
                lr = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)
            else:
                lr = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8) * epoch / warmup_epoch
        else:
            if epoch >= warmup_epoch:
                lr = lr_min + (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / MAX_EPOCHES)) / 2       # Cosine Annealing
            else:
                lr = lr_min + (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / MAX_EPOCHES)) / 2 * epoch / warmup_epoch
    else:
        if _type == 'exp':
            lr = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)
        else:
            lr = lr_min + (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / MAX_EPOCHES)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def snapshot_ensemble_lr(optimizer, epoch, CYCLE=4, LR_INIT=0.001, LR_MIN=0.0001):
    scheduler = lambda x: ((LR_INIT-LR_MIN)/2)*(np.cos(np.pi*(np.mod(x-1,CYCLE)/(CYCLE)))+1)+LR_MIN
    for param_group in optimizer.param_groups:
        param_group['lr'] = scheduler(epoch)


def bce_loss(prediction, label, smooth_label=False):
    label = label.long()
    mask = label.float()
    
    if smooth_label:
        num_positive = torch.sum((mask!=0).float()).float()
        num_negative = torch.sum((mask==0).float()).float()
        mask[mask > 0] = 1.0 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    else:
        num_positive = torch.sum((mask==1).float()).float()
        num_negative = torch.sum((mask==0).float()).float()
        mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
        mask[mask == 2] = 0
    cost = torch.nn.functional.binary_cross_entropy(prediction.float(),label.float(), weight=mask, reduce=False)
    
    return torch.sum(cost) / (num_positive+1e-6)


def weighted_mse_loss(prediction, label, smooth_label=True):
    mask = label.float()
    num_positive = torch.sum((mask!=0).float()).float()
    num_negative = torch.sum((mask==0).float()).float()
    mask[mask != 0] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    loss = torch.nn.functional.mse_loss(prediction.float(),label.float(),reduce=False, size_average=False)
    
    return torch.sum(loss) / (num_positive+1e-4)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    main()
