import argparse
from pathlib import Path
import os
import wandb
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

# Project imports
from networks import VNet
from datasets import BraTS_TrainCache, Cache,BraTS_Train

from loss import DiceCeLoss
from evaluate import eval_one_dice
from evaluate import test_single_case
from utils import set_seed

os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_DIR"] = "/tmp"       
os.environ["WANDB_DISABLE_CODE"] = "true"  

CROP_SIZE = (96, 128, 128)
STRIDE = tuple([x // 2 for x in list(CROP_SIZE)])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument(
    "--deterministic",
    type=int,
    default=1,
    help="whether use deterministic training",
)
parser.add_argument("--batch-size", type=int, default=4, help="batch size for dataloader")
parser.add_argument("--num-workers", type=int, default=4, help="number of workers for the dataloader")
parser.add_argument("--num-cls", type=int, default=4, help="the number of class")
parser.add_argument("--num-channels", type=int, default=4, help="input channels")
parser.add_argument(
    "--max-epoch", type=int, default=1000, help="maximum epoch number to train"
)

parser.add_argument("--data-path", type=str, required=True, help="dataset path")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")

parser.add_argument("--resume-checkpoint", type=str, default=None, help="if provided, the script will resume from that checkpoint")
parser.add_argument("--checkpoint-path", type=str, default="", help="checkpoints path")
parser.add_argument("--wandb-project-name",type=str, default=None,help="wandb project name")

parser.add_argument("--cache-dataset",action='store_true')
args = parser.parse_args()

# Arguments
num_cls = args.num_cls
num_channels = args.num_channels
max_epoch = args.max_epoch
batch_size = args.batch_size
num_workers = args.num_workers
resume_checkpoint = args.resume_checkpoint
checkpoint_path = Path(args.checkpoint_path)
wandb_project_name = args.wandb_project_name
data_path = args.data_path
seed=args.seed
cache_dataset=args.cache_dataset

set_seed(seed)
# Optional wandb initialization
if wandb_project_name is not None:
    wandb.init(project=wandb_project_name,
               name='pretraining'
               )
os.makedirs(checkpoint_path,exist_ok=True)

# Model initialization
model = VNet(
    n_channels=num_channels,
    n_classes=num_cls,
    n_filters=16,
    normalization="batchnorm",
).to(DEVICE)

model.train()
model.cuda()
if torch.cuda.is_available():
    if torch.cuda.get_device_capability(DEVICE)[0] >= 7:
        model.compile()

# Optimizer and scheduler
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
scheduler = LambdaLR(optimizer=optimizer,
                         lr_lambda=lambda epoch: (1-epoch/max_epoch)**0.9)

# Resuming logic
start_epoch = 0
if resume_checkpoint is not None:
    print(f'Resuming checkpoint {Path(resume_checkpoint).name}')
    checkpoint = torch.load(resume_checkpoint,weights_only=False)
    start_epoch = checkpoint["epoch"] + 1
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

# Dataset and dataloader
train_splitfile = (Path(__file__).resolve().parent).parent / 'datalist' / 'train.txt'
val_splitfile = (Path(__file__).resolve().parent).parent / 'datalist' / 'val.txt'
if cache_dataset:
    train_cache = Cache(data_file_path=data_path,splitfile=train_splitfile)
    val_cache = Cache(data_file_path=data_path,splitfile=val_splitfile)

    train_dataset = BraTS_TrainCache(train_cache)
    val_dataset = BraTS_TrainCache(val_cache)
else:
    train_dataset = BraTS_Train(data_file_path=data_path,splitfile=train_splitfile)
    val_dataset = BraTS_Train(data_file_path=data_path,splitfile=val_splitfile)

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True,persistent_workers=True,prefetch_factor=1)
val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=num_workers,pin_memory=True)

loss_criterion = DiceCeLoss(num_cls)
for epoch in range(start_epoch, max_epoch):
    # Training phase
    model.train()
    all_losses = []
    for batch in tqdm(train_loader):

        image, label = batch['vol'].float().to(DEVICE,non_blocking=True), batch['seg'].to(DEVICE,non_blocking=True)
        _, logits = model(image)
        _, _, loss = loss_criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        all_losses.append(loss.item())
        optimizer.step()
    loss_mean = sum(all_losses)/len(all_losses)
    if wandb_project_name is not None:
        wandb.log(
            {
                'train/loss':loss_mean,
                'train/lr':optimizer.param_groups[0]['lr'],
                'epoch':epoch
            }
        )
    scheduler.step()

    # Saving phase
    if (epoch +1) % 25 == 0 or (epoch +1) == max_epoch:
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler":scheduler.state_dict()
        }
        torch.save(checkpoint, checkpoint_path / f"ch_{epoch + 1}.pth.tar")

    # Evaluation Phase
    model.eval()
    dice_all_wt = []
    dice_all_tc = []
    dice_all_et = []
    dice_all_mean = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            image, label = batch['vol'].float().squeeze(0).numpy(), batch['seg'].squeeze(0).numpy()
            predict, _ = test_single_case(model, image, STRIDE, CROP_SIZE, num_cls)
            dice_wt, dice_tc, dice_et, dice_mean = eval_one_dice(predict, label)
            dice_all_wt.append(dice_wt)
            dice_all_tc.append(dice_tc)
            dice_all_et.append(dice_et)
            dice_all_mean.append(dice_mean)
    dice_all_wt = np.mean(np.array(dice_all_wt))
    dice_all_tc = np.mean(np.array(dice_all_tc))
    dice_all_et = np.mean(np.array(dice_all_et))
    dice_all_mean = np.mean(np.array(dice_all_mean))
    if wandb_project_name is not None:
        wandb.log(
            {
            'eval/dice_wt':dice_all_wt,
            'eval/dice_tc':dice_all_tc,
            'eval/dice_et':dice_all_et,
            'eval/dice_mean':dice_all_mean,
            }
        )
if cache_dataset:
    train_cache.close() # type: ignore
    val_cache.close()   # type: ignore