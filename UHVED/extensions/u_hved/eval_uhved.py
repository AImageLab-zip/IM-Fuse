#coding=utf-8
import argparse
import os
import time
import logging
import numpy as np
import wandb
import torch
import torch.optim
import sys
from tensorboardX import SummaryWriter
from utils.utils import setup_seed
from u_hved_net_torch import U_HVED
from data.transforms import *
from data.datasets_nii import Brats_loadall_test_nii, Brats_loadall_val_nii, Brats_loadall_nii
from utils.lr_scheduler import MultiEpochsDataLoader 
from utils.predict import test_softmax
from data.data_utils import init_fn

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--datapath', default=None, type=str)
parser.add_argument('--dataname', default='BRATS2023', type=str)
parser.add_argument('--savepath', default=None, type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--seed', default=999, type=int)
parser.add_argument('--debug', action='store_true', default=False)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
MODALITIES_img = ['T1', 'T1c', 'T2', 'Flair']

def main():
    ##########setting seed
    setup_seed(args.seed)
    ##########print args
    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True)

    ##########setting models
    num_cls = 4
    model = U_HVED(num_classes=num_cls)
    print (model)
    model = torch.nn.DataParallel(model).cuda()

    ########## Setting data
    train_file = 'datalist/train.txt'
    test_file = 'datalist/test15splits.csv'
    val_file = 'datalist/val15splits.csv'
    train_set = Brats_loadall_nii(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls, train_file=train_file)
    test_set = Brats_loadall_test_nii(root=args.datapath, test_file=test_file)
    val_set = Brats_loadall_val_nii(root=args.datapath, num_cls=num_cls, val_file=val_file)
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    val_loader = MultiEpochsDataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    ##########Resume Training
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch'] 
            
        ########## validation and test
        print('validate ...')
        with torch.no_grad():
            dice_score, seg_loss = test_softmax(
                val_loader,
                model,
                dataname = args.dataname)
        
        val_WT, val_TC, val_ET, val_ETpp = dice_score 
        val_dice = (val_ET + val_WT + val_TC)/3 
        print('Validate model epoch {}, WT = {:.3}, TC = {:.3}, ET = {:.3}, ETpp = {:.3}, val_dice  = {:.3}, loss = {:.3}'.format(epoch, val_WT.item(), val_TC.item(), val_ET.item(), val_ETpp.item(), val_dice.item(), seg_loss.cpu().item()))
                
        print('testing ...')
        with torch.no_grad():
            dice_score, seg_loss = test_softmax(
                test_loader,
                model,
                dataname = args.dataname)
        
        test_WT, test_TC, test_ET, test_ETpp = dice_score 
        test_dice = (test_ET + test_WT + test_TC)/3  
        print('Testing model epoch {}, WT = {:.3}, TC = {:.3}, ET = {:.3}, ET_pp = {:.3}, test_dice = {:.3}, loss = {:.3}'.format(epoch, test_WT.item(), test_TC.item(), test_ET.item(), test_ETpp.item(), test_dice.item(), seg_loss.cpu().item()))
        

if __name__ == '__main__':
    main()
