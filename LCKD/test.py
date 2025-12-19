import torch

import os
import argparse
from pathlib import Path

import torch
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn

import os
import os.path as osp
from DualNet import DualNet
from BraTSDataSet import BraTSEvalDataSet, my_collate
import timeit
import loss_Dual as loss
from engine import Engine
from math import ceil
from pathlib import Path
from DualNet import conv3x3x3
import wandb
import traceback
from tqdm import tqdm
from predict import AverageMeter, test_softmax

parser = argparse.ArgumentParser()

parser.add_argument('--savepath', required=True, type=str)
parser.add_argument('--resume', required=True, type=str)
parser.add_argument('--datapath', required=True, type=str)
path = os.path.dirname(__file__)

test_file =Path(__file__).parent / 'datalist' / 'test15slits.csv'
if __name__ == '__main__':
    args = parser.parse_args()
    masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
    mask_name = ['t2', 't1c', 't1', 'flair', 
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']

    datapath = args.datapath
    save_path = Path(args.savepath)
    num_cls = 4
    dataname = args.dataname
    index = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    test_loader =  torch.utils.data.DataLoader(
                            BraTSEvalDataSet(args.datapath, args.data_list),
                            batch_size=1, shuffle=False, pin_memory=True)
    
    model = DualNet(args=args, norm_cfg='IN', activation_cfg='LeakyReLU',
                            num_classes=3, weight_std=True, self_att=False, cross_att=False)
    
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    output_path = save_path / 'results.txt'

    test_score = AverageMeter()
    with torch.no_grad():

        print('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(masks[index*5:(index+1)*5]):
            print('{}'.format(mask_name[i]))
            dice_score = test_softmax(
                            test_loader,
                            model,
                            dataname = dataname,
                            feature_mask = mask,
                            compute_loss=False,
                            save_masks=True,
                            save_dir=save_path,
                            index = index)
            val_WT, val_TC, val_ET, val_ETpp = dice_score
            
            with open(output_path, 'a') as file:
                file.write('Performance missing scenario = {}, WT = {:.4f}, TC = {:.4f}, ET = {:.4f}, ETpp = {:.4f}\n'.format(mask, val_WT.item(), val_TC.item(), val_ET.item(), val_ETpp.item()))

            test_score.update(dice_score)
        print('Avg scores: {}'.format(test_score.avg))
        with open(output_path, 'a') as file:
                file.write('Avg scores: {}'.format(test_score.avg))

