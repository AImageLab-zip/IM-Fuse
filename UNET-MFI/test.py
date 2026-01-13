import torch
import os
import argparse
from pathlib import Path
from templates.dummies.dummy import DummyDataset,DummyModel
from test_utils import AverageMeter, softmax_output_dice_class4,set_seed
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from Ddataset import BraTSDataset
from transforms import *
from Model import no_share_unet

DEVICE = torch.device('cuda')
set_seed(42)
H, W, T = 240, 240, 155
patch_size = 120
overlap = 40
use_TTA = True
parser = argparse.ArgumentParser()

parser.add_argument('--datapath', required=True, type=str)
parser.add_argument('--savepath', required=True, type=str)
parser.add_argument('--resume', required=True, type=str)
parser.add_argument('--num-workers', default=8, type=int)
path = os.path.dirname(__file__)


args = parser.parse_args()
masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
        [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
        [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
        [True, True, True, True]]

ordered_names = ['t2f', 't1c','t1n','t2w']
mask_names = ['_'.join([ordered_names[i] for i in range(4) if mask[i]]) for mask in masks]
masks_int = [np.array(mask).astype(np.int32)for mask in masks]

datapath = args.datapath
test_file = Path(__file__).parent / 'datalist' / 'test.txt'
save_path = args.savepath

model = no_share_unet(in_channel=1, out_channel=3, diff=True,deepSupvision=True).to(DEVICE)
model.eval()
checkpoint = torch.load(args.resume,weights_only=False)  
model.load_state_dict(checkpoint['model'])

output_path = f"{args.savepath}"
assert not os.path.isdir(output_path), f'{output_path} must be a file, not a directory'
if os.path.exists(output_path):
    os.remove(output_path)
total_score = AverageMeter()

with torch.no_grad():
    for i, mask in tqdm(enumerate(masks),desc='Evaluating all the masks'):

        test_set = BraTSDataset(test_file, root=datapath, mode='test', for_train=False, code = masks_int[i],
                                transforms='Compose([Pad((0, 0, 0, 5, 0)),NumpyType((np.float32, np.int64)),])')
        # batch_size MUST be == 1
        test_loader = DataLoader(dataset=test_set,batch_size=1,shuffle=False,num_workers=args.num_workers) 
        assert test_loader.batch_size == 1, 'keep batch size 1'
        mask_specific_score = AverageMeter()

        for i, (x1, x2, x3, x4, target, mask) in tqdm(enumerate(test_loader), total=len(test_loader)):  ##xi:b*1*240*240*160
            x1, x2, x3, x4 = x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE), x4.to(DEVICE)
            mask = mask.to(DEVICE)
            b,c,h,w,l = x1.shape
            cur_ret = torch.zeros((b,3,h,w,l)).to(DEVICE)
            cur_count = torch.zeros((b,3,h,w,l)).to(DEVICE)
            for row in range(0,240-patch_size+1,overlap):
                for col in range(0,240-patch_size+1,overlap):
                    for height in range(0,160-patch_size+1,overlap):
                        cur_x1, cur_x2, cur_x3, cur_x4 = x1[:,:,row:row+patch_size,col:col+patch_size,height:height+patch_size],\
                                                            x2[:,:,row:row+patch_size,col:col+patch_size,height:height+patch_size],\
                                                            x3[:,:,row:row+patch_size,col:col+patch_size,height:height+patch_size],\
                                                            x4[:,:,row:row+patch_size,col:col+patch_size,height:height+patch_size]
                        if not use_TTA:
                            cur_output = model(cur_x1, cur_x2, cur_x3, cur_x4, mask)[-5:]
                            cur_ret[:, :, row:row + patch_size, col:col + patch_size, height:height + patch_size] += \
                                (cur_output[0] + cur_output[1] + cur_output[2] + cur_output[3] + cur_output[4]) / 5
                        else:
                            cur_output = sum(model(cur_x1, cur_x2, cur_x3, cur_x4, mask)[-5:])/5
                            cur_output += (sum(model(cur_x1.flip(dims=(2,)), cur_x2.flip(dims=(2,)), cur_x3.flip(dims=(2,)), cur_x4.flip(dims=(2,)), mask)[-5:])/5).flip(dims=(2,))
                            cur_output += (sum(
                                model(cur_x1.flip(dims=(3,)), cur_x2.flip(dims=(3,)), cur_x3.flip(dims=(3,)),
                                        cur_x4.flip(dims=(3,)), mask)[-5:]) / 5).flip(dims=(3,))
                            cur_output += (sum(
                                model(cur_x1.flip(dims=(4,)), cur_x2.flip(dims=(4,)), cur_x3.flip(dims=(4,)),
                                        cur_x4.flip(dims=(4,)), mask)[-5:]) / 5).flip(dims=(4,))
                            cur_output += (sum(
                                model(cur_x1.flip(dims=(2,3)), cur_x2.flip(dims=(2,3)), cur_x3.flip(dims=(2,3)),
                                        cur_x4.flip(dims=(2,3)), mask)[-5:]) / 5).flip(dims=(2,3))
                            cur_output += (sum(
                                model(cur_x1.flip(dims=(2,4)), cur_x2.flip(dims=(2,4)), cur_x3.flip(dims=(2,4)),
                                        cur_x4.flip(dims=(2,4)), mask)[-5:]) / 5).flip(dims=(2,4))
                            cur_output += (sum(
                                model(cur_x1.flip(dims=(3,4)), cur_x2.flip(dims=(3,4)), cur_x3.flip(dims=(3,4)),
                                        cur_x4.flip(dims=(3,4)), mask)[-5:]) / 5).flip(dims=(3,4))
                            cur_output += (sum(
                                model(cur_x1.flip(dims=(2,3, 4)), cur_x2.flip(dims=(2,3, 4)), cur_x3.flip(dims=(2,3, 4)),
                                        cur_x4.flip(dims=(2,3, 4)), mask)[-5:]) / 5).flip(dims=(2,3, 4))

                            cur_output /= 8.0
                            cur_ret[:, :, row:row + patch_size, col:col + patch_size, height:height + patch_size] += cur_output
                        cur_count[:,:,row:row+patch_size,col:col+patch_size,height:height+patch_size] += 1
                cur_ret /= cur_count ##b*3*240*240*160
                cur_ret = torch.sigmoid(cur_ret)
                output = cur_ret[:, :, :H, :W, :T].cpu().numpy() ##b*3*240*240*155
                target = target[:, :, :H, :W, :T].numpy() ##b*3*240*240*155

            
            _ , brats_dice = softmax_output_dice_class4(output=output,target=target)
            # val_WT, val_TC, val_ET, val_ETpp = brats_dice
            mask_specific_score.update(brats_dice)
        mask_score_avg = mask_specific_score.avg
        total_score.update(mask_score_avg)
        mask_score_avg = mask_score_avg[0]
        with open(output_path, 'a') as file:
            file.write(f'Available modals = {mask_names[i]:<21}--> WT = {mask_score_avg[0].item():.4f}, TC = {mask_score_avg[1].item():.4f}, ET = {mask_score_avg[2].item():.4f}, ETpp = {mask_score_avg[3].item():.4f}\n')
        
    avg_totalscore = total_score.avg[0]
    with open(output_path, 'a') as file:
            file.write(f'Avg scores {"":<29}--> WT = {mask_score_avg[0].item():.4f}, TC = {mask_score_avg[1].item():.4f}, ET = {mask_score_avg[2].item():.4f}, ETpp = {mask_score_avg[3].item():.4f}\n')
        
