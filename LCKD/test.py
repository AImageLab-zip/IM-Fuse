import torch
import os
import argparse
from pathlib import Path
from test_utils import AverageMeter, softmax_output_dice_class4,set_seed,predict_sliding,mask_to_mode,fix_segmentation
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from BraTSDataSet import BraTSValDataSet
from DualNet import DualNet
import nibabel as nib

DEVICE = torch.device('cuda')
set_seed(42)

parser = argparse.ArgumentParser()

parser.add_argument('--datapath', required=True, type=str)
parser.add_argument('--savepath', required=True, type=str)
parser.add_argument('--resume', required=True, type=str)
parser.add_argument('--num-workers', default=8, type=int)
parser.add_argument('--test_file',default= Path(__file__).parent / 'datalist' / 'test15splits.csv',type=str)
path = os.path.dirname(__file__)


args = parser.parse_args()
masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
        [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
        [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
        [True, True, True, True]]

ordered_names = ['t2f', 't1','t1c','t2w']

mask_names = ['_'.join([ordered_names[i] for i in range(4) if mask[i]]) for mask in masks]
print(mask_names)
datapath = Path(args.datapath)
test_file = Path(args.test_file)
save_path = args.savepath

test_set = BraTSValDataSet(datapath=datapath,list_path=test_file) 
# batch_size MUST be == 1
test_loader = DataLoader(dataset=test_set,batch_size=1,shuffle=False,num_workers=args.num_workers) 
assert test_loader.batch_size == 1, 'keep batch size 1'

model = DualNet(norm_cfg='IN',activation_cfg='LeakyReLU',weight_std=True,num_classes=3,self_att=False,cross_att=False).to(DEVICE)
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
        
        mask_specific_score = AverageMeter()

        for element in tqdm(test_loader,total=len(test_loader),desc=f'Testing: {mask_names[i]}'):
            image = element['image'].to(DEVICE) # TODO: modify this to match your dataset class
            image_res = element['image_res'].to(DEVICE)
            target = element['label'].to(DEVICE) # TODO: modify this to match your dataset class

            mode = mask_to_mode(mask)
            output = predict_sliding(mode=mode,net=model,img_list=[image,image_res],tile_size=(80,160,160),classes=3)
            #output = F.softmax(output,dim=0)
            output = fix_segmentation(output) 
            target = fix_segmentation(target.squeeze(0))
            import nibabel as nib
            import numpy as np
            affine = np.array([
                    [1, 0, 0, 0],   # voxel size 1 mm in x
                    [0, 1, 0, 0],   # voxel size 1 mm in y
                    [0, 0, 1, 0],   # voxel size 1 mm in z
                    [0, 0, 0, 1]
                ], dtype=float)

            nib.save(nib.Nifti1Image(image[0,0].detach().cpu().numpy(),affine),'/homes/ocarpentiero/IM-Fuse/LCKD/test_outputs/image.nii.gz')
            nib.save(nib.Nifti1Image(output.detach().cpu().numpy(),affine),'/homes/ocarpentiero/IM-Fuse/LCKD/test_outputs/label.nii.gz')
            nib.save(nib.Nifti1Image(target.detach().cpu().numpy(),affine),'/homes/ocarpentiero/IM-Fuse/LCKD/test_outputs/target.nii.gz')
            _ , brats_dice = softmax_output_dice_class4(output=output,target=target)
            # val_WT, val_TC, val_ET, val_ETpp = brats_dice
            mask_specific_score.update(brats_dice)
        mask_score_avg = mask_specific_score.avg
        total_score.update(mask_score_avg)
        mask_score_avg = mask_score_avg[0]
        print(mask_score_avg)
        with open(output_path, 'a') as file:
            file.write(f'Available modals = {mask_names[i]:<21}--> WT = {mask_score_avg[0].item():.4f}, TC = {mask_score_avg[1].item():.4f}, ET = {mask_score_avg[2].item():.4f}, ETpp = {mask_score_avg[3].item():.4f}\n')

    avg_totalscore = total_score.avg[0]
    with open(output_path, 'a') as file:
            file.write(f'Avg scores {"":<29}--> WT = {mask_score_avg[0].item():.4f}, TC = {mask_score_avg[1].item():.4f}, ET = {mask_score_avg[2].item():.4f}, ETpp = {mask_score_avg[3].item():.4f}\n')
        
