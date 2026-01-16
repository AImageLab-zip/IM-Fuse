import torch
import os
import argparse
from pathlib import Path
from templates.dummies.dummy import DummyDataset,DummyModel
from test_utils import AverageMeter, softmax_output_dice_class4,set_seed, BaseDataSets_3D, CPH_3d
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

DEVICE = torch.device('cuda')
set_seed(42)

parser = argparse.ArgumentParser()

parser.add_argument('--datapath', required=True, type=str)
parser.add_argument('--savepath', required=True, type=str)
parser.add_argument('--resume', required=True, type=str)
parser.add_argument('--num-workers', default=8, type=int)
parser.add_argument('--batch-size', default=31, type=int)
path = os.path.dirname(__file__)


args = parser.parse_args()
masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
        [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
        [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
        [True, True, True, True]]

ordered_names = ['t1c', 't1n','t2w','t2f']
mask_names = ['_'.join([ordered_names[i] for i in range(4) if mask[i]]) for mask in masks]

datapath = args.datapath
test_file = Path(__file__).parent / 'datalist' / 'test.txt'
save_path = args.savepath

test_set = BaseDataSets_3D(root_dir=datapath, split_file=test_file)
# batch_size MUST be == 1
test_loader = DataLoader(dataset=test_set,batch_size=1,shuffle=False,num_workers=args.num_workers) 
assert test_loader.batch_size == 1, 'keep batch size 1'

model = CPH_3d(batch_size=31).to(DEVICE) 
model.eval()
checkpoint = torch.load(args.resume)  
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
            target = element['target'].to(DEVICE) # TODO: modify this to match your dataset class

            for idx, value in enumerate(mask):
                if not value:
                    image[:,idx] = 0 

            output = model(image) 
            output = F.sigmoid(output) 
            output = (output > 0.5)
            
            # output and target must have shape (1,240,240,155)
            #TODO REMEMBER THAT THIS MODEL PREDICTS THE 3 CLASSES DIRECTLY
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
        
