import torch
import os
import argparse
from pathlib import Path
from templates.dummies.dummy import DummyDataset,DummyModel
from test_utils import AverageMeter, softmax_output_dice_class4,set_seed
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
path = os.path.dirname(__file__)


args = parser.parse_args()
masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
        [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
        [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
        [True, True, True, True]]

mask_names = ['t2', 't1c', 't1', 'flair', 
             't1ce_t2', 't1ce_t1', 'flair_t1', 't1_t2', 'flair_t2', 'flair_t1ce',
             'flair_t1ce_t1', 'flair_t1_t2', 'flair_t1ce_t2', 't1ce_t1_t2',
             'flair_t1ce_t1_t2']

datapath = args.datapath
test_file = Path(__file__).parent / 'datalist' / 'test15splits.csv'
save_path = args.savepath

test_set = DummyDataset(datapath=datapath,splitfile=test_file)  #TODO replace with your dataset class
# batch_size MUST be == 1
test_loader = DataLoader(dataset=test_set,batch_size=1,shuffle=False,num_workers=args.num_workers) 
assert test_loader.batch_size == 1, 'keep batch size 1'

model = DummyModel().to(DEVICE) #TODO replace with your model class
model.eval()
#checkpoint = torch.load(args.resume)  
#model.load_state_dict(checkpoint)   #TODO remember to access the right dict key

output_path = f"{args.savepath}" # TODO change your naming convention if you want to include the epoch number as well
assert not os.path.isdir(output_path), f'{output_path} must be a file, not a directory'
if os.path.exists(output_path):
    os.remove(output_path)
total_score = AverageMeter()
with torch.no_grad():
    for i, mask in tqdm(enumerate(masks),desc='Evaluating all the masks'):
        mask_specific_score = AverageMeter()

        for element in tqdm(test_loader,total=len(test_loader),desc=f'Testing: {mask_names[i]}'):
            image = element['image'].to(DEVICE) # TODO: modify this to match your dataset class
            target = element['label'].to(DEVICE) # TODO: modify this to match your dataset class

            for idx, value in enumerate(mask):
                if not value:
                    image[:,idx] = 0 # TODO: implement your own masking logic

            output = model(image,mask) # TODO: implement your own masking logic
            output = F.softmax(output,dim=1) # TODO: apply softmax if necessary
            output = torch.argmax(output, dim=1) # TODO: if necessary, convert probabilities into labels
            
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
        
