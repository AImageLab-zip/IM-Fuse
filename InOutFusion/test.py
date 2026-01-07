import torch
import os
import argparse
from pathlib import Path

from test_utils import AverageMeter, softmax_output_dice_class4,set_seed
from loader.Dataloader import Brain

from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms as T
from utils import tsfm_tfusion
from net.Network_InOut import RsInOut_U_Hemis3D

DEVICE = torch.device('cuda')
set_seed(42)
selected_modals = ['t1c', 't1n', 't2w', 't2f']

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

ordered_names = ['t1c', 't1','t2w','t2f']
mask_names = ['_'.join([ordered_names[i] for i in range(4) if mask[i]]) for mask in masks]
test_modals_list = [[ordered_names[i] for i in range(4) if mask[i]]for mask in masks]
datapath = args.datapath
test_file = Path(__file__).parent / 'datalist' / 'test.txt'
save_path = args.savepath

model = RsInOut_U_Hemis3D(in_channels=1, out_channels=4,
                           levels=4, feature_maps=8, method='TF', phase='test')
model.eval()
checkpoint = torch.load(args.resume,weights_only=False)  
model.load_state_dict(checkpoint['model'])   

output_path = f"{args.savepath}" # TODO change your naming convention if you want to include the epoch number as well
assert not os.path.isdir(output_path), f'{output_path} must be a file, not a directory'
if os.path.exists(output_path):
    os.remove(output_path)
total_score = AverageMeter()
with torch.no_grad():
    for i, mask in tqdm(enumerate(masks),desc='Evaluating all the masks'):

        test_set = Brain(test_file, selected_modals, args.datapath, inputs_transform=T.Compose([
                tsfm_tfusion.Normalize(),
                tsfm_tfusion.NpToTensor()
            ]),
                labels_transform=T.Compose([
                tsfm_tfusion.ToLongTensor()
            ]), t_join_transform=None, 
            join_transform=tsfm_tfusion.Compose([
        tsfm_tfusion.ThrowFirstZ(),
        tsfm_tfusion.RandomCrop(128)
    ]),
            phase='test',test_modals=test_modals_list[i])

        test_loader = DataLoader(dataset=test_set,batch_size=1,shuffle=False,num_workers=args.num_workers) 
        assert test_loader.batch_size == 1, 'keep batch size 1'

        mask_specific_score = AverageMeter()

        for element in tqdm(test_loader,total=len(test_loader),desc=f'Testing: {mask_names[i]}'):
            target = element[4].unsqueeze(dim=1).type(torch.float32).to(DEVICE).detach()
            m_d = element[6].to(DEVICE).detach()
            image = []
            for k in range(4):
                image.append(element[k].unsqueeze(dim=1).type(torch.float32).to(DEVICE).detach())

            output = model(image,m_d) 
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
        
