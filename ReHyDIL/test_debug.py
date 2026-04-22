import torch
import os
import argparse
from pathlib import Path

from test_utils import AverageMeter, softmax_output_dice_class4, set_seed, BaseDataSets_3D, CPH_3d
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
import numpy as np
import nibabel as nib

os.makedirs("/homes/ocarpentiero/IM-Fuse/ReHyDIL/outputs/target", exist_ok=True)
os.makedirs("/homes/ocarpentiero/IM-Fuse/ReHyDIL/outputs/output", exist_ok=True)
DEVICE = torch.device('cuda')
set_seed(42)

parser = argparse.ArgumentParser()

parser.add_argument('--datapath', required=True, type=Path)
parser.add_argument('--savepath', required=True, type=Path)
parser.add_argument('--resume', required=True, type=Path)
parser.add_argument('--num-workers', default=8, type=int)
parser.add_argument('--slice-batch-size', default=31, type=int)

path = os.path.dirname(__file__)

args = parser.parse_args()
masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False],
         [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True],
         [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
masks = [[True,False,False,False]]
ordered_names = ['t1c', 't1n', 't2f', 't2w']
mask_names = ['_'.join([ordered_names[i] for i in range(4) if mask[i]]) for mask in masks]

datapath = args.datapath
test_file = Path(__file__).parent / 'datalist' / 'test.txt'
save_path = args.savepath

test_set = BaseDataSets_3D(root_dir=datapath, split_file=test_file)
# batch_size MUST be == 1
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=args.num_workers)
assert test_loader.batch_size == 1, 'keep batch size 1'

model = CPH_3d(args.slice_batch_size).to(DEVICE)
model.eval()
checkpoint = torch.load(args.resume, weights_only=False)['model']
if any(k.startswith("_orig_mod.") for k in checkpoint.keys()):
    checkpoint = {k.replace("_orig_mod.", "", 1): v for k, v in checkpoint.items()}
model.load_checkpoint(checkpoint)

output_path = f"{args.savepath}"
assert not os.path.isdir(output_path), f'{output_path} must be a file, not a directory'
if os.path.exists(output_path):
    os.remove(output_path)
total_score = AverageMeter()


def save_nifti(volume_hwz: torch.Tensor, path: str) -> None:
    vol_np = volume_hwz.detach().cpu().numpy().astype(np.uint8)
    nii = nib.Nifti1Image(vol_np, affine=np.eye(4))
    nib.save(nii, path)


with torch.no_grad():
    for i, mask in tqdm(enumerate(masks), desc='Evaluating all the masks'):
        mask_specific_score = AverageMeter()

        for j, element in tqdm(enumerate(test_loader), total=len(test_loader), desc=f'Testing: {mask_names[i]}'):

            image = element['image'].to(DEVICE).float()

            image[:,1] = image[:,0]
            image[:,2] = image[:,0]
            image[:,3] = image[:,0]
            target = element['target'].to(DEVICE)

            '''            for idx, value in enumerate(mask):
                if not value:
                    image[:, idx] = 0'''

            output = model(image)
            output = F.sigmoid(output)
            
            output = F.pad(output, (0, 0,  # D (no padding)
                                  8, 8,  # W
                                  8, 8))  # H
            output = (output > 0.5)

            output_napari = torch.zeros_like(target.squeeze(0))
            output_napari[output[0,0]!=0] = 2
            output_napari[output[0,1]!=0] = 1
            output_napari[output[0,2]!=0] = 3
            target_napari = target.squeeze(0)

            save_nifti(
                target_napari,
                f"/homes/ocarpentiero/IM-Fuse/ReHyDIL/outputs/target/sample_{j}_target.nii.gz",
            )
            save_nifti(
                output_napari,
                f"/homes/ocarpentiero/IM-Fuse/ReHyDIL/outputs/output/sample_{j}_output.nii.gz",
            )

            _, brats_dice = softmax_output_dice_class4(output=output, target=target)
            # val_WT, val_TC, val_ET, val_ETpp = brats_dice
            mask_specific_score.update(brats_dice)
        mask_score_avg = mask_specific_score.avg
        total_score.update(mask_score_avg)
        mask_score_avg = mask_score_avg[0]
        with open(output_path, 'a') as file:
            file.write(
                f'Available modals = {mask_names[i]:<21}--> WT = {mask_score_avg[0].item():.4f}, TC = {mask_score_avg[1].item():.4f}, ET = {mask_score_avg[2].item():.4f}, ETpp = {mask_score_avg[3].item():.4f}\n')

    avg_totalscore = total_score.avg[0]
    with open(output_path, 'a') as file:
        file.write(
            f'Avg scores {"":<29}--> WT = {mask_score_avg[0].item():.4f}, TC = {mask_score_avg[1].item():.4f}, ET = {mask_score_avg[2].item():.4f}, ETpp = {mask_score_avg[3].item():.4f}\n')
