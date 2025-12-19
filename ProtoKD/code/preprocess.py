"""
preprocess data (nii.gz) to npy
"""

import os
import SimpleITK as sitk
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def read_nii(path):
    itkimg = sitk.ReadImage(path)
    npimg = sitk.GetArrayFromImage(itkimg)
    npimg = npimg.astype(np.float32)
    return npimg


def convert_label(gt):
    # This has been updated, since BraTS 2023 uses label 3 for the enhancing tumor
    # Enhancing tumor label: 3 --> 1
    # Peritumoral edema label: 2 --> 3
    # NCR/NET label: 1 --> 2
    new_gt = (gt == 3) * 1 + (gt == 1) * 2 + (gt == 2) * 3
    return new_gt


def zscore_nonzero(img):
    # use non-zero ROI mean and std to normalize
    mask = img.copy()
    mask[img > 0] = 1
    mean = np.sum(mask * img) / np.sum(mask)
    std = np.sqrt(np.sum(mask * (img - mean) ** 2) / np.sum(mask))
    img = (img - mean) / std
    return img

def preprocess_single_subject(arg):
    sub, compressed,output_path = arg
    t1n_path = sub / (sub.name + "-t1n.nii.gz")
    t2w_path = sub / (sub.name + "-t2w.nii.gz")
    t1c_path = sub / (sub.name + "-t1c.nii.gz")
    t2f_path = sub / (sub.name + "-t2f.nii.gz")
    seg_path = sub / (sub.name + "-seg.nii.gz")

    t1n_img = read_nii(t1n_path)
    t2w_img = read_nii(t2w_path)
    t1c_img = read_nii(t1c_path)
    t2f_img = read_nii(t2f_path)
    
    seg = read_nii(seg_path)
    seg = seg.astype(np.int8)

    t1n_img = zscore_nonzero(t1n_img)
    t2w_img = zscore_nonzero(t2w_img)
    t1c_img = zscore_nonzero(t1c_img)
    t2f_img = zscore_nonzero(t2f_img)
    seg = convert_label(seg)

    data = np.stack([t1n_img, t2w_img, t1c_img, t2f_img, seg])  # 5*155*240*240
    data = data[:, 5:145, 24:216, 24:216]  # crop to 5*140*192*192
    if compressed:
        np.savez_compressed(output_path / (sub.name + '.npz'), data=data)
    else:
        np.save(output_path / (sub.name + '.npy'), data)



    
parser = ArgumentParser()
parser.add_argument('--input-path',type=str,metavar='INPUT-PATH',required=True)
parser.add_argument('--output-path',type=str,metavar='OUTPUT-PATH',required=True)
parser.add_argument('--num-workers',type=int,metavar='NUM-WORKERS', default=1)
parser.add_argument('--compressed',action='store_true')
args = parser.parse_args()

input_path = Path(args.input_path)
output_path = Path(args.output_path)
num_workers = args.num_workers
compressed = args.compressed

if not output_path.exists():
    os.makedirs(output_path)

list_of_subjects = list(input_path.iterdir())
list_of_args=[(sub,compressed,output_path)for sub in list_of_subjects]
with ProcessPoolExecutor(max_workers=num_workers) as pool:
    results = list(tqdm(pool.map(preprocess_single_subject,list_of_args),total=len(list_of_args)))

