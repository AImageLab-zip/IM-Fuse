# -*- coding: utf-8 -*-
import os
import numpy as np
import SimpleITK as sitk
import skimage.io as io
import warnings
from argparse import ArgumentParser
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def read_nii(path, sitk_type=None):
    if sitk_type is None:
        img = sitk.ReadImage(path)
    else:
        img = sitk.ReadImage(path, sitk_type)
    arr = sitk.GetArrayFromImage(img)  # (Z, H, W)
    return arr

def robust_norm(volume):
    v = volume.astype(np.float32)
    nz = v[v > 0]
    if nz.size > 0:
        lo, hi = np.percentile(nz, [1, 99])
        v = np.clip(v, lo, hi)
        nz = v[v > 0]
        if nz.size > 0 and nz.std() > 0:
            v = (v - nz.mean()) / (nz.std() + 1e-8)
    v_min, v_max = v.min(), v.max()
    if v_max > v_min:
        v = (v - v_min) / (v_max - v_min + 1e-8)
    else:
        v = np.zeros_like(v, dtype=np.float32)
    return v

def crop_to_224_centered_on_mask(volumes, mask, size=224):
    assert len(volumes) > 0
    Z, H, W = mask.shape
    union2d = (mask > 0).any(axis=0).astype(np.uint8)  # (H,W)

    if union2d.any():
        ys, xs = np.where(union2d > 0)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    else:
        cy, cx = H // 2, W // 2

    half = size // 2
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half

    pad_y0, pad_y1 = max(0, -y0), max(0, y1 - H)
    pad_x0, pad_x1 = max(0, -x0), max(0, x1 - W)

    def pad_and_crop(arr):
        if pad_y0 or pad_y1 or pad_x0 or pad_x1:
            arr = np.pad(arr,
                         ((0, 0), (pad_y0, pad_y1), (pad_x0, pad_x1)),
                         mode='constant', constant_values=0)
        yy0, yy1 = y0 + pad_y0, y1 + pad_y0
        xx0, xx1 = x0 + pad_x0, x1 + pad_x0
        return arr[:, yy0:yy1, xx0:xx1]

    out_vols = [pad_and_crop(v) for v in volumes]
    out_mask = pad_and_crop(mask)
    for v in out_vols + [out_mask]:
        assert v.shape[1] == size and v.shape[2] == size, f"Crop failed: got {v.shape}"
    return out_vols, out_mask

def build_labels_3ch(mask_slice):
    WT = (mask_slice > 0).astype(np.uint8)
    TC = ((mask_slice == 1) | (mask_slice == 3)).astype(np.uint8)
    ET = (mask_slice == 3).astype(np.uint8)
    all_label3 = np.stack([WT, TC, ET], axis=0).astype(np.uint8)
    return WT, all_label3

def to_slice_id(n):
    s = str(n)
    if len(s) == 1: return '00' + s
    if len(s) == 2: return '0' + s
    return s

def process_single_subject(case_id: Path):
    try:
        paths = {
            "flair": case_id / (case_id.name + flair_name),
            "t1":    case_id / (case_id.name + t1_name),
            "t1ce":  case_id / (case_id.name + t1ce_name),
            "t2":    case_id / (case_id.name + t2_name),
            "seg":   case_id / (case_id.name + mask_name),
        }

        for k, v in paths.items():
            if not v.exists():
                raise FileNotFoundError(f"[{case_id.name}] missing modality: {k}")

        flair = read_nii(paths["flair"], sitk.sitkInt16)
        t1    = read_nii(paths["t1"],    sitk.sitkInt16)
        t1ce  = read_nii(paths["t1ce"],  sitk.sitkInt16)
        t2    = read_nii(paths["t2"],    sitk.sitkInt16)
        seg   = read_nii(paths["seg"],   sitk.sitkUInt8)

        shapes = {flair.shape, t1.shape, t1ce.shape, t2.shape, seg.shape}
        if len(shapes) != 1:
            print(f"[WARN] Shape mismatch in {case_id.name}: {shapes}. Skipped.")
            return

        flair_n = robust_norm(flair)
        t1_n    = robust_norm(t1)
        t1ce_n  = robust_norm(t1ce)
        t2_n    = robust_norm(t2)

        (flair_c, t1_c, t1ce_c, t2_c), seg_c = crop_to_224_centered_on_mask(
            [flair_n, t1_n, t1ce_n, t2_n], seg, size=CROP_SIZE
        )

        Z = seg_c.shape[0]
        saved = 0

        for z in range(Z):
            mask_np = seg_c[z]

            if SKIP_EMPTY_SLICES and mask_np.max() == 0:
                continue

            WT_Label, all_label3 = build_labels_3ch(mask_np)

            slice_id = to_slice_id(z + 1)
            name = f"{case_id.name}_{slice_id}.npz"

            np.savez_compressed(outputFlair_path   / name, data=flair_c[z].astype(np.float32))
            np.savez_compressed(outputT1_path      / name, data=t1_c[z].astype(np.float32))
            np.savez_compressed(outputT2_path      / name, data=t2_c[z].astype(np.float32))
            np.savez_compressed(outputT1ce_path    / name, data=t1ce_c[z].astype(np.float32))
            np.savez_compressed(outputMaskWT_path  / name, data=WT_Label.astype(np.uint8))
            np.savez_compressed(outputMaskAll_path / name, data=all_label3.astype(np.uint8))

            saved += 1

    except Exception as e:
        print(f"[ERROR] {case_id.name}: {e}")


parser = ArgumentParser()
parser.add_argument('--datapath',required=True,type=str)
parser.add_argument('--outputpath',required=True, type=str)
parser.add_argument('-y',action='store_true',help='“Add this option to automatically confirm and delete the destination folder.')
parser.add_argument('--num-workers', default=8,type=int)
args = parser.parse_args()

root_dir = Path(args.datapath)
out_root = Path(args.outputpath)
skip_confirm = args.y
num_workers = args.num_workers

if os.path.exists(out_root):
    if skip_confirm:
        shutil.rmtree(out_root)
        os.makedirs(out_root,exist_ok=False)  
    else: 
        while True:
            answer = input(f'I found a preexisting output directory:{out_root}.\nDo you want to delete it? y/n\n').strip().lower()
            if answer == 'y':
                shutil.rmtree(out_root)
                os.makedirs(out_root,exist_ok=False)
                break
else:
    os.makedirs(out_root,exist_ok=False)

flair_name = "-t2f.nii.gz"
t1_name    = "-t1n.nii.gz"
t1ce_name  = "-t1c.nii.gz"
t2_name    = "-t2w.nii.gz"
mask_name  = "-seg.nii.gz"


outputFlair_path   = out_root / "imgs_flair"
outputT1_path      = out_root / "imgs_t1"
outputT2_path      = out_root / "imgs_t2"
outputT1ce_path    = out_root / "imgs_t1ce"
outputMaskWT_path  = out_root / "masks"
outputMaskAll_path = out_root / "masks_all"


SKIP_EMPTY_SLICES = False

CROP_SIZE = 224

for p in [outputFlair_path, outputT1_path, outputT2_path, outputT1ce_path, 
          outputMaskWT_path, outputMaskAll_path]:
    os.makedirs(p, exist_ok=True)


case_ids = sorted(p for p in root_dir.iterdir() if p.is_dir())

with ThreadPoolExecutor(max_workers=num_workers) as ex:
    results = list(tqdm(ex.map(process_single_subject, case_ids),total=len(case_ids)))












