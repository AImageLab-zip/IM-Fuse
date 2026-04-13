import os
import torch
from torch.utils.data import Dataset

from rand import Uniform
from transforms import Rot90, Flip, Identity, Compose
from transforms import GaussianBlur, Noise, Normalize, RandSelect
from transforms import RandCrop, CenterCrop, Pad,RandCrop3D,RandomRotion,RandomFlip,RandomIntensityChange
from transforms import NumpyType
from data_utils import pkload
from torch.utils.data import DataLoader
import numpy as np
from metrics import dice_coef
import time
from pathlib import Path
import pandas as pd
import ast

class BraTSDataset(Dataset):
    def __init__(self, list_file,root:Path, for_train=True,mode = 'train',code = None,transforms=''):
        paths, names = [], []
        if mode == 'val' and str(list_file).endswith('.csv'):
            df = pd.read_csv(list_file) 
            cases = df["case"].tolist()
            masks = df["mask"].apply(ast.literal_eval).tolist()

            paths = [root/(line + '.npz') for line in cases]
            names = cases
            self.masks_int = [np.array(mask).astype(np.int32)for mask in masks]
        else:
            with open(list_file) as f:
                for line in f:
                    line = line.strip()
                    names.append(line)
                    path = root/(line + '.npz')
                    paths.append(path)

        self.names = names
        self.paths = paths
        self.transforms = eval(transforms or 'Identity()')
        # self.for_train = for_train
        self.mode = mode
        self.code = code

    def __getitem__(self, index):
        path = self.paths[index]

        if self.mode == 'train':
            L,W,H = 128,128,128
        else:
            L,W,H = 240,240,160

        data = np.load(path)
        x = data['images']
        y = data['label']

        ### transforms work with nhwtc

        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])


        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]

        y = np.ascontiguousarray(y)
        x, y = x.squeeze(),y.squeeze()
        WT_Label = y.copy()
        WT_Label[y == 1] = 1.
        WT_Label[y == 2] = 1.
        WT_Label[y == 3] = 1.
        TC_Label = y.copy()
        TC_Label[y == 1] = 1.
        TC_Label[y == 2] = 0.
        TC_Label[y == 3] = 1.
        ET_Label = y.copy()
        ET_Label[y == 1] = 0.
        ET_Label[y == 2] = 0.
        ET_Label[y == 3] = 1.
        all_label = np.empty((3,L,W,H))
        all_label[0,...] = WT_Label
        all_label[1, ...] = TC_Label
        all_label[2, ...] = ET_Label
        all_label = all_label.astype("float32") #(3,240,240,155)


        # print(x.shape, y.shape)  # (240, 240, 155, 4) (240, 240, 155)

        x, y = torch.from_numpy(x), torch.from_numpy(all_label)
        x1,x2,x3,x4 = x[0,...],x[1,...],x[2,...],x[3,...]   ###('flair', 't1ce', 't1', 't2')
        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
        x3 = x3.unsqueeze(0)
        x4 = x4.unsqueeze(0)

        ### get multi-modality code
        if self.mode == 'train':
            mask_code = np.random.randint(2,size=4)
            while sum(mask_code)==0:
                mask_code = np.random.randint(2, size=4)
        elif self.mode == 'val':
            mask_code = self.masks_int[index]
        else:
            mask_code = np.array(self.code) # Use this during testing

        mask_code = torch.from_numpy(mask_code)
        x1, x2, x3, x4 = x1*mask_code[0],x2*mask_code[1],x3*mask_code[2],x4*mask_code[3]

        return x1,x2,x3,x4, y,mask_code

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]

if __name__ == '__main__':
    a = torch.load('best_model/val_fold1_18_0.8367.pth')
    print(a['epoch'])


