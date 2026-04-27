#!/usr/bin/env python3
# encoding: utf-8
# Code modified from https://github.com/Wangyixinxin/ACN
import glob
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
import ast

mask_array = np.array([[True, False, False, False], [False, True, False, False], [False, False, True, False], [False, False, False, True],
                      [True, True, False, False], [True, False, True, False], [True, False, False, True], [False, True, True, False], [False, True, False, True], [False, False, True, True], [True, True, True, False], [True, True, False, True], [True, False, True, True], [False, True, True, True],
                      [True, True, True, True]])


class Brats2018(Dataset):
    def __init__(self, patients_dir, crop_size, modes, train=True, normalization = True, dataset='brats'):
        self.patients_dir = patients_dir
        self.modes = modes
        self.train = train
        self.crop_size = crop_size
        self.normalization = normalization
        self.dataset = dataset

    def __len__(self):
        return len(self.patients_dir)

    def __getitem__(self, index):
        patient_dir = self.patients_dir[index]
        volumes = []
        modes = list(self.modes) + ['seg']
        p = "-" if self.dataset == 'brats' else '_'
        for mode in modes:
            patient_id = os.path.split(patient_dir)[-1]
            volume_path = os.path.join(patient_dir, patient_id + p + mode + '.nii.gz')
            volume = nib.load(volume_path).get_fdata()
            if not mode == "seg" and self.normalization:
                volume = self.normlize(volume)  # [0, 1.0]
            volumes.append(volume)                  # [h, w, d]
        seg_volume = volumes[-1]
        volumes = volumes[:-1]
        volume, seg_volume = self.aug_sample(volumes, seg_volume)

        if self.dataset == 'fets':
            wt_volume = ((seg_volume ==1)|(seg_volume==2)|(seg_volume==4)).astype('uint8')# peritumoral edema ED
            tc_volume = ((seg_volume == 1)|(seg_volume==4)).astype('uint8') # enhancing tumor core NET
            et_volume = ((seg_volume == 4)).astype('uint8') # enhancing tumor ET
            bg_volume = ((seg_volume == 0)).astype('uint8')
        elif self.dataset == 'brats':
            wt_volume = ((seg_volume == 1)|(seg_volume==2)|(seg_volume==3)).astype('uint8')  # peritumoral edema ED
            tc_volume = ((seg_volume == 1)|(seg_volume==3)).astype('uint8')                 # enhancing tumor core NET
            et_volume = ((seg_volume == 3)).astype('uint8')                                 # enhancing tumor ET
            bg_volume = ((seg_volume == 0)).astype('uint8')
        
        seg_volume = [wt_volume, tc_volume, et_volume, bg_volume]
        seg_volume = np.concatenate(seg_volume, axis=0).astype("float32")

        return (torch.tensor(volume.copy(), dtype=torch.float),
                torch.tensor(seg_volume.copy(), dtype=torch.float))

class Brats2018_divide(Dataset):
    def __init__(self, patients_dir, crop_size, modes, train=True, normalization = True, data_file_path=''):
        self.patients_dir = patients_dir
        self.modes = modes
        self.train = train
        self.crop_size = crop_size
        self.normalization = normalization
        self.data_file_path = data_file_path
        df = pd.read_csv(data_file_path)
        self.data_list = df['case']
        
        if not self.train:
            self.masks = df['mask'].apply(ast.literal_eval)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        patient_id = self.data_list[index]
        volumes = []
        modes = list(self.modes) + ['seg']

        for mode in modes:
            id = str(os.path.split(patient_id)[-1])
            volume_path = os.path.join(self.patients_dir, patient_id, id + "_" + mode + '.nii')
            volume = nib.load(volume_path).get_fdata()
            if not mode == "seg" and self.normalization:
                volume = self.normlize(volume)      # [0, 1.0]
            volumes.append(volume)                  # [h, w, d]
        seg_volume = volumes[-1]
        volumes = volumes[:-1]
        volume, seg_volume = self.aug_sample(volumes, seg_volume)

        # 1-NCR/NET, 2-ED, 4-ET, 0-BGD
        ncr_volume = ((seg_volume == 1)).astype('uint8')    # peritumoral edema NCR
        ed_volume = ((seg_volume == 2)).astype('uint8')   # enhancing tumor core ED
        et_volume = ((seg_volume == 4)).astype('uint8')     # enhancing tumor ET
        bg_volume = ((seg_volume == 0)).astype('uint8')
        
        seg_volume = [ncr_volume, ed_volume, et_volume, bg_volume]
        seg_volume = np.concatenate(seg_volume, axis=0).astype("float32")

        if self.train:
            mask_idx = np.random.choice(15, 1)
            mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0) #(4)
        else:
            mask = np.array(self.masks[index])
            mask = torch.squeeze(torch.from_numpy(mask), dim=0)

        return (torch.tensor(volume.copy(), dtype=torch.float),
                torch.tensor(seg_volume.copy(), dtype=torch.float), 
                mask)

    def aug_sample(self, volumes, mask):
        """
            Args:
                volumes: list of array, [h, w, d]
                mask: array [h, w, d], segmentation volume
            Ret: x, y: [channel, h, w, d]

        """
        x = np.stack(volumes, axis=0)       # [N, H, W, D]
        y = np.expand_dims(mask, axis=0)    # [channel, h, w, d]

        if self.train:
            # crop volume
            x, y = self.random_crop(x, y)
            if random.random() < 0.5:
                x = np.flip(x, axis=1)
                y = np.flip(y, axis=1)
            if random.random() < 0.5:
                x = np.flip(x, axis=2)
                y = np.flip(y, axis=2)
            if random.random() < 0.5:
                x = np.flip(x, axis=3)
                y = np.flip(y, axis=3)
        else:
            x, y = self.center_crop(x, y)

        return x, y

    def random_crop(self, x, y):
        """
        Args:
            x: 4d array, [channel, h, w, d]
        """
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = random.randint(0, height - crop_size[0] - 1)
        sy = random.randint(0, width - crop_size[1] - 1)
        sz = random.randint(0, depth - crop_size[2] - 1)
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def center_crop(self, x, y):
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = (height - crop_size[0] - 1) // 2
        sy = (width - crop_size[1] - 1) // 2
        sz = (depth - crop_size[2] - 1) // 2
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def normlize(self, x):
        return (x - x.min()) / (x.max() - x.min())
    
    def normlize_brain(self, x, epsilon=1e-8):
        average        = x[np.nonzero(x)].mean()
        std            = x[np.nonzero(x)].std() + epsilon
        mask           = x>0
        sub_mean       = np.where(mask, x-average, x)
        x_normalized   = np.where(mask, sub_mean/std, x)
        return x_normalized

class Brats2023_divide(Dataset):
    def __init__(self, patients_dir, crop_size, modes, train=True, normalization = True, data_file_path=''):
        self.patients_dir = patients_dir
        self.modes = modes
        self.train = train
        self.crop_size = crop_size
        self.normalization = normalization
        if self.train:
            with open(data_file_path, 'r') as f:
                self.data_list = [i.strip() for i in f.readlines()]
            self.data_list.sort()
        else:
            df = pd.read_csv(data_file_path)
            self.data_list = df['case']
            self.masks = df['mask'].apply(ast.literal_eval)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        patient_id = self.data_list[index]
        volumes = []
        modes = list(self.modes) + ['seg']
        for mode in modes:
            #patient_id = os.path.split(patient_dir)[-1]
            volume_path = os.path.join(self.patients_dir, patient_id, patient_id + "-" + mode + '.nii.gz')
            volume = nib.load(volume_path).get_fdata()
            if not mode == "seg" and self.normalization:
                volume = self.normlize(volume)  # [0, 1.0]
            volumes.append(volume)                  # [h, w, d]
        seg_volume = volumes[-1]
        volumes = volumes[:-1]
        volume, seg_volume = self.aug_sample(volumes, seg_volume)
        # volume = (4, 160, 192, 128)
        # seg_volume = (1, 160, 192, 128)
        
        # GD-enhancing tumor (ET — label 3), the peritumoral edematous/invaded tissue (ED — label 2), and the necrotic tumor core (NCR — label 1)
        ncr_volume = ((seg_volume == 1)).astype('uint8')    # peritumoral edema NCR
        ed_volume = ((seg_volume == 2)).astype('uint8')   # enhancing tumor core ED
        et_volume = ((seg_volume == 3)).astype('uint8')     # enhancing tumor ET
        bg_volume = ((seg_volume == 0)).astype('uint8')
        
        seg_volume = [ncr_volume, ed_volume, et_volume, bg_volume]
        seg_volume = np.concatenate(seg_volume, axis=0).astype("float32")

        if self.train:
            mask_idx = np.random.choice(15, 1)
            mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0) #(4)
        else:
            mask = np.array(self.masks[index])
            mask = torch.squeeze(torch.from_numpy(mask), dim=0)

        return (torch.tensor(volume.copy(), dtype=torch.float),
                torch.tensor(seg_volume.copy(), dtype=torch.float), 
                mask)

    def aug_sample(self, volumes, mask):
        """
            Args:
                volumes: list of array, [h, w, d]
                mask: array [h, w, d], segmentation volume
            Ret: x, y: [channel, h, w, d]

        """
        x = np.stack(volumes, axis=0)       # [N, H, W, D]
        y = np.expand_dims(mask, axis=0)    # [channel, h, w, d]

        if self.train:
            # crop volume
            x, y = self.random_crop(x, y)
            if random.random() < 0.5:
                x = np.flip(x, axis=1)
                y = np.flip(y, axis=1)
            if random.random() < 0.5:
                x = np.flip(x, axis=2)
                y = np.flip(y, axis=2)
            if random.random() < 0.5:
                x = np.flip(x, axis=3)
                y = np.flip(y, axis=3)
        else:
            x, y = self.center_crop(x, y)

        return x, y

    def random_crop(self, x, y):
        """
        Args:
            x: 4d array, [channel, h, w, d]
        """
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = random.randint(0, height - crop_size[0] - 1)
        sy = random.randint(0, width - crop_size[1] - 1)
        sz = random.randint(0, depth - crop_size[2] - 1)
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def center_crop(self, x, y):
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = (height - crop_size[0] - 1) // 2
        sy = (width - crop_size[1] - 1) // 2
        sz = (depth - crop_size[2] - 1) // 2
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def normlize(self, x):
        return (x - x.min()) / (x.max() - x.min())
    
    def normlize_brain(self, x, epsilon=1e-8):
        average        = x[np.nonzero(x)].mean()
        std            = x[np.nonzero(x)].std() + epsilon
        mask           = x>0
        sub_mean       = np.where(mask, x-average, x)
        x_normalized   = np.where(mask, sub_mean/std, x)
        return x_normalized
    
def split_dataset(data_root, test_p):
    patients_dir = glob.glob(os.path.join(data_root, "*GG", "Brats18*"))
    patients_dir.sort()
    N = int(len(patients_dir)*test_p)
    train_patients_list =  patients_dir[N:]
    val_patients_list   =  patients_dir[:N]

    return train_patients_list, val_patients_list
    
def make_data_loaders(config):
    if config['dataset'] == 'fets':
        train_list = glob.glob(os.path.join(config['path_to_data'], 'train', 'FeTS*'))
        val_list = glob.glob(os.path.join(config['path_to_data'], 'test', 'FeTS*'))

    elif config['dataset'] == 'brats':
        train_list = glob.glob(os.path.join(config['path_to_data'], 'train', 'BraTS*'))
        val_list = glob.glob(os.path.join(config['path_to_data'], 'test', 'BraTS*'))

    crop_size = np.zeros((3))
    crop_size[0] = config['inputshape'][0]
    crop_size[1] = config['inputshape'][1]      
    crop_size[2] = config['inputshape'][2]
    crop_size    = crop_size.astype(np.uint16)
    crop_size    = (160, 192, 128)

    train_ds = Brats2018(train_list, crop_size=crop_size, modes=config['modalities'], train=True)
    val_ds = Brats2018(val_list, crop_size=crop_size, modes=config['modalities'], train=False)

    loaders = {}
    loaders['train'] = DataLoader(train_ds, batch_size=int(config['batch_size_tr']),
                                  num_workers=4,
                                  pin_memory=True,
                                  shuffle=True)
    loaders['eval'] = DataLoader(val_ds, batch_size=int(config['batch_size_va']),
                                  num_workers=4,
                                  pin_memory=True,
                                  shuffle=False)
    return loaders

def make_data_loaders_divide(config, train_list_path, val_list_path, test_list_path):
    train_file_path = os.path.join(config['root'], train_list_path)
    val_file_path = os.path.join(config['root'], val_list_path)
    test_file_path = os.path.join(config['root'], test_list_path)

    crop_size = np.zeros((3))
    crop_size[0] = config['inputshape'][0]
    crop_size[1] = config['inputshape'][1]      
    crop_size[2] = config['inputshape'][2]
    crop_size    = crop_size.astype(np.uint16)
    crop_size    = (160, 192, 128)

    train_ds = Brats2018_divide(config['patient_dir'], crop_size=crop_size, modes=config['modalities'], train=True, data_file_path=train_file_path)
    val_ds = Brats2018_divide(config['patient_dir'], crop_size=crop_size, modes=config['modalities'], train=False, data_file_path=val_file_path)
    test_ds = Brats2018_divide(config['patient_dir'], crop_size=crop_size, modes=config['modalities'], train=False, data_file_path=test_file_path)
    loaders = {}
    loaders['train'] = DataLoader(train_ds, batch_size=int(config['batch_size_tr']),
                                  num_workers=0,
                                  pin_memory=False,
                                  shuffle=True)
    loaders['val'] = DataLoader(val_ds, batch_size=int(config['batch_size_va']),
                                  num_workers=0,
                                  pin_memory=False,
                                  shuffle=False)
    loaders['test'] = DataLoader(test_ds, batch_size=int(config['batch_size_va']),
                                  num_workers=0,
                                  pin_memory=False,
                                  shuffle=False)
    return loaders

def make_data_loaders_divide_2023(config, train_list_path, val_list_path, test_list_path):
    train_file_path = os.path.join(config['root'], train_list_path)
    val_file_path = os.path.join(config['root'], val_list_path)
    test_file_path = os.path.join(config['root'], test_list_path)

    crop_size = np.zeros((3))
    crop_size[0] = config['inputshape'][0]
    crop_size[1] = config['inputshape'][1]      
    crop_size[2] = config['inputshape'][2]
    crop_size    = crop_size.astype(np.uint16)
    crop_size    = (160, 192, 128)

    train_ds = Brats2023_divide(config['patient_dir'], crop_size=crop_size, modes=config['modalities'], train=True, data_file_path=train_file_path)
    val_ds = Brats2023_divide(config['patient_dir'], crop_size=crop_size, modes=config['modalities'], train=False, data_file_path=val_file_path)
    test_ds = Brats2023_divide(config['patient_dir'], crop_size=crop_size, modes=config['modalities'], train=False, data_file_path=test_file_path)

    loaders = {}
    loaders['train'] = DataLoader(train_ds, batch_size=int(config['batch_size_tr']),
                                  num_workers=0,
                                  pin_memory=False,
                                  shuffle=True)
    loaders['val'] = DataLoader(val_ds, batch_size=int(config['batch_size_va']),
                                  num_workers=0,
                                  pin_memory=False,
                                  shuffle=False)
    loaders['test'] = DataLoader(test_ds, batch_size=int(config['batch_size_va']),
                                  num_workers=0,
                                  pin_memory=False,
                                  shuffle=False)
    return loaders

