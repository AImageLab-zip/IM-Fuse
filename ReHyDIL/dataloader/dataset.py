import os
import torch
import random
import logging
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

import random
from PIL import Image
import torchvision.transforms.functional as F


def random_augmentation(slice_path):

    img = Image.open(slice_path).convert("L")

    angle = random.uniform(-15, 15)
    img = F.rotate(img, angle)

    if random.random() < 0.5:
        img = F.hflip(img)


    if random.random() < 0.5:
        img = F.vflip(img)

    return img


class BaseDataSets(Dataset):
    def __init__(self, data_dir=None, mode='train',img_mode='t2',mask_name='masks', list_name='slice_nidus_all.list',images_rate=1, transform=None):
        self._data_dir = data_dir
        self.sample_list = []
        self.mode = mode
        self.img_mode = img_mode
        self.mask_name = mask_name
        self.list_name = list_name
        self.transform = transform

        list_path = os.path.join(self._data_dir,self.list_name)

        with open(list_path, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.sample_list.append(line)
        logging.info(f'Creating total {self.img_mode} {self.mode} dataset with {len(self.sample_list)} examples')

        if images_rate !=1 and self.mode == "train":
            images_num = int(len(self.sample_list) * images_rate)
            self.sample_list = self.sample_list[:images_num]
        logging.info(f"Creating factual {self.img_mode} {self.mode} dataset with {len(self.sample_list)} examples")

    def __len__(self):
        return len(self.sample_list)
    def __sampleList__(self):
        return self.sample_list
        
    def __getitem__(self, idx):
        if self.mode=='val_3d':
            case = idx
        else:
            case = self.sample_list[idx]

        img_np_path = os.path.join(self._data_dir,'imgs_{}/{}.npy'.format(self.img_mode,case))
        mask_np_path = os.path.join(self._data_dir,'{}/{}.npy'.format(self.mask_name,case))

        img_np = np.load(img_np_path)
        mask_np = np.load(mask_np_path)

        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=0)
        if len(mask_np.shape) == 2:
            mask_np = np.expand_dims(mask_np, axis=0)
        sample = {'image': img_np.copy(), 'mask': mask_np.copy(),'idx':case}
        return sample

import random
from torch.utils.data import Sampler



class PatientBatchSampler(Sampler):
    def __init__(self, slices_list, batch_size):
        self.slices_list = slices_list
        self.batch_size = batch_size

        self.patient_to_indices = {}
        for idx, sample_name in enumerate(slices_list):
            arr = sample_name.rsplit('_', 1)
            patient_id = arr[0] + "_"
            if patient_id not in self.patient_to_indices:
                self.patient_to_indices[patient_id] = []
            self.patient_to_indices[patient_id].append(idx)

        self.patientID_list = list(self.patient_to_indices.keys())

    def __iter__(self):

        patient_indices = {
            pid: indices.copy()
            for pid, indices in self.patient_to_indices.items()
        }

        for pid in patient_indices:
            random.shuffle(patient_indices[pid])

        available_patients = [pid for pid in self.patientID_list if patient_indices[pid]]


        batches = []
        while len(available_patients) > 0:

            curr_bs = min(len(available_patients), self.batch_size)
            selected_pids = random.sample(available_patients, curr_bs)
            batch = []
            for pid in selected_pids:
                index_popped = patient_indices[pid].pop(0)
                batch.append(index_popped)
                if not patient_indices[pid]:
                    available_patients.remove(pid)
            batches.append(batch)

        for batch in batches:
            yield batch

    def __len__(self):
        return 0




class PrevBaseDataSets(Dataset):
    def __init__(self, data_dir=None, mode='train', prev_img_mode='ct', mask_name='masks', list_name='slice_nidus_all.list',
                 images_rate=1, transform=None):
        self._data_dir = data_dir
        self.sample_list = []
        self.mode = mode
        self.prev_img_mode = prev_img_mode
        self.mask_name = mask_name
        self.list_name = list_name
        self.transform = transform

        list_path = os.path.join(self._data_dir, self.list_name)

        with open(list_path, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.sample_list.append(line)

        logging.info(f'Creating total {self.prev_img_mode} {self.mode} dataset with {len(self.sample_list)} examples')

        if images_rate != 1 and self.mode == "train":
            images_num = int(len(self.sample_list) * images_rate)
            self.sample_list = self.sample_list[:images_num]
        logging.info(f"Creating factual {self.prev_img_mode} {self.mode} dataset with {len(self.sample_list)} examples")

    def __len__(self):
        return len(self.sample_list)

    def __sampleList__(self):
        return self.sample_list

    def __getitem__(self, idx):
        if self.mode == 'val_3d':
            case = idx
        else:
            case = self.sample_list[idx]

        img_np_path = os.path.join(self._data_dir, 'imgs_{}/{}.npy'.format(self.prev_img_mode, case))
        mask_np_path = os.path.join(self._data_dir, '{}/{}.npy'.format(self.mask_name, case))

        img_np = np.load(img_np_path)
        mask_np = np.load(mask_np_path)

        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=0)
        if len(mask_np.shape) == 2:
            mask_np = np.expand_dims(mask_np, axis=0)
        sample = {'image': img_np.copy(), 'mask': mask_np.copy(), 'idx': case}
        return sample
        