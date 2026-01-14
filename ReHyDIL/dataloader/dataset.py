import os
import torch
import random
import logging
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from PIL import Image
import torchvision.transforms.functional as F
from pathlib import Path

from torch.utils.data import Sampler

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
    def __init__(self, root_dir:Path, mode,modality, split_file,images_rate:float=1.0, transform=None):
        self.mode = mode
        self.modality = modality
        #self.mask_name = 'masks_all'
        #self.list_name = split_file
        self.transform = transform

        assert images_rate ==1, 'images_rate < 1 was never fully implemented by the original paper'
        
        sub_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                sub_list.append(line)

        self.images = [root_dir / f'imgs_{modality}/{sub}_{str(num).zfill(3)}.npz' for sub in sub_list for num in range(1,156)]
        self.masks = [root_dir / f'masks_all/{sub}_{str(num).zfill(3)}.npz' for sub in sub_list for num in range(1,156)]
        logging.info(f'Creating a {self.modality} {self.mode} dataset with {len(self.images)} examples')
        
        if images_rate !=1 and self.mode == "train":
            images_num = int(len(self.images) * images_rate) #TODO CAMBIARE SELF.SAMPLE_LIST IN IMAGES E ROBA
            self.images = self.images[:images_num]
            self.masks = self.masks[:images_num]
        logging.info(f"Creating factual {self.modality} {self.mode} dataset with {len(self.images)} examples")

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        if self.mode=='val_3d':
            raise RuntimeError('val_3d option not supported')

        img_np = np.load(self.images[idx])['data']
        mask_np = np.load(self.masks[idx])['data']

        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=0)
        if len(mask_np.shape) == 2:
            mask_np = np.expand_dims(mask_np, axis=0)
        sample = {'image': img_np.copy(), 'mask': mask_np.copy(),'idx':(self.images[idx].name)}
        return sample


class PatientBatchSampler(Sampler):
    def __init__(self, split_file, batch_size):
        sub_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                sub_list.append(line)

        self.slices_list = [f'{sub}_{num}.npz' for sub in sub_list for num in range(1,156)]
        self.batch_size = batch_size

        self.patient_to_indices = {}
        for idx, sample_name in enumerate(self.slices_list):
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
        return len(self.slices_list)

