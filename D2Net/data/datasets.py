import os
import torch
from torch.utils.data import Dataset

from .rand import Uniform
from .transforms import Rot90, Flip, Identity, Compose
from .transforms import GaussianBlur, Noise, Normalize, RandSelect
from .transforms import RandCrop, CenterCrop, Pad,RandCrop3D,RandomRotion,RandomFlip,RandomIntensityChange
from .transforms import NumpyType
from .transforms import *
from .data_utils import pkload

import numpy as np
import glob
import cv2
from pathlib import Path
from multiprocessing import shared_memory, Pool, Lock, Manager, cpu_count
from tqdm import tqdm
import lz4.frame
import pandas as pd
import ast

def _load_and_compress(args):
    idx, img, is_npz = args
    path, mask = img
    if is_npz:
        with np.load(path) as f:
            data = f["data"]
    else:
        data = np.load(path)
    comp = lz4.frame.compress(data.tobytes())
    return idx, (comp, data.shape, data.dtype.str, str(path),mask)
# Todo fix dataset logic since it's still the one from ProtoKD
class Cache:
    def __init__(self, data_file_path, splitfile, num_workers=None):
        df = pd.read_csv(splitfile)
        ids = df['case'].tolist()
        masks = df['mask'].tolist()
        masks = [ast.literal_eval(mask) for mask in masks]

        probe = next(data_file_path.iterdir())
        is_npz = probe.suffix == ".npz"
        ext = ".npz" if is_npz else ".npy"
        self.imglist = zip([data_file_path / (x + ext) for x in ids],masks)

        tasks = [(i, p, is_npz) for i, p in enumerate(self.imglist)]
        n = num_workers or max(1, cpu_count() // 2)


        results = []
        with Pool(n) as pool:
            for r in tqdm(pool.imap_unordered(_load_and_compress, tasks),
                        total=len(tasks),
                        desc="Loading and compressing"):
                results.append(r)


        self.cache = {} 
        self.list_of_names = []
        for idx, (comp, shape, dtype_str, orig_path,mask) in tqdm(results,desc='Building the Cache'):
            shm = shared_memory.SharedMemory(create=True, size=len(comp))
            if shm.buf is None:
                raise RuntimeError('Problems with shared memory')
            shm.buf[:len(comp)] = comp
            self.cache[idx] = (shm.name, shape, dtype_str, len(comp),orig_path, mask)
            self.list_of_names.append(shm.name)

    def __getitem__(self, idx): 
        return self.cache[idx]
    
    def __len__(self): 
        return len(self.cache)

    def close(self):
        for name in self.list_of_names:
            try:
                shm = shared_memory.SharedMemory(name=name)
                shm.close(); shm.unlink()
            except FileNotFoundError:
                pass



class BraTS_TrainCache(Dataset):
    def __init__(self,cache:Cache,testing:bool = False,semantic_edge=False, canny=False,):
        self.cache = cache
        if not testing:
            self.transforms = Compose([ \
                                RandCrop3D((128,128,128)), \
                                RandomRotion(10),  \
                                RandomIntensityChange((0.1,0.1)), \
                                RandomFlip(0), \
                                NumpyType((np.float32, np.int64)), \
                                ])
        else:
            self.transforms=Compose([ \
                                    Pad((0, 16, 16, 5, 0)), \
                                    NumpyType((np.float32, np.int64)), \
                                    ])
        self.semantic_edge=semantic_edge
        self.canny=canny
        self.testing = testing

    def __getitem__(self, index):
        sh_name, shape, dtype,size,orig_path,mask = self.cache[index]
        loaded_sh = shared_memory.SharedMemory(name=sh_name)
        if loaded_sh.buf is None:
            raise RuntimeError('Problems with the shared mem')
        compressed_file = bytes(loaded_sh.buf[:size])
        raw = lz4.frame.decompress(compressed_file)
        data = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape).copy()
        label = data[..., 0]
        image = data[...,1:]
        image, label = image[None, ...], label[None, ...]
        
        image,label = self.transforms([image, label])


        if self.semantic_edge and  (not self.testing):
            edge_label = self._mask2mask_semantic(label)  # [1,4,128,128,128]
        elif (not self.semantic_edge) and (not self.testing):
            edge_label = self._mask2maskb(label)          # [1,128,128,128]
        else:
            edge_label = np.array(0.01)

        image = torch.from_numpy(np.ascontiguousarray(np.squeeze(image.transpose(0, 4, 1, 2, 3),axis=0)))
        label = torch.from_numpy(np.ascontiguousarray(np.squeeze(label)))
        return_dict = {
            'image':image,
            'label':label,
            'edge_label':edge_label, 
            'mask':mask,
            'name': orig_path
        }
        return return_dict
    
    def __len__(self):
        return(len(self.cache))
    
    
    def _mask2maskb(self, mask): # mask := ori-label
        maskb = np.array(mask).astype('int32')
        b,h,w,d = maskb.shape
        maskb [maskb == 255] = -1
        maskb_ = np.array(mask).astype('float32')

        if self.canny:
            mask_tmp = np.zeros((b,h,w,d),np.float32)
            for i in range(d):
                mask_tmp[:,:,:,i] = cv2.Canny(np.uint8(maskb_[:,:,:,i]), 0, 0.001)
            # mask_tmp = mask_tmp > 0
            mask_tmp[mask_tmp > 0] = 1
            # mask_tmp = torch.from_numpy(mask_tmp).cuda().float() 
        else:
            kernel = np.ones((2,2),np.float32)/4
            mask_tmp = np.zeros((b,h,w,d),np.float32)
            for i in range(d):
                mask_tmp[:,:,:,i] = cv2.filter2D(maskb_[:,:,:,i],-1, kernel)
            mask_tmp = abs(mask_tmp - maskb_)
            mask_tmp[mask_tmp > 0.005] = 1

        return mask_tmp #mask_tmp          # [1,128,128,128]
    

    def _mask2mask_semantic(self, mask):
        _mask = np.array(mask).astype('float32')
        b,h,w,d = _mask.shape
        mask_tmp = np.zeros((b,3,h,w,d),np.float32) # 4
        mask_tmp[:,0,:,:,:] = (_mask==1)
        mask_tmp[:,1,:,:,:] = (_mask==2)
        mask_tmp[:,2,:,:,:] = (_mask==4)
        
        if self.canny:
            semantic_mask = np.zeros((b,3,h,w,d),np.float32)
            for n in range(3):
                for i in range(d):
                    semantic_mask[:,n,:,:,i] = cv2.Canny(np.uint8(mask_tmp[:,n,:,:,i]), 0, 0.001)
            semantic_mask[semantic_mask > 0] = 1
        else:
            kernel = np.ones((9,9),np.float32)/81
            semantic_mask = np.zeros((b,3,h,w,d),np.float32)
            for n in range(3):
                for i in range(d):
                    semantic_mask[:,n,:,:,i] = cv2.filter2D(mask_tmp[:,n,:,:,i],-1, kernel)
            semantic_mask = abs(semantic_mask - mask_tmp)  # smoothing edge label: (0-1)
            semantic_mask[mask_tmp > 0.005] = 1  # hard edge label: [0,1]

        return semantic_mask       # [1,3,128,128,128]
    


'''
class BraTSDataset(Dataset):
    def __init__(self, list_file, root='', for_train=False, transforms='', semantic_edge=False, canny=False, true_valid_data=False):
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, name + '_') ##
                paths.append(path)

        self.names = names
        self.paths = paths

        self.transforms = eval(transforms or 'Identity()')
        self.list_file = list_file
        self.for_train = for_train
        self.semantic_edge = semantic_edge
        self.canny = canny
        self.true_valid_data = true_valid_data

    def __getitem__(self, index):
        path = self.paths[index].split('/')[-1]
        if self.list_file.split('/')[-2] == 'train':
            save_path = ".data/BraTS_2018/pkl_file/MICCAI_BraTS2018_TrainingData/"
        if self.list_file.split('/')[-2] == 'valid':
            save_path = "./data/BraTS_2018/pkl_file/MICCAI_BraTS2018_ValidationData/"
        if self.list_file.split('/')[-2] == 'test':
            save_path = "./data/BraTS_2018/pkl_file/MICCAI_BraTS2018_TestingData/"
        x, y = pkload(save_path + path + 'data_f32.pkl')
        
        x, y = x[None, ...], y[None, ...]
        
        x,y = self.transforms([x, y])


        if self.semantic_edge and  (not self.true_valid_data):
            edge_label = self._mask2mask_semantic(y)  # [1,4,128,128,128]
        elif (not self.semantic_edge) and (not self.true_valid_data):
            edge_label = self._mask2maskb(y)          # [1,128,128,128]
        else:
            edge_label = np.array(0.01)

        if self.for_train :  # Operation only for training data      and not (self.true_valid_data):
            y[y == 4] = 3 # For the loss calculation

        x = np.ascontiguousarray(np.squeeze(x.transpose(0, 4, 1, 2, 3),axis=0))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(np.squeeze(y,axis=0))

        x, y = torch.from_numpy(x), torch.from_numpy(y)
       
        if (not self.true_valid_data):
            return x, y, torch.from_numpy(edge_label)  # For training data and train_val data
        else:
            return x, y   # Only for true_validation data

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]
    
    
    def _mask2maskb(self, mask): # mask := ori-label
        maskb = np.array(mask).astype('int32')
        b,h,w,d = maskb.shape
        maskb [maskb == 255] = -1
        maskb_ = np.array(mask).astype('float32')

        if self.canny:
            mask_tmp = np.zeros((b,h,w,d),np.float32)
            for i in range(d):
                mask_tmp[:,:,:,i] = cv2.Canny(np.uint8(maskb_[:,:,:,i]), 0, 0.001)
            # mask_tmp = mask_tmp > 0
            mask_tmp[mask_tmp > 0] = 1
            # mask_tmp = torch.from_numpy(mask_tmp).cuda().float() 
        else:
            kernel = np.ones((2,2),np.float32)/4
            mask_tmp = np.zeros((b,h,w,d),np.float32)
            for i in range(d):
                mask_tmp[:,:,:,i] = cv2.filter2D(maskb_[:,:,:,i],-1, kernel)
            mask_tmp = abs(mask_tmp - maskb_)
            mask_tmp[mask_tmp > 0.005] = 1

        return mask_tmp #mask_tmp          # [1,128,128,128]
    

    def _mask2mask_semantic(self, mask):
        _mask = np.array(mask).astype('float32')
        b,h,w,d = _mask.shape
        mask_tmp = np.zeros((b,3,h,w,d),np.float32) # 4
        mask_tmp[:,0,:,:,:] = (_mask==1)
        mask_tmp[:,1,:,:,:] = (_mask==2)
        mask_tmp[:,2,:,:,:] = (_mask==4)
        
        if self.canny:
            semantic_mask = np.zeros((b,3,h,w,d),np.float32)
            for n in range(3):
                for i in range(d):
                    semantic_mask[:,n,:,:,i] = cv2.Canny(np.uint8(mask_tmp[:,n,:,:,i]), 0, 0.001)
            semantic_mask[semantic_mask > 0] = 1
        else:
            kernel = np.ones((9,9),np.float32)/81
            semantic_mask = np.zeros((b,3,h,w,d),np.float32)
            for n in range(3):
                for i in range(d):
                    semantic_mask[:,n,:,:,i] = cv2.filter2D(mask_tmp[:,n,:,:,i],-1, kernel)
            semantic_mask = abs(semantic_mask - mask_tmp)  # smoothing edge label: (0-1)
            semantic_mask[mask_tmp > 0.005] = 1  # hard edge label: [0,1]

        return semantic_mask       # [1,3,128,128,128]
'''