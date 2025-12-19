import random
import lz4.frame
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from multiprocessing import shared_memory
from tqdm import tqdm
from multiprocessing import shared_memory, Pool, Lock, Manager, cpu_count

_LOCK = None

def random_crop(img, crop_size=(80, 80, 80)):
    # img [c,d,h,w]
    _, d, h, w = img.shape
    pd = np.random.randint(d - crop_size[0])
    ph = np.random.randint(h - crop_size[1])
    pw = np.random.randint(w - crop_size[2])
    patch = img[
        :, pd : pd + crop_size[0], ph : ph + crop_size[1], pw : pw + crop_size[2]
    ]
    return patch


def random_flip(data, p=0.5):
    # [C,D,H,W]
    if random.random() < p:
        data = np.flip(data, axis=1)
    if random.random() < p:
        data = np.flip(data, axis=2)
    if random.random() < p:
        data = np.flip(data, axis=3)
    return data


def random_scale_one_channel(data, p=0.5):
    if random.random() < p:
        scale = random.uniform(0.9, 1.1)
        data = data * scale
    return data


def random_scale(data, c=4, p=0.5):
    for i in range(c):
        data[i] = random_scale_one_channel(data[i], p)
    return data

def _cache_one_file(args):
    index, path, is_npz = args
    if is_npz:
        with np.load(path) as f:
            data = f["data"]
    else:
        data = np.load(path)
    compressed = lz4.frame.compress(data.tobytes())
    shm = shared_memory.SharedMemory(create=True, size=len(compressed))
    if shm.buf is None:
        raise RuntimeError('Problems with shared memory')
    shm.buf[:len(compressed)] = compressed
    meta = (shm.name, data.shape, data.dtype.str, len(compressed))
    return index, meta

def _init_worker_lock():
    global _LOCK
    _LOCK = Lock()

def _load_and_compress(args):
    idx, path, is_npz = args
    if is_npz:
        with np.load(path) as f:
            data = f["data"]
    else:
        data = np.load(path)
    comp = lz4.frame.compress(data.tobytes())
    return idx, (comp, data.shape, data.dtype.str, str(path))

class Cache:
    def __init__(self, data_file_path, splitfile, num_workers=None):
        data_file_path = Path(data_file_path)
        with open(splitfile) as f:
            ids = [l.strip() for l in f if l.strip()]
        probe = next(data_file_path.iterdir())
        is_npz = probe.suffix == ".npz"
        ext = ".npz" if is_npz else ".npy"
        self.imglist = [data_file_path / (x + ext) for x in ids]

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
        for idx, (comp, shape, dtype_str, orig_path) in tqdm(results):
            shm = shared_memory.SharedMemory(create=True, size=len(comp))
            if shm.buf is None:
                raise RuntimeError('Problems with shared memory')
            shm.buf[:len(comp)] = comp
            self.cache[idx] = (shm.name, shape, dtype_str, len(comp))
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
    def __init__(self,cache:Cache, crop_size=(80, 80, 80), flip=True, scale=True):
        self.cache = cache
        self.crop_size=crop_size
        self.flip=flip
        self.scale=scale

    def __getitem__(self, index):
        sh_name, shape, dtype,size = self.cache[index]
        loaded_sh = shared_memory.SharedMemory(name=sh_name)
        if loaded_sh.buf is None:
            raise RuntimeError('Problems with the shared mem')
        compressed_file = bytes(loaded_sh.buf[:size])
        raw = lz4.frame.decompress(compressed_file)
        data = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape).copy()
        data = random_crop(data, self.crop_size)
        if self.flip:
            data = random_flip(data)
        label = data[4:]
        image = data[0:4]
        if self.scale:
            image = random_scale(image)
        image = torch.from_numpy(np.ascontiguousarray(image))
        label = torch.from_numpy(np.ascontiguousarray(label))

        return_dict = {
            'vol':image,
            'seg':label
        }
        return return_dict
    def __len__(self):
        return(len(self.cache))

class BraTS_Train(Dataset):
    def __init__(
        self, data_file_path, splitfile, crop_size=(80, 80, 80), flip=True, scale=True
    ):
        # just for train
        # base_dir  dataset path of brats dataset e.g. '../data'
        # student_modality: must be list e.g.[0] [1,2] or None(pre-train of teacher)
        data_file_path = Path(data_file_path)
        imglist = []
        with open(splitfile, "r") as f:
            lines = f.readlines()
            for ll in lines:
                imglist.append(ll.replace("\n", ""))
        one_sub = str(list(data_file_path.iterdir())[0])
        if one_sub.endswith('.npy'):
            self.compressed = False
            self.imglist = [data_file_path / (x + ".npy") for x in imglist]
        elif one_sub.endswith('.npz'):
            self.compressed = True
            self.imglist = [data_file_path / (x + ".npz") for x in imglist]
        else:
            raise RuntimeError(f'Found an invalid file in the input directory {one_sub}')


        self.crop_size = crop_size
        self.flip = flip
        self.scale = scale


    def __getitem__(self, index):

        if self.compressed:
            loaded = np.load(self.imglist[index])
            data = loaded['data']
        else:
            data = np.load(self.imglist[index])

        data = random_crop(data, self.crop_size)
        if self.flip:
            data = random_flip(data)
        label = data[4:]
        image = data[0:4]
        if self.scale:
            image = random_scale(image)
        image = torch.from_numpy(np.ascontiguousarray(image))
        label = torch.from_numpy(np.ascontiguousarray(label))

        return_dict = {
            'vol':image,
            'seg':label,
            'name':str(self.imglist[index])
        }
        return return_dict

    def __len__(self):
        return len(self.imglist)


'''
    def __getitem__(self, index):

        if self.compressed:
            loaded = np.load(self.imglist[index])
            data = loaded['data']
        else:
            data = np.load(self.imglist[index])

        data = random_crop(data, self.crop_size)
        if self.flip:
            data = random_flip(data)
        label = data[4:]
        image = data[0:4]
        if self.scale:
            image = random_scale(image)
        image = torch.from_numpy(np.ascontiguousarray(image))
        label = torch.from_numpy(np.ascontiguousarray(label))
'''

'''

import random
import lz4.frame
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


def random_crop(img, crop_size=(80, 80, 80)):
    # img [c,d,h,w]
    _, d, h, w = img.shape
    pd = np.random.randint(d - crop_size[0])
    ph = np.random.randint(h - crop_size[1])
    pw = np.random.randint(w - crop_size[2])
    patch = img[
        :, pd : pd + crop_size[0], ph : ph + crop_size[1], pw : pw + crop_size[2]
    ]
    return patch


def random_flip(data, p=0.5):
    # [C,D,H,W]
    if random.random() < p:
        data = np.flip(data, axis=1)
    if random.random() < p:
        data = np.flip(data, axis=2)
    if random.random() < p:
        data = np.flip(data, axis=3)
    return data


def random_scale_one_channel(data, p=0.5):
    if random.random() < p:
        scale = random.uniform(0.9, 1.1)
        data = data * scale
    return data


def random_scale(data, c=4, p=0.5):
    for i in range(c):
        data[i] = random_scale_one_channel(data[i], p)
    return data


class BraTS_Train(Dataset):
    def __init__(
        self, data_file_path, splitfile, crop_size=(80, 80, 80), flip=True, scale=True
    ):
        # just for train
        # base_dir  dataset path of brats dataset e.g. '../data'
        # student_modality: must be list e.g.[0] [1,2] or None(pre-train of teacher)
        data_file_path = Path(data_file_path)
        imglist = []
        with open(splitfile, "r") as f:
            lines = f.readlines()
            for ll in lines:
                imglist.append(ll.replace("\n", ""))
        one_sub = str(list(data_file_path.iterdir())[0])
        if one_sub.endswith('.npy'):
            self.compressed = False
            self.imglist = [data_file_path / (x + ".npy") for x in imglist]
        elif one_sub.endswith('.npz'):
            self.compressed = True
            self.imglist = [data_file_path / (x + ".npz") for x in imglist]
        else:
            raise RuntimeError(f'Found an invalid file in the input directory {one_sub}')


        self.crop_size = crop_size
        self.flip = flip
        self.scale = scale
        self.cache = {}

    def __getitem__(self, index):
        if index not in self.cache:
            if self.compressed:
                loaded = np.load(self.imglist[index])
                data = loaded['data']
            else:
                data = np.load(self.imglist[index])
            compressed_file = lz4.frame.compress(data.tobytes())
            self.cache[index] = (compressed_file, data.shape, data.dtype)
        else:
            compressed_file, shape, dtype = self.cache[index]
            data = np.frombuffer(lz4.frame.decompress(compressed_file), dtype=dtype).reshape(shape).copy()

        data = random_crop(data, self.crop_size)
        if self.flip:
            data = random_flip(data)
        label = data[4:]
        image = data[0:4]
        if self.scale:
            image = random_scale(image)
        image = torch.from_numpy(np.ascontiguousarray(image))
        label = torch.from_numpy(np.ascontiguousarray(label))

        return_dict = {
            'vol':image,
            'seg':label,
            'name':str(self.imglist[index])
        }
        return return_dict

    def __len__(self):
        return len(self.imglist)

'''
