import os
import numpy as np
import math

def missing_list():
    n = int(math.pow(2,4))
    list = [[],[],[],[]]
    for i in range(n):
        if i % 2 == 1:
            list[0].append(i)
        if (i % 4)//2 == 1:
            list[1].append(i)
        if (i % 8)//4 == 1:
            list[2].append(i)
        if i // 8 == 1:
            list[3].append(i)
    return list

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return

def pad_zero(s, length=3):
    s = str(s)
    assert len(s) < length + 1
    if len(s) < length:
        s = '0' * (length - len(s)) + s
    return s

def zscore(x):
    x = (x - x.mean()) / x.std()
    return x

def min_max(x):
    mi, ma = x.min(), x.max()
    x = (x - mi) / (ma - mi)
    return x

def percentile(x, prct):
    low, high = np.percentile(x, prct[0]), np.percentile(x, prct[1])
    x[x < low] = low
    x[x > high] = high
    return x

def parse_image_name(name):
    n = name.split('.')[0]
    mod, pid, index, pn = n.split('_')
    return mod, pid, index, pn, 'modality'+name[len(mod):]

def center_crop(img, size):
    h, w = img.shape
    x, y = (h - size) // 2, (w - size) // 2
    img_ = img[x: x+size, y: y+size]
    return img_
