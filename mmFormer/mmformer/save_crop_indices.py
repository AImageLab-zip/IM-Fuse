import os
import numpy as np
import medpy.io as medio
join=os.path.join
import pandas as pd

#'../../../../DB/BraTS18/ori_data/Training'
src_path = '/work/grana_neuro/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
name_list = os.listdir(src_path)


def sup_128(xmin, xmax):
    if xmax - xmin < 128:
        print ('#' * 100)
        ecart = int((128-(xmax-xmin))/2)
        xmax = xmax+ecart+1
        xmin = xmin-ecart
    if xmin < 0:
        xmax-=xmin
        xmin=0
    return xmin, xmax

def crop(vol):
    if len(vol.shape) == 4:
        vol = np.amax(vol, axis=0)
    assert len(vol.shape) == 3

    x_dim, y_dim, z_dim = tuple(vol.shape)
    x_nonzeros, y_nonzeros, z_nonzeros = np.where(vol != 0)

    x_min, x_max = np.amin(x_nonzeros), np.amax(x_nonzeros)
    y_min, y_max = np.amin(y_nonzeros), np.amax(y_nonzeros)
    z_min, z_max = np.amin(z_nonzeros), np.amax(z_nonzeros)

    x_min, x_max = sup_128(x_min, x_max)
    y_min, y_max = sup_128(y_min, y_max)
    z_min, z_max = sup_128(z_min, z_max)

    return x_min, x_max, y_min, y_max, z_min, z_max

def normalize(vol):
    mask = vol.sum(0) > 0
    for k in range(4):
        x = vol[k, ...]
        y = x[mask]
        x = (x - y.mean()) / y.std()
        vol[k, ...] = x

    return vol

results = []

for file_name in name_list:
    case_id = file_name.split('/')[-1]

    # Load volumes
    flair, _ = medio.load(os.path.join(src_path, file_name, case_id + '-t2f.nii.gz'))
    t1ce, _ = medio.load(os.path.join(src_path, file_name, case_id + '-t1c.nii.gz'))
    t1, _ = medio.load(os.path.join(src_path, file_name, case_id + '-t1n.nii.gz'))
    t2, _ = medio.load(os.path.join(src_path, file_name, case_id + '-t2w.nii.gz'))

    # Compute crop indices
    vol = np.stack((flair, t1ce, t1, t2), axis=0).astype(np.float32)
    x_min, x_max, y_min, y_max, z_min, z_max = crop(vol)

    # Append results to the list
    results.append([case_id, x_min, x_max, y_min, y_max, z_min, z_max])

# Save to CSV using pandas
df = pd.DataFrame(results, columns=["case_id", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"])
df.to_csv("crop_indices.csv", index=False)
