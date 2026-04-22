import numpy as np
from pathlib import Path
from tqdm import tqdm

input_path = Path('/work/grana_neuro/missing_modalities/BRATS2023_Training_D2Net')

for sub in tqdm(list(input_path.iterdir())):
    np_sub = np.load(sub)['data']
    if np_sub.shape != (240, 240, 155, 5):
        print(f'{sub} has shape: {np_sub.shape}')