import nibabel as nib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
import numpy as np
def check_one_subj(sub_path):
    try:
        t1c = nib.loadsave.load(sub_path / (sub_path.name + '_t1ce.nii')).get_fdata(dtype=np.float32)
        t1n = nib.loadsave.load(sub_path / (sub_path.name + '_t1.nii')).get_fdata(dtype=np.float32)
        t2f = nib.loadsave.load(sub_path / (sub_path.name + '_flair.nii')).get_fdata(dtype=np.float32)
        t2w = nib.loadsave.load(sub_path / (sub_path.name + '_t2.nii')).get_fdata(dtype=np.float32)
        modals = [t1c,t1n,t2f,t2w]
        for modal in modals:
            #if np.sum(modal < 0) >0:
                #print(sub_path)
            pass
    except:
        print(f'problems with: {sub_path}')
    
def main():
    input_path = Path('/work/grana_neuro/MICCAI_BraTS_2018_Data_Training')
    all_subs = list((input_path / 'HGG').iterdir()) + list((input_path / 'LGG').iterdir())
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(tqdm(ex.map(check_one_subj,all_subs)))
if __name__ == '__main__':
    main()