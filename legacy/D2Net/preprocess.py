from argparse import ArgumentParser
import os
import shutil
import sys
import numpy as np
import nibabel as nib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def nib_load(file_name:str|Path,dtype):
    return nib.nifti1.load(file_name).get_fdata().astype(dtype)

def process_one_subject(args):
    '''
    input_path --> folder containing the 5 nifti images
    output_path --> folder that will contain all the preprocessed images
    '''
    input_path, output_path = args
    modalities = ('t2f', 't1c', 't1n', 't2w')

    if isinstance(input_path,str):
        input_path = Path(input_path)
    
    if isinstance(output_path,str):
        output_path = Path(output_path)

    #print('Starting the preprocessing')
    """ Set all Voxels that are outside of the brain mask to 0"""
    label = np.array(nib_load(input_path / (input_path.name + '-' + 'seg.nii.gz'),dtype=np.uint8), dtype='float32', order='C')
    images = np.stack([
        np.array(nib_load(input_path / (input_path.name + '-' + modal + '.nii.gz'),dtype=np.float32), dtype='float32', order='C')
        for modal in modalities], axis = -1)

    mask = images.sum(-1) > 0

    for k in range(4):
        x = images[..., k] 
        y = x[mask] 
        
        lower = np.percentile(y, 0.2)
        upper = np.percentile(y, 99.8)
        
        x[mask & (x < lower)] = lower
        x[mask & (x > upper)] = upper

        y = x[mask]

        x -= y.mean()
        x /= y.std()

        images[..., k] = x

    everything = np.concatenate([label[..., None], images], axis=-1)

    output_filename = output_path / (input_path.name +'.npz')
    np.savez_compressed(output_filename, data=everything)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-path',type=str,help='Folder containing the BraTS2023 training data',required=True)
    parser.add_argument('--output-path',type=str, help='Folder that will contain the preprocessed dataset',required=True)
    parser.add_argument('--interactive',action='store_true')
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    interactive = args.interactive
    if interactive:
        if os.path.exists(output_path):
            while True:
                answer = input(f'{output_path} already exists. Do you want to delete it? (y/n)').strip().lower()
                if answer == 'y':
                    shutil.rmtree(output_path)
                    os.makedirs(output_path)
                    break
                elif answer == 'n':
                    sys.exit(0)
                else:
                    print('Please provide a valid answer.')
        else:
            os.makedirs(output_path)
    else:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
            os.makedirs(output_path)
        else:
            os.makedirs(output_path)

    list_of_subjects = list(input_path.iterdir())
    args_list = [(sub, output_path) for sub in list_of_subjects]
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_one_subject,args_list)))
    

