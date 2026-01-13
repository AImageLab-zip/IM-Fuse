import pickle
import os
import numpy as np
import nibabel as nib
from argparse import ArgumentParser
from pathlib import Path
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

modalities = ('t2f', 't1c', 't1n', 't2w')

def process_one_sub(args:tuple[Path,Path]):
    input_path, output_dir = args
    base_path = str(input_path / input_path.name)
    output_path = output_dir / (input_path.name + '.npz')

    label = np.astype(nib.load(base_path + '-seg.nii.gz').get_fdata(dtype=np.float32),np.uint8)
    label = np.ascontiguousarray(label)

    images = np.stack([
        nib.load(base_path + '-' + modal + '.nii.gz').get_fdata(dtype=np.float32)
        for modal in modalities
    ], axis=-1)
    images = np.ascontiguousarray(images)
    mask = images.sum(-1) > 0
    for k in range(4):
        x = images[..., k] #
        x[~mask] = 0 
        y = x[mask] #
        
        lower = np.percentile(y, 0.2) # 算分位数
        upper = np.percentile(y, 99.8)
        
        x[mask & (x < lower)] = lower
        x[mask & (x > upper)] = upper

        y = x[mask]

        mean = y.mean()
        std = y.std()
        assert std != 0, f'Zero standard deviation for: {input_path}'
        x -= mean
        x /= std

        images[..., k] = x

    np.savez_compressed(output_path,images=images,label=label)




def main():
    parser = ArgumentParser()
    parser.add_argument('--input-path',type=Path,required=True)
    parser.add_argument('--output-path',type=Path,required=True)
    parser.add_argument('--interactive',action='store_true')
    parser.add_argument('--num-workers',type=int,default=8)
    args = parser.parse_args()

    input_path:Path = args.input_path
    output_path:Path = args.output_path
    interactive:bool = args.interactive
    num_workers:int = args.num_workers

    # Deleting and creating the output directory
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
    
    input_list = [sub for sub in input_path.iterdir() if sub.is_dir()]
    processing_args = list(zip(input_list, [output_path for _ in input_list]))
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        results = list(tqdm(ex.map(process_one_sub,processing_args),total=len(processing_args)))



if __name__ == '__main__':
    main()


