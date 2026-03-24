# Standard library
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import shutil

# External dependencies
import medpy.io as medio
import numpy as np
from tqdm import tqdm

# Internal modules
from brainchmark.preprocessing.config import CropConfig, ClampConfig, NormConfig
from brainchmark.datasets.config import DatasetType

def preprocess_case(
    file: dict[str, Path | str],
    output_dir: Path,
    crop_config: CropConfig,
    clamp_config: ClampConfig,
    norm_config: NormConfig,
    dataset_type: DatasetType
) -> str:
    output_file = output_dir / f"{file['name']}.npz"

    modals = ["t1c", "t1n", "t2f", "t2w"]

    image_files = []

    for modal in modals:
        image, _ = medio.load(file[modal])  # type: ignore[index]
        image_files.append(image)
    images = np.stack(image_files, axis=0)

    seg, _ = medio.load(file["seg"]) # type: ignore[index]
    seg = np.expand_dims(seg,axis=0)
    if dataset_type is DatasetType.BRATS18:
        seg[seg==4] = 3

    images, seg = crop_config.fn(images,seg,crop_config)
    images = clamp_config.fn(images,clamp_config)
    images = norm_config.fn(images,norm_config)
    np.savez_compressed(output_file, images=images, seg=seg)
    return str(output_file)

def run_preprocessing(
    input_dir: Path,
    output_dir: Path,
    crop_config: CropConfig,
    clamp_config: ClampConfig,
    norm_config: NormConfig,
    dataset_type: DatasetType,
    yes: bool
) -> None:
    """Run the preprocessing pipeline with the selected crop configuration."""
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"'{output_dir}' is not a valid directory.")

        if not yes:
            answer = input(
                f"Do you wish to permanently delete the folder '{output_dir}'? [y/N]: "
            ).strip().lower()
            if answer not in ("y", "yes"):
                raise RuntimeError("Operation cancelled by user.")

        shutil.rmtree(output_dir)
        print(f"Folder '{output_dir}' has been deleted.")

    output_dir.mkdir(parents=True)

    # Getting the file list:
    input_files = []
    if dataset_type == DatasetType.BRATS18:
        for folder in ['HGG','LGG']:
            for sub in (input_dir / folder).iterdir():
                input_files.append({
                    'name':sub.name,
                    't1c':sub/f'{sub.name}_t1ce.nii',
                    't1n':sub/f'{sub.name}_t1.nii',
                    't2f':sub/f'{sub.name}_flair.nii',
                    't2w':sub/f'{sub.name}_t2.nii',
                    'seg':sub/f'{sub.name}_seg.nii'
                })
    elif dataset_type == DatasetType.BRATS23:
        for sub in input_dir.iterdir():
            if sub.is_dir():
                input_files.append({
                    'name':sub.name,
                    't1c':sub/f'{sub.name}-t1c.nii.gz',
                    't1n':sub/f'{sub.name}-t1n.nii.gz',
                    't2f':sub/f'{sub.name}-t2f.nii.gz',
                    't2w':sub/f'{sub.name}-t2w.nii.gz',
                    'seg':sub/f'{sub.name}-seg.nii.gz'
                })
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    num_workers = os.cpu_count() or 1

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(preprocess_case, file, output_dir, crop_config,clamp_config,norm_config,dataset_type)
            for file in input_files
        ]

        for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Preprocessing {len(input_files)} files into {output_dir} using {num_workers} workers",
        ):
            future.result()

    print('Preprocessing complete!')


    
    

                
