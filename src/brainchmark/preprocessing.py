from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import typer

class DatasetType(StrEnum):
    BRATS18 = "brats18"
    BRATS23 = "brats23"

class CropMode(StrEnum):
    NONE = "none"
    CENTER = "center"
    NON_EMPTY = "non-empty"


@dataclass(frozen=True)
class CropConfig:
    mode: CropMode
    size: tuple[int, int, int] | None = None
    min_size: tuple[int, int, int] | None = None


def build_crop_config(
    crop_mode: CropMode,
    crop_size: tuple[int, int, int] | None,
    crop_min_size: tuple[int, int, int] | None,
) -> CropConfig:
    """Validate crop-related CLI options and return a structured config."""
    if crop_mode is CropMode.NONE:
        if crop_size is not None or crop_min_size is not None:
            raise typer.BadParameter(
                "Do not pass --crop-size or --crop-min-size with --crop-mode none."
            )
        return CropConfig(mode=crop_mode)

    if crop_mode is CropMode.CENTER:
        if crop_size is None:
            raise typer.BadParameter(
                "--crop-size is required when --crop-mode center."
            )
        if crop_min_size is not None:
            raise typer.BadParameter(
                "--crop-min-size is not used with --crop-mode center."
            )
        return CropConfig(mode=crop_mode, size=crop_size)

    if crop_min_size is None:
        raise typer.BadParameter(
            "--crop-min-size is required when --crop-mode non-empty."
        )
    if crop_size is not None:
        raise typer.BadParameter(
            "--crop-size is not used with --crop-mode non-empty."
        )
    return CropConfig(mode=crop_mode, min_size=crop_min_size)


def run_preprocessing(
    input_dir: Path,
    output_dir: Path,
    dataset_type: str,
    crop_config: CropConfig,
) -> None:
    """Run the preprocessing pipeline with the selected crop configuration."""
    print(
        "Running preprocessing with "
        f"dataset_type={dataset_type}, "
        f"input_dir={input_dir}, "
        f"output_dir={output_dir}, "
        f"crop_mode={crop_config.mode}, "
        f"crop_size={crop_config.size}, "
        f"crop_min_size={crop_config.min_size}"
    )
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
                input_files.append({
                    'name':sub.name,
                    't1c':sub/f'{sub.name}-t1c.nii',
                    't1n':sub/f'{sub.name}-t1n.nii',
                    't2f':sub/f'{sub.name}-t2f.nii',
                    't2w':sub/f'{sub.name}-t2w.nii',
                    'seg':sub/f'{sub.name}-seg.nii'
                })
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    

                
