from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

BRATS23_ROOT = Path(
    "/work/grana_neuro/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/"
)
MODALITY_SUFFIXES = ("-t2f.nii.gz", "-t1c.nii.gz", "-t1n.nii.gz", "-t2w.nii.gz")
MIN_CROP_SIZE = 128


def sup_128(xmin: int, xmax: int) -> tuple[int, int]:
    if xmax - xmin < MIN_CROP_SIZE:
        gap = int((MIN_CROP_SIZE - (xmax - xmin)) / 2)
        xmax = xmax + gap + 1
        xmin = xmin - gap
    if xmin < 0:
        xmax -= xmin
        xmin = 0
    return xmin, xmax


def crop_bounds(vol: np.ndarray) -> tuple[int, int, int, int, int, int]:
    if vol.ndim == 4:
        vol = np.amax(vol, axis=0)
    if vol.ndim != 3:
        raise ValueError(f"expected a 3D or 4D volume, got shape {vol.shape}")

    x_nonzeros, y_nonzeros, z_nonzeros = np.where(vol != 0)
    if len(x_nonzeros) == 0:
        raise ValueError("volume contains only zeros")

    x_min, x_max = int(np.amin(x_nonzeros)), int(np.amax(x_nonzeros))
    y_min, y_max = int(np.amin(y_nonzeros)), int(np.amax(y_nonzeros))
    z_min, z_max = int(np.amin(z_nonzeros)), int(np.amax(z_nonzeros))

    x_min, x_max = sup_128(x_min, x_max)
    y_min, y_max = sup_128(y_min, y_max)
    z_min, z_max = sup_128(z_min, z_max)

    return x_min, x_max, y_min, y_max, z_min, z_max


def load_case_volume(case_dir: Path) -> np.ndarray:
    case_id = case_dir.name
    volumes = []
    for suffix in MODALITY_SUFFIXES:
        modality_path = case_dir / f"{case_id}{suffix}"
        if not modality_path.is_file():
            raise FileNotFoundError(f"missing modality file: {modality_path}")
        volumes.append(np.asarray(nib.load(str(modality_path)).get_fdata(), dtype=np.float32))
    return np.stack(volumes, axis=0)


def process_case(case_dir: Path) -> tuple[str, tuple[int, int, int] | None, str | None]:
    try:
        volume = load_case_volume(case_dir)
        x_min, x_max, y_min, y_max, z_min, z_max = crop_bounds(volume)
        return case_dir.name, (x_max - x_min, y_max - y_min, z_max - z_min), None
    except Exception as exc:
        return case_dir.name, None, str(exc)


def main() -> None:
    case_dirs = sorted(path for path in BRATS23_ROOT.iterdir() if path.is_dir())
    if not case_dirs:
        raise RuntimeError(f"no case directories found under {BRATS23_ROOT}")

    crop_sizes: list[tuple[int, int, int]] = []
    failed_cases: list[tuple[str, str]] = []
    max_workers = min(len(case_dirs), os.cpu_count() or 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        iterator = executor.map(process_case, case_dirs)
        for case_id, crop_size, error in tqdm(iterator, total=len(case_dirs)):
            if crop_size is not None:
                crop_sizes.append(crop_size)
            else:
                failed_cases.append((case_id, error or "unknown error"))

    if not crop_sizes:
        raise RuntimeError("no valid cases were processed")

    crop_sizes_array = np.asarray(crop_sizes, dtype=np.float64)
    average_crop = crop_sizes_array.mean(axis=0)
    min_crop = crop_sizes_array.min(axis=0)
    max_crop = crop_sizes_array.max(axis=0)

    print(f"processed_cases: {len(crop_sizes)}")
    print(f"failed_cases: {len(failed_cases)}")
    print(f"workers: {max_workers}")
    print(
        "average_nonempty_crop_size: "
        f"({average_crop[0]:.3f}, {average_crop[1]:.3f}, {average_crop[2]:.3f})"
    )
    print(f"min_crop_size: ({int(min_crop[0])}, {int(min_crop[1])}, {int(min_crop[2])})")
    print(f"max_crop_size: ({int(max_crop[0])}, {int(max_crop[1])}, {int(max_crop[2])})")

    if failed_cases:
        print("failed_case_details:")
        for case_id, message in failed_cases:
            print(f"  {case_id}: {message}")


if __name__ == "__main__":
    main()
