"""Center crop/pad the relabeled internal dataset to the fixed BraTS-GLI
acquisition grid (182, 218, 182) used everywhere else in this codebase (see
`BRATS_FULL_VOLUME_SHAPE` in `src/mimose/testing/pipeline.py` and
`src/mimose/training/trainers/base_trainer.py`, and `DEFAULT_INPUT_SHAPE` in
`src/mimose/models/tiny_mimosa.py` / `src/mimose/models/many_mimosas.py`).

The internal dataset's native grid is (240, 240, 155) (in-plane 240x240,
155 axial slices), which does not match (182, 218, 182) in any axis: x and y
are larger than the target and need center cropping, while z is smaller and
needs center padding. This script does both, per axis, on every volume
(modalities + segmentation), adjusting each image's origin so the cropped/
padded volume stays correctly positioned in physical space (`sitk.Crop` and
`sitk.ConstantPad` both update the origin for us).

Writes a new raw directory tree rather than editing in place, mirroring
`relabel_internal_raw_labels.py`.

Usage:
    python src/onetime_scripts/center_crop_internal_relabeled.py \\
        --input-dir /work/phd_mimose/internal_raw/internal_relabeled \\
        --output-dir /work/phd_mimose/internal_raw/internal_relabeled_cropped
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import SimpleITK as sitk
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

MODALITIES = ("t1c", "t1n", "t2f", "t2w")
TARGET_SIZE = (182, 218, 182)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="Relabeled internal dataset (native grid).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dataset, center cropped/padded to 182x218x182.")
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args(argv)


def center_crop_or_pad(image: sitk.Image, target_size: tuple[int, int, int]) -> sitk.Image:
    current_size = image.GetSize()

    crop_lower = [0, 0, 0]
    crop_upper = [0, 0, 0]
    pad_lower = [0, 0, 0]
    pad_upper = [0, 0, 0]
    for axis, (current, target) in enumerate(zip(current_size, target_size, strict=True)):
        diff = current - target
        if diff > 0:
            crop_lower[axis] = diff // 2
            crop_upper[axis] = diff - crop_lower[axis]
        elif diff < 0:
            total_pad = -diff
            pad_lower[axis] = total_pad // 2
            pad_upper[axis] = total_pad - pad_lower[axis]

    if any(crop_lower) or any(crop_upper):
        image = sitk.Crop(image, crop_lower, crop_upper)
    if any(pad_lower) or any(pad_upper):
        image = sitk.ConstantPad(image, pad_lower, pad_upper, 0.0)
    return image


def crop_case(case_dir: Path, output_case_dir: Path) -> None:
    case_id = case_dir.name
    output_case_dir.mkdir(parents=True, exist_ok=True)

    for modality in MODALITIES:
        src = case_dir / f"{case_id}-{modality}.nii.gz"
        image = sitk.ReadImage(str(src))
        cropped = center_crop_or_pad(image, TARGET_SIZE)
        sitk.WriteImage(cropped, str(output_case_dir / f"{case_id}-{modality}.nii.gz"))

    seg_path = case_dir / f"{case_id}-seg.nii.gz"
    seg = sitk.ReadImage(str(seg_path))
    cropped_seg = center_crop_or_pad(seg, TARGET_SIZE)
    sitk.WriteImage(cropped_seg, str(output_case_dir / f"{case_id}-seg.nii.gz"))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    case_dirs = sorted(d for d in args.input_dir.iterdir() if d.is_dir())
    if not case_dirs:
        raise SystemExit(f"no case directories found under {args.input_dir}")

    failures = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(crop_case, case_dir, args.output_dir / case_dir.name): case_dir.name
            for case_dir in case_dirs
        }
        with Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task_id = progress.add_task(f"Crop/pad {len(case_dirs)} cases", total=len(futures))
            for future in as_completed(futures):
                case_id = futures[future]
                exc = future.exception()
                if exc is not None:
                    failures.append((case_id, str(exc)))
                progress.update(task_id, advance=1)

    print(f"cropped/padded {len(case_dirs) - len(failures)}/{len(case_dirs)} cases -> {args.output_dir}")
    if failures:
        print(f"\n{len(failures)} case(s) failed:")
        for case_id, message in failures:
            print(f"  {case_id}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
