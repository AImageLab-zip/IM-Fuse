"""Relabel the internal dataset's raw segmentations from the legacy BraTS
label convention (NCR/NET=1, edema=2, enhancing tumor=4) to the convention the
trained models actually expect (NCR/NET=1, edema=2, enhancing tumor=3), which
is what every `*-000.npz`/`*-001.npz` file under e.g.
`/work/phd_mimose/mmformer-preprocessed/` and the raw training cases under
`/work/phd_mimose/unpacked/` already use.

`mimose`'s scoring code (`src/mimose/testing/pipeline.py`, e.g.
`softmax_output_dice_class4`) hard-codes label 3 for enhancing tumor with no
remapping step anywhere in the preprocessing pipeline. Internal raw cases use
label 4 for enhancing tumor instead, so `target == 3` is always empty for
those cases -- this silently tanks WT/TC/ET/ETpp Dice and HD95 for every
internal-dataset case, independent of the docker harness vs. `mimose test`
CLI path (both call the same scoring functions on the same wrongly-labeled
data).

Writes a new raw directory tree rather than editing in place: modality files
(t1c/t1n/t2f/t2w) are copied unchanged (hardlinking would be preferable to
avoid doubling the ~2GB dataset on disk, but this filesystem's `link()`
rejects cross-directory hardlinks with EREMOTEIO, so a real copy it is), and
`<id>-seg.nii.gz` is rewritten with label 4 remapped to 3, preserving the
original image's origin/spacing/direction.

Usage:
    python src/onetime_scripts/relabel_internal_raw_labels.py \\
        --input-dir /work/phd_mimose/internal_raw/internal \\
        --output-dir /work/phd_mimose/internal_raw/internal_relabeled
"""
from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="Raw internal dataset (label convention 0/1/2/4).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output raw dataset (label convention 0/1/2/3).")
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args(argv)


def relabel_case(case_dir: Path, output_case_dir: Path) -> None:
    case_id = case_dir.name
    output_case_dir.mkdir(parents=True, exist_ok=True)

    for modality in MODALITIES:
        src = case_dir / f"{case_id}-{modality}.nii.gz"
        dst = output_case_dir / f"{case_id}-{modality}.nii.gz"
        shutil.copyfile(src, dst)

    seg_path = case_dir / f"{case_id}-seg.nii.gz"
    image = sitk.ReadImage(str(seg_path))
    array = sitk.GetArrayFromImage(image)

    unexpected = sorted(set(np.unique(array)) - {0, 1, 2, 4})
    if unexpected:
        raise ValueError(f"{seg_path}: unexpected label values {unexpected}, refusing to relabel blindly")

    array = np.where(array == 4, np.uint8(3), array).astype(array.dtype)

    relabeled = sitk.GetImageFromArray(array)
    relabeled.CopyInformation(image)
    sitk.WriteImage(relabeled, str(output_case_dir / f"{case_id}-seg.nii.gz"))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    case_dirs = sorted(d for d in args.input_dir.iterdir() if d.is_dir())
    if not case_dirs:
        raise SystemExit(f"no case directories found under {args.input_dir}")

    failures = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(relabel_case, case_dir, args.output_dir / case_dir.name): case_dir.name
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
            task_id = progress.add_task(f"Relabel {len(case_dirs)} cases", total=len(futures))
            for future in as_completed(futures):
                case_id = futures[future]
                exc = future.exception()
                if exc is not None:
                    failures.append((case_id, str(exc)))
                progress.update(task_id, advance=1)

    print(f"relabeled {len(case_dirs) - len(failures)}/{len(case_dirs)} cases -> {args.output_dir}")
    if failures:
        print(f"\n{len(failures)} case(s) failed:")
        for case_id, message in failures:
            print(f"  {case_id}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
