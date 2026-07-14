"""Preprocess raw BraTS-style case directories using the exact crop/clamp/norm
settings embedded in a `.mimosepkg`'s manifest (see `preprocessing_config` in
`mimose.model_export.build_export_package`), reusing the same
`preprocess_case()` the `mimose preprocess` CLI uses.

Unlike `mimose preprocess`, this does not filter cases against the shipped
split.json -- the operator has already scoped `--raw-data-dir` to exactly the
cases they want evaluated, so every case directory found there is processed.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


class PreprocessingConfigError(ValueError):
    """Raised when a package has no (or an incomplete) embedded preprocessing config."""


def _as_tuple(value: Any) -> tuple | None:
    return tuple(value) if value is not None else None


def preprocess_raw_cases(
    raw_dir: Path,
    output_dir: Path,
    preprocessing_config: dict[str, Any] | None,
    *,
    num_workers: int = 8,
) -> list[str]:
    """Preprocess every case directory under `raw_dir` into `output_dir`.

    Returns the sorted list of case ids that were successfully processed.
    Each case directory `raw_dir/<id>/` must contain
    `<id>-{t1c,t1n,t2f,t2w,seg}.nii.gz`, matching the layout
    `mimose preprocess` expects.
    """
    if preprocessing_config is None:
        raise PreprocessingConfigError(
            "this .mimosepkg has no embedded preprocessing config (it was exported "
            "before this was tracked, or from a minimal export-only YAML) -- "
            "preprocess the raw data yourself with `mimose preprocess` and pass "
            "--data-dir/--split-file to the harness instead of --raw-data-dir."
        )

    from mimose.preprocessing.config import build_clamp_config, build_crop_config, build_norm_config
    from mimose.preprocessing.pipeline import preprocess_case

    crop_config = build_crop_config(
        preprocessing_config["crop_mode"],
        _as_tuple(preprocessing_config.get("crop_size")),
        _as_tuple(preprocessing_config.get("crop_min_size")),
    )
    clamp_config = build_clamp_config(
        preprocessing_config["clamp_mode"],
        _as_tuple(preprocessing_config.get("clamp_percentile")),
        _as_tuple(preprocessing_config.get("clamp_min")),
        _as_tuple(preprocessing_config.get("clamp_max")),
    )
    norm_config = build_norm_config(
        preprocessing_config["norm_mode"],
        _as_tuple(preprocessing_config.get("norm_min_max_range")),
        _as_tuple(preprocessing_config.get("norm_mean")),
        _as_tuple(preprocessing_config.get("norm_std")),
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = []
    for sub in sorted(raw_dir.iterdir()):
        if not sub.is_dir():
            continue
        input_files.append(
            {
                "name": sub.name,
                "t1c": sub / f"{sub.name}-t1c.nii.gz",
                "t1n": sub / f"{sub.name}-t1n.nii.gz",
                "t2f": sub / f"{sub.name}-t2f.nii.gz",
                "t2w": sub / f"{sub.name}-t2w.nii.gz",
                "seg": sub / f"{sub.name}-seg.nii.gz",
            }
        )

    processed: list[str] = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(preprocess_case, file, output_dir, crop_config, clamp_config, norm_config): file["name"]
            for file in input_files
        }
        for future in as_completed(futures):
            name = futures[future]
            future.result()
            processed.append(name)

    return sorted(processed)


def write_test_only_split(case_ids: list[str], dataset_type: str, split_path: Path) -> Path:
    """Write a split.json-shaped file whose entire `test` set is `case_ids`,
    so `run_testing` can evaluate every preprocessed case with no separate
    hand-authored split file.
    """
    import json

    split_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        dataset_type: {
            "train": [],
            "val": [],
            "test": [{"sub": case_id, "mask": None} for case_id in case_ids],
        }
    }
    split_path.write_text(json.dumps(payload, indent=2))
    return split_path
