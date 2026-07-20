"""Docker entrypoint for the CPU-only preprocessing image: read a
`.mimosepkg`'s embedded preprocessing config and preprocess raw case
directories into `.npz` + a matching `split.json`, ready to be handed to the
GPU testing container's `--data-dir`/`--split-file` (see harness/run.py).

Deliberately never imports the model class / torch -- only
`mimose.preprocessing`/`mimose.enums`, so this image never needs a GPU or
CUDA-dependent packages (mamba_ssm, causal_conv1d) to be importable.

Usage:

    python -m harness.preprocess_run \
        --package /submission/run.mimosepkg \
        --raw-data-dir /input \
        --output-dir /shared
"""
from __future__ import annotations

import argparse
from pathlib import Path

from harness.load import extract_package
from harness.preprocess import PreprocessingConfigError, preprocess_raw_cases, write_test_only_split


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True, help="Path to the .mimosepkg archive.")
    parser.add_argument("--raw-data-dir", type=Path, required=True, help="Directory of raw case subdirectories.")
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write preprocessed .npz cases + split.json into. "
             "Mount this same directory into the testing container.",
    )
    parser.add_argument("--extract-dir", type=Path, default=Path("/tmp/mimosepkg"))
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    resolved = extract_package(args.package, args.extract_dir)
    manifest = resolved["manifest"]
    preprocessing_config = manifest.get("preprocessing")
    print(f"package={args.package} mimose_version={manifest['mimose_version']}")

    if preprocessing_config is None:
        raise PreprocessingConfigError(
            f"{args.package} has no embedded 'preprocessing' config in its manifest "
            "-- it was exported before this was tracked, or from a minimal "
            "export-only YAML. Re-export with the full training config, or "
            "preprocess manually and point the testing container at --data-dir "
            "directly instead of chaining this preprocessing container."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "preprocessed"
    case_ids = preprocess_raw_cases(
        args.raw_data_dir, data_dir, preprocessing_config, num_workers=args.num_workers
    )
    if not case_ids:
        raise SystemExit(f"no case directories found under {args.raw_data_dir}")

    dataset_type = str(preprocessing_config["dataset_type"])
    split_path = write_test_only_split(case_ids, dataset_type, args.output_dir / "split.json")

    print(f"preprocessed {len(case_ids)} cases -> {data_dir}")
    print(f"split file -> {split_path} (dataset_type={dataset_type})")


if __name__ == "__main__":
    main()
