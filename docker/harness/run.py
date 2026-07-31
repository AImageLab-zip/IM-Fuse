"""Docker entrypoint: ingest a `.mimosepkg` archive and run mask-sweep test
inference against a preprocessed data directory, using `harness.testing_loop`
-- a harness-local copy of the same mask-sweep/Dice/HD95 loop `mimose test`
uses, kept outside `src/mimose` because `mimose.testing` is deliberately
excluded from every exported `.mimosepkg` (see `EXCLUDED_TOP_LEVEL_DIRS` in
`src/mimose/model_export.py`).

See `.claude/docker-inference-pipeline-guide.md` for the design this
implements. Usage (inside the container, or locally against a venv that has
mimose's runtime dependencies installed):

    python -m harness.run \
        --package /submission/run.mimosepkg \
        --data-dir /input \
        --output-dir /output \
        --dataset-type brats18 \
        --split-file /input/split.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from harness.load import resolve_package


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True, help="Path to the .mimosepkg archive.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory of preprocessed .npz test cases.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write the test report into.")
    parser.add_argument(
        "--dataset-type", required=True, choices=("brats18", "brats23", "brats25"),
        help="Dataset type; also selects which key of --split-file to read.",
    )
    parser.add_argument("--split-file", type=Path, required=True, help="Split JSON listing test case ids.")
    parser.add_argument("--fold", type=int, default=None, help="Fold index, if --split-file is fold-nested.")
    parser.add_argument("--extract-dir", type=Path, default=Path("/tmp/mimosepkg"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    resolved = resolve_package(args.package, args.extract_dir)
    manifest = resolved["manifest"]
    print(
        f"package={args.package} mimose_version={manifest['mimose_version']} "
        f"model_class={manifest['model_class']} run_suffix={manifest.get('run_suffix')}"
    )

    from harness.testing_loop import run_testing_loop

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "test_report.txt"

    result_path = run_testing_loop(
        data_dir=args.data_dir,
        output_path=output_path,
        checkpoint_path=resolved["checkpoint_path"],
        dataset_type=args.dataset_type,
        model_class=resolved["model_class"],
        model_kwargs=resolved["model_kwargs"],
        split_file=args.split_file,
        fold=args.fold,
        num_workers=args.num_workers,
        seed=args.seed,
        fp16=args.fp16,
    )
    print(f"Test report written to {result_path}")
    print(f"Excel summary written to {result_path.with_suffix('.xlsx')}")
    print(f"Per-subject scores written to {result_path.with_name(f'{result_path.stem}_per_subject.csv')}")


if __name__ == "__main__":
    main()
