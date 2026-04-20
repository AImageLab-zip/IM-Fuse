# BrainchMark

BrainchMark is the active benchmark package in this repository for brain tumor segmentation under missing-modality conditions.

The operational code lives under `src/brainchmark`. The rest of the repository also contains legacy model directories and experiment workspaces, but `brainchmark` is the maintained CLI/package surface.

## What Exists Today

BrainchMark currently provides:

- dataset-aware preprocessing for BraTS-style datasets
- a Typer CLI with `preprocess`, `train`, and `test`
- YAML-driven execution with CLI overrides
- a small GUI generated from the CLI surface
- active model integrations for `imfuse` and `mmformer`

Supported dataset types:

- `brats18`
- `brats23`

## Installation

The package metadata currently pins Python to `3.12.13`.

Recommended setup with `uv`:

```bash
uv sync
source .venv/bin/activate
```

For development tools too:

```bash
uv sync --extra dev
source .venv/bin/activate
```

Manual fallback:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Entry Points

After installation:

```bash
brainchmark --help
brainchmark-gui
```

CLI commands:

- `preprocess`
- `train`
- `test`
- `self-destruct`

`self-destruct` is obviously not part of the serious workflow...but that’s exactly what I want you to believe… so why not try it, hehehe.

## Preprocessing Quick Start

Run preprocessing from YAML:

```bash
brainchmark preprocess --config src/brainchmark/data/configs/preprocessing.yaml
```

Run preprocessing from flags:

```bash
brainchmark preprocess \
  --input-dir /path/to/brats23 \
  --output-dir /path/to/preprocessed \
  --dataset-type brats23 \
  --crop-mode non_empty \
  --crop-min-size 128 128 128 \
  --norm-mode subject_zscore
```

The preprocessing output is one compressed `.npz` file per case containing:

- `images`
- `seg`

More detail: [docs/preprocessing.md](docs/preprocessing.md)

## Training Quick Start

Example with a reference config:

```bash
brainchmark train --config src/brainchmark/data/configs/imfuse_23.yaml
```

Override selected values from the CLI:

```bash
brainchmark train \
  --config src/brainchmark/data/configs/mmformer_23.yaml \
  --batch-size 2 \
  --wandb-run-name mmformer-debug
```

The current reference configs are:

- `src/brainchmark/data/configs/imfuse_18.yaml`
- `src/brainchmark/data/configs/imfuse_23.yaml`
- `src/brainchmark/data/configs/mmformer_18.yaml`
- `src/brainchmark/data/configs/mmformer_23.yaml`

More detail: [docs/training.md](docs/training.md)

## Testing Quick Start

Evaluate a checkpoint across the standard 15 mask patterns:

```bash
brainchmark test --config src/brainchmark/data/configs/imfuse_23.yaml
```

Or directly:

```bash
brainchmark test \
  --data-dir /path/to/preprocessed \
  --checkpoint-path /path/to/checkpoint.pth \
  --output-path /path/to/results.txt \
  --dataset-type brats23 \
  --model imfuse
```

The test command writes the text report at `output_path` and also creates a sibling Excel summary with the same stem and `.xlsx` suffix.

More detail: [docs/testing.md](docs/testing.md)

## Configuration Model

The common pattern is:

- keep stable experiment settings in YAML
- override a few values from the CLI

CLI values override YAML values when both are present.

Reference configs live in [src/brainchmark/data/configs](src/brainchmark/data/configs).

## GUI

Launch the GUI with:

```bash
brainchmark-gui
```

It is useful for:

- browsing command options
- filling config/CLI parameters interactively
- launching CLI-backed workflows without typing long commands

## Project Layout

Key package areas:

- [src/brainchmark/cli.py](src/brainchmark/cli.py): CLI entrypoints
- [src/brainchmark/gui.py](src/brainchmark/gui.py): GUI launcher
- [src/brainchmark/preprocessing/](src/brainchmark/preprocessing): preprocessing config and pipeline
- [src/brainchmark/models/](src/brainchmark/models): active model integrations
- [src/brainchmark/datasets/](src/brainchmark/datasets): dataset abstractions and masking logic
- [src/brainchmark/training/](src/brainchmark/training): trainer/runtime/config code
- [src/brainchmark/testing/](src/brainchmark/testing): evaluation pipeline

Documentation:

- [docs/preprocessing.md](docs/preprocessing.md)
- [docs/training.md](docs/training.md)
- [docs/testing.md](docs/testing.md)
- [docs/extending.md](docs/extending.md)

## Development

Basic checks:

```bash
python -m py_compile src/brainchmark/cli.py
python -m py_compile src/brainchmark/gui.py
python -m py_compile src/brainchmark/preprocessing/config.py
python -m py_compile src/brainchmark/training/trainers/base_trainer.py
```

Lint and type-check:

```bash
ruff check .
mypy src
```

Run tests:

```bash
pytest
```

## Current Status

The current state is:

- preprocessing is solid and actively usable
- training is implemented and config-driven
- testing is implemented for mask-sweep evaluation
- the active trainer stack is centered on the IMFuse-style trainer/runtime, which is also currently reused for mmFormer
- the docs and configs are intended to reflect the active `brainchmark` package, not the legacy folders
