# BrainchMark

BrainchMark is a research framework for brain tumor segmentation under missing-modality conditions.

The repository is organized as a benchmark-oriented workspace with multiple model directories and a Python package under `src/brainchmark` that currently provides:

- a preprocessing pipeline for BraTS-style datasets
- a Typer-based CLI
- a small GUI built on top of the CLI
- configuration-driven execution through YAML files

## Current Scope

The `brainchmark` package is the operational core of the project.

Implemented or partially implemented components include:

- dataset-aware preprocessing for `brats18` and `brats23`
- cropping, clamping, and normalization configuration/validation
- command-line entrypoints:
  - `brainchmark`
  - `brainchmark-gui`
- a training command scaffold intended for config-driven training setup

The repository also contains several model folders at the project root. Those are part of the broader experimentation workspace, but they are not documented here as a unified public API.

## Installation

BrainchMark targets Python `>=3.13`.

Recommended setup with `uv`:

```bash
uv sync
source .venv/bin/activate
```

That installs the project and the locked dependency set from `uv.lock`.

If you want the development tools too:

```bash
uv sync --extra dev
source .venv/bin/activate
```

If you prefer a manual fallback:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## CLI

After installation, the main entrypoint is:

```bash
brainchmark --help
```

Currently exposed commands include:

- `hello`
- `preprocess`
- `train`

The most mature command is `preprocess`.

## Preprocessing Quick Start

This is the part of the framework that currently has the clearest story and the least drama.

Run preprocessing from a YAML file:

```bash
brainchmark preprocess --config src/brainchmark/data/configs/example.yaml
```

Run preprocessing from CLI flags:

```bash
brainchmark preprocess \
  --input-dir /path/to/brats23 \
  --output-dir /path/to/preprocessed \
  --dataset-type brats23 \
  --crop-mode center \
  --crop-size 128 128 128 \
  --clamp-mode subject \
  --clamp-percentile 0.5 99.5 \
  --norm-mode min_max \
  --norm-min-max-range 0.0 1.0
```

Preprocessing writes one compressed `.npz` file per case containing:

- `images`
- `seg`

For the full option reference and valid parameter combinations, see [docs/preprocessing.md](/home/ocarpentiero/PycharmProjects/IM-Fuse/docs/preprocessing.md).

## Configuration

BrainchMark supports YAML-driven execution for reproducible runs.

Typical pattern:

- put the boring stable stuff in YAML
- override the spicy bits from the CLI when needed

Example:

```bash
brainchmark preprocess \
  --config src/brainchmark/data/configs/example.yaml \
  --output-dir /tmp/brainchmark-run
```

Current example configs live in [src/brainchmark/data/configs](/home/ocarpentiero/PycharmProjects/IM-Fuse/src/brainchmark/data/configs).

## GUI

The GUI is generated from the Typer command signatures and can be launched with:

```bash
brainchmark-gui
```

It is useful for:

- browsing command options
- filling preprocessing parameters interactively
- running CLI-backed workflows without typing long commands

If you do not feel like remembering twenty flags before coffee, this is the button-heavy path.

## Project Layout

Key package files:

- [src/brainchmark/cli.py](/home/ocarpentiero/PycharmProjects/IM-Fuse/src/brainchmark/cli.py): CLI entrypoints
- [src/brainchmark/gui.py](/home/ocarpentiero/PycharmProjects/IM-Fuse/src/brainchmark/gui.py): Tk/ttkbootstrap GUI
- [src/brainchmark/preprocessing/config.py](/home/ocarpentiero/PycharmProjects/IM-Fuse/src/brainchmark/preprocessing/config.py): preprocessing validation and config builders
- [src/brainchmark/preprocessing/pipeline.py](/home/ocarpentiero/PycharmProjects/IM-Fuse/src/brainchmark/preprocessing/pipeline.py): preprocessing execution pipeline
- [src/brainchmark/datasets/config.py](/home/ocarpentiero/PycharmProjects/IM-Fuse/src/brainchmark/datasets/config.py): dataset enums
- [docs/preprocessing.md](/home/ocarpentiero/PycharmProjects/IM-Fuse/docs/preprocessing.md): preprocessing documentation

## Development

Basic checks:

```bash
python -m py_compile src/brainchmark/cli.py
python -m py_compile src/brainchmark/gui.py
python -m py_compile src/brainchmark/preprocessing/config.py
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

## Status

This repository is under active development.

Right now:

- preprocessing is the most solid workflow
- the GUI is useful for exploring and launching commands
- the training command is still more scaffold than battle-tested pipeline

So the current vibe is:

- good for structured preprocessing
- promising for benchmark orchestration
- not done pretending to be finished
