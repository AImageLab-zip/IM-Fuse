# BrainchMark🧠

BrainchMark is the main package in this repository for benchmarking brain tumor segmentation with missing imaging modalities.

The code you should actually use lives in `src/brainchmark`. The rest of the repo includes legacy model folders and older experiment workspaces that are still useful for reference, but the maintained CLI and package surface is `brainchmark`.

## Installation

The package metadata currently pins Python to `3.12.13`.

Recommended setup with `uv`:

```bash
git clone https://github.com/AImageLab-zip/IM-Fuse
cd IM-Fuse
bash install.sh
```

The installation step can take a while the first time. Some dependencies, including the local Mamba build path, may need to compile native code before the environment is ready.

## Supported Hardware and Software

BrainchMark is currently supported on:

- Linux
- NVIDIA GPUs based on the Turing architecture or newer

In practice, this means GPUs such as the NVIDIA RTX 2000 series and later are supported.

Linux is the only supported operating system for now.

## Entry Points

After installation:

```bash
brainchmark --help
brainchmark-gui
```

CLI commands:

- `setup`
- `preprocess`
- `train`
- `test`
- `self-destruct`

`self-destruct` is obviously part of the serious workflow.

## What Exists Today

BrainchMark currently provides:

- an interactive `setup` command for generating local configs from packaged templates
- dataset-aware preprocessing for BraTS-style datasets
- a Typer CLI with `preprocess`, `train`, and `test`
- YAML-driven execution with CLI overrides
- a small GUI generated from the CLI surface
- active model integrations for `imfuse`, `mmformer`, and `dcseg`

Supported dataset types:

- `brats18`
- `brats23`
## Preprocessing Quick Start

If you want to rewrite the packaged reference configs first:

```bash
brainchmark setup
```

This copies the packaged templates from `src/brainchmark/data/config_templates` into `src/brainchmark/data/configs`, then patches the local dataset, preprocessing-output, and artifact-root paths.

Run preprocessing from YAML:

```bash
brainchmark preprocess --config src/brainchmark/data/configs/imfuse_23.yaml
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
brainchmark train --config imfuse_23.yaml
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
- `src/brainchmark/data/configs/dcseg_18.yaml`
- `src/brainchmark/data/configs/dcseg_23.yaml`

More detail: [docs/training.md](docs/training.md)

## Testing Quick Start

Evaluate a checkpoint across the standard 15 mask patterns:

```bash
brainchmark test --config imfuse_23.yaml
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

For a dedicated guide to writing configs, see [docs/yaml-config.md](docs/yaml-config.md).

For training startup, the CLI now shows a Rich `dots` status immediately while it loads training modules, resolves config, and initializes the trainer. This is expected before the full launch summary panel appears.

## GUI

Launch the GUI with:

```bash
brainchmark-gui
```

It is useful for:

- browsing command options
- filling config/CLI parameters interactively
- launching CLI-backed workflows without typing long commands

## Contributing

Contributions are welcome and will be evaluated quickly.

If you want to add your own model, trainer, dataset integration, runtime component, or other custom extension, follow [docs/extending.md](docs/extending.md).

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

- [docs/yaml-config.md](docs/yaml-config.md)
- [docs/preprocessing.md](docs/preprocessing.md)
- [docs/training.md](docs/training.md)
- [docs/testing.md](docs/testing.md)
- [docs/extending.md](docs/extending.md)
- [docs/components/README.md](docs/components/README.md)

If you need very detailed, component-by-component extension notes, start with [docs/components/README.md](docs/components/README.md).
