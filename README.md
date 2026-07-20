# MiMoSe🧠

MiMoSe is the main package in this repository for benchmarking brain tumor segmentation with missing imaging modalities.

The code you should actually use lives in `src/mimose`. The rest of the repo includes legacy model folders and older experiment workspaces that are still useful for reference, but the maintained CLI and package surface is `mimose`.

We also encourage fellow researchers to extend the repository and open PRs with new models, trainers, datasets, and evaluation ideas. More detail: [docs/extending.md](docs/extending.md).

## Installation

The package metadata currently pins Python to `3.12.13`.

Recommended setup with `uv`:

Ensure that `uv` is installed system-wide before running the setup steps below.

```bash
git clone https://github.com/AImageLab-zip/MiMoSe
cd MiMoSe
bash install.sh
```

The installation step can take a while the first time. Some dependencies, including the local Mamba build path, may need to compile native code before the environment is ready.

After the installation, rewrite the packaged reference configs running:

```bash
mimose setup
```

## Supported Hardware and Software

MiMoSe is currently supported on:

- Linux
- NVIDIA GPUs based on the Turing architecture or newer

In practice, this means GPUs such as the NVIDIA RTX 2000 series and later are supported.

Linux is the only supported operating system for now.

## Entry Points

After installation:

```bash
mimose --help
mimose --version
```

CLI commands:

- `setup`
- `update`
- `preprocess`
- `preprocess-train`
- `train`
- `test`
- `flops`
- `self-destruct`

`self-destruct` is obviously part of the serious workflow.

## What Exists Today

MiMoSe currently provides:

- an interactive `setup` command for generating local configs from packaged templates
- dataset-aware preprocessing for BraTS-style datasets
- a Typer CLI with `preprocess`, `train`, and `test`
- YAML-driven execution with CLI overrides
- active model integrations for `imfuse`, `mmformer`, `dcseg`, `rfnet`, and `tinymimosa`

Supported dataset types:

- `brats18`
- `brats23`
- `brats25`

## Preprocessing Quick Start

This copies the packaged templates from `src/mimose/data/config_templates` into `src/mimose/data/configs`, then patches the local dataset, preprocessing-output, and artifact-root paths.

Run preprocessing from YAML:

```bash
mimose preprocess --config imfuse_23.yaml
```

Run preprocessing followed immediately by training:

```bash
mimose preprocess-train --config imfuse_23.yaml
```

The preprocessing output is one compressed `.npz` file per case containing:

- `images`
- `seg`

More detail: [docs/preprocessing.md](docs/preprocessing.md)

## Training Quick Start

Example with a reference config:

```bash
mimose train --config imfuse_23.yaml
```

More detail: [docs/training.md](docs/training.md)

## Testing Quick Start

Evaluate a checkpoint across the standard 15 mask patterns:

```bash
mimose test --config imfuse_23.yaml
```

The test command writes the text report at `output_path` and also creates a sibling Excel summary with the same stem and `.xlsx` suffix.

More detail: [docs/testing.md](docs/testing.md)

## Configuration Model

The common pattern is:

- keep stable experiment settings in YAML
- override a few values from the CLI

CLI values override YAML values when both are present.

Reference configs live in [src/mimose/data/configs](src/mimose/data/configs).

For a dedicated guide to writing configs, see [docs/yaml-config.md](docs/yaml-config.md).

For training startup, the CLI now shows a Rich `dots` status immediately while it loads training modules, resolves config, and initializes the trainer. This is expected before the full launch summary panel appears.

## Contributing

Contributions are welcome and will be evaluated quickly.

## Versioning

The installed package version is exposed through:

```bash
mimose --version
```

If you want to add your own model, trainer, dataset integration, runtime component, or other custom extension, follow [docs/extending.md](docs/extending.md).

## Project Layout

Key package areas:

- [src/mimose/cli.py](src/mimose/cli.py): CLI entrypoints
- [src/mimose/gui.py](src/mimose/gui.py): GUI launcher
- [src/mimose/preprocessing/](src/mimose/preprocessing): preprocessing config and pipeline
- [src/mimose/models/](src/mimose/models): active model integrations
- [src/mimose/datasets/](src/mimose/datasets): dataset abstractions and masking logic
- [src/mimose/training/](src/mimose/training): trainer/runtime/config code
- [src/mimose/testing/](src/mimose/testing): evaluation pipeline

Documentation:

- [docs/yaml-config.md](docs/yaml-config.md)
- [docs/preprocessing.md](docs/preprocessing.md)
- [docs/training.md](docs/training.md)
- [docs/testing.md](docs/testing.md)
- [docs/extending.md](docs/extending.md)
- [docs/components/README.md](docs/components/README.md)

If you need very detailed, component-by-component extension notes, start with [docs/components/README.md](docs/components/README.md).
