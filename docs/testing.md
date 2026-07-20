# Testing

`mimose test` evaluates a checkpoint by sweeping the standard 15 missing-modality masks and writing a text report plus an Excel summary.

The active testing path currently uses:

- the preprocessed `.npz` dataset format used by the training pipeline
- `IMFuseDataset` for sample loading
- the selected model's `predict(images, mask)` method

## Basic Usage

Run from YAML:

```bash
mimose test --config src/mimose/data/configs/imfuse_23.yaml
```

Run from CLI:

```bash
mimose test \
  --data-dir /path/to/preprocessed \
  --checkpoint-path /path/to/final_weights_only.safetensors \
  --output-path /path/to/results.txt \
  --dataset-type brats23 \
  --model dcseg
```

Run from CLI with an online checkpoint cache:

```bash
mimose test \
  --data-dir /path/to/preprocessed \
  --art-dir /path/to/artifacts/run1 \
  --online \
  --hf-repo owner/repo \
  --hf-run-name run1 \
  --output-path /path/to/results.txt \
  --dataset-type brats23 \
  --model dcseg
```

## Required Inputs

The test command requires:

- `data_dir`
- `output_path`
- `dataset_type`

Use one of these checkpoint inputs:

- `checkpoint_path` for a local `.safetensors` weights file
- `hf_repo` together with `hf_run_name`, `online=true`, and `art_dir` for a downloaded cached checkpoint
- `art_dir` by itself to auto-resolve `art_dir/checkpoints/final_weights_only.safetensors`

`model` is optional in the CLI because it defaults to `imfuse`, but for a real run you should set it explicitly unless the config already does.

For the overall MiMoSe YAML format, see [docs/yaml-config.md](yaml-config.md).
For a very detailed extension guide for the testing path, see [docs/components/testing.md](components/testing.md).

Active tested model choices currently include:

- `imfuse`
- `mmformer`
- `dcseg`
- `rfnet`
- `tinymimosa`

Dataset type choices:

- `brats18`
- `brats23`
- `brats25`

## What It Produces

The command writes:

- a text report at `output_path`
- an Excel summary at `output_path` with the suffix changed to `.xlsx`

The text report contains one line per modality mask plus an average summary.

It evaluates these 15 masks:

- single-modality subsets
- two-modality subsets
- three-modality subsets
- all four modalities

The current metrics reported are:

- `WT`
- `TC`
- `ET`
- `ETpp`

## Current Constraints

- testing currently requires CUDA
- batch size is fixed to `1`
- the evaluation path is built for the BraTS-style 4-class setup used by the active models

If a checkpoint is missing, the command raises a clean CLI error.

If `online` is enabled, MiMoSe downloads the checkpoint from Hugging Face using `hf_repo` and `hf_run_name`, stores it at `art_dir/checkpoints/hf/<repo>/<run>/final_weights_only.safetensors`, and reuses that cached file on later runs if it is already present.

## YAML Fields

The test command reads these fields from the combined config files:

- `data_dir`
- `checkpoint_path`
- `art_dir`
- `online`
- `hf_repo`
- `hf_run_name`
- `output_path`
- `model`
- `custom_model_kwargs`
- `split_file`
- `num_workers`
- `seed`
- `dataset_type`

## Example

```yaml
data_dir: /work/grana_neuro/mimose/dcseg23-preprocessed
checkpoint_path: /work/grana_neuro/mimose/runs/dcseg23/checkpoints/final_weights_only.safetensors
output_path: /work/grana_neuro/mimose/runs/dcseg23/results.txt
model: dcseg
custom_model_kwargs:
  num_cls: 4
  fusion_type: RFM
dataset_type: brats23
num_workers: 8
seed: 42
```

Online-checkpoint example:

```yaml
data_dir: /work/grana_neuro/mimose/dcseg23-preprocessed
art_dir: /work/grana_neuro/mimose/runs/dcseg23
online: true
hf_repo: owner/repo
hf_run_name: dcseg23
output_path: /work/grana_neuro/mimose/runs/dcseg23/results.txt
model: dcseg
dataset_type: brats23
num_workers: 8
seed: 42
```

## Related Docs

- [README.md](../README.md)
- [docs/yaml-config.md](yaml-config.md)
- [docs/training.md](training.md)
