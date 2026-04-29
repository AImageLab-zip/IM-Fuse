# YAML Config Guide

This document explains how to write a YAML config for MiMoSe.

If you are extending the codebase itself rather than just writing configs, the detailed component extension guides live under [docs/components/README.md](components/README.md).

MiMoSe uses a flat YAML style. One file can contain fields for:

- preprocessing
- training
- testing

Each command reads only the keys it needs and ignores the rest. While it is possible to have 3 different yaml files, it is strongly consigliato to use a unified config, since there a

## General Rules

- CLI values override YAML values when both are provided.
- Most fields are top-level keys.
- `custom_model_kwargs`, `custom_loss_kwargs`, and `custom_trainer_kwargs` are nested dictionaries.
- Shipped examples live in `src/brainchmark/data/configs/`.

The easiest workflow is:

1. copy a shipped config
2. change paths and run-specific values
3. keep the rest until you need a deliberate override

## Common Field Groups

### Preprocessing

- `input_dir`: path
- `output_dir`: path
- `dataset_type`: string enum
  - `brats18`
  - `brats23`
- `crop_mode`: string enum
  - `none`
  - `center`
  - `non_empty`
- `crop_size`: list of 3 integers or `null`
- `crop_min_size`: list of 3 integers or `null`
- `clamp_mode`: string enum
  - `none`
  - `subject`
  - `dataset`
- `clamp_percentile`: list of 2 floats or `null`
- `clamp_min`: list of 4 floats or `null`
- `clamp_max`: list of 4 floats or `null`
- `norm_mode`: string enum
  - `none`
  - `min_max`
  - `subject_zscore`
  - `dataset_zscore`
- `norm_min_max_range`: list of 2 floats or `null`
- `norm_mean`: list of 4 floats or `null`
- `norm_std`: list of 4 floats or `null`
- `yes`: boolean

### Training

- `data_dir`: path
- `art_dir`: path
- `trainer`: string enum
  - `imfuse`
  - `dcseg`
- `model`: string enum
  - `imfuse`
  - `mmformer`
  - `dcseg`
  - `rfnet`
- `loss`: string enum
  - `imfuse`
  - `dcseg`
- `optimizer`: string enum
  - `radam`
  - `adamw`
  - `adam`
  - `sgd`
- `betas`: list of 2 floats or `null`
- `momentum`: float
- `scheduler`: string enum
  - `poly`
  - `cosine`
  - `step`
  - `multistep`
  - `plateau`
- `poly_total_iters`: integer or `null`
- `poly_power`: float
- `cosine_t_max`: integer or `null`
- `cosine_eta_min`: float or `null`
- `step_step_size`: integer or `null`
- `step_gamma`: float or `null`
- `multistep_milestones`: list of integers or `null`
- `multistep_gamma`: float or `null`
- `plateau_mode`: string or `null`
- `plateau_factor`: float or `null`
- `plateau_patience`: integer or `null`
- `lr`: float
- `weight_decay`: float
- `batch_size`: integer
- `num_epochs`: integer
- `num_workers`: integer
- `fp16`: boolean
- `resume`: boolean
- `pretrain`: path or `null`
- `seed`: integer or `null`
- `wandb_project`: string or `null`
- `wandb_mode`: string or `null`
- `wandb_run_name`: string or `null`
- `dataset_type`: string enum
  - `brats18`
  - `brats23`
- `split_file`: path or `null`
- `transform_kind`: string enum or `null`
  - `imfuse`
  - `dcseg`
  - `rfnet`

### Testing

- `checkpoint_path`: path
- `output_path`: path
- `data_dir`: path
- `model`: string enum
  - `imfuse`
  - `mmformer`
  - `dcseg`
  - `rfnet`
- `dataset_type`: string enum
  - `brats18`
  - `brats23`
- `num_workers`: integer
- `seed`: integer or `null`
- `split_file`: path or `null`

### Nested Override Blocks

- `custom_model_kwargs`: dictionary
- `custom_loss_kwargs`: dictionary
- `custom_trainer_kwargs`: dictionary

## Field Type Notes

### Paths

Path-like values are written as strings:

```yaml
data_dir: /work/user/preprocessed
checkpoint_path: /work/user/run/checkpoints/model_last.pth
```

### Enums

Enum-like values are written as lowercase strings:

```yaml
trainer: dcseg
optimizer: adam
dataset_type: brats23
```

### Booleans

Use YAML booleans:

```yaml
fp16: false
resume: true
yes: false
```

### Integers

Use plain integers:

```yaml
batch_size: 2
num_epochs: 500
num_workers: 8
```

### Floats

Use plain decimal values or scientific notation:

```yaml
lr: 0.0002
weight_decay: 0.0001
eps: 1.0e-7
```

## Minimal Examples

### Preprocess Example

```yaml
input_dir: /path/to/brats23
output_dir: /path/to/preprocessed
dataset_type: brats23

crop_mode: non_empty
crop_size: null
crop_min_size: [128, 128, 128]

clamp_mode: none
clamp_percentile: null
clamp_min: null
clamp_max: null

norm_mode: subject_zscore
norm_min_max_range: null
norm_mean: null
norm_std: null

yes: false
```

### Training Example

```yaml
data_dir: /path/to/preprocessed
art_dir: /path/to/run
trainer: dcseg
model: dcseg
loss: dcseg

custom_model_kwargs:
  num_cls: 4
  fusion_type: RFM

custom_trainer_kwargs:
  patch_size: 112
  transform_kind: dcseg
  train_masking_mode: random
  val_masking_mode: validation
  use_recon_loss: true
  use_reg_loss: true
  use_ana_contrastive: true
  use_mod_contrastive: true

optimizer: adam
betas: [0.9, 0.999]

scheduler: poly
poly_total_iters: 500
poly_power: 0.9

lr: 0.0002
weight_decay: 0.0001
batch_size: 1
num_epochs: 500
num_workers: 8
fp16: false
resume: false
pretrain: null
seed: 999

wandb_project: SegmentationMM
wandb_mode: online
wandb_run_name: dcseg23_training
dataset_type: brats23
```

### Testing Example

```yaml
data_dir: /path/to/preprocessed
checkpoint_path: /path/to/checkpoint.pth
output_path: /path/to/results.txt
model: dcseg

custom_model_kwargs:
  num_cls: 4
  fusion_type: RFM

dataset_type: brats23
num_workers: 8
seed: 42
```

## Nested Blocks

### `custom_model_kwargs`

Passed directly to the selected model class.

Type: dictionary with model-specific keys.

```yaml
custom_model_kwargs:
  num_cls: 4
  fusion_type: RFM
```

### `custom_loss_kwargs`

Passed directly to the selected loss class.

Type: dictionary with loss-specific keys.

```yaml
custom_loss_kwargs:
  num_classes: 4
  eps: 1.0e-7
```

### `custom_trainer_kwargs`

Passed directly to the selected trainer.

Type: dictionary with trainer-specific keys.

```yaml
custom_trainer_kwargs:
  patch_size: 112
  transform_kind: dcseg
  train_masking_mode: random
  val_masking_mode: validation
```

Common trainer-specific keys and types:

- `iter_per_epoch`: integer or `null`
- `region_fusion_start_epoch`: integer
- `patch_size`: integer
- `debug`: boolean
- `transform_kind`: string enum
- `train_masking_mode`: string enum
  - `random`
  - `validation`
  - `test`
- `val_masking_mode`: string enum
  - `random`
  - `validation`
  - `test`
- `use_recon_loss`: boolean
- `use_reg_loss`: boolean
- `use_ana_contrastive`: boolean
- `use_mod_contrastive`: boolean
- `regularization_alpha`: float
- `anatomy_contrastive_method`: string

## Nulls and Lists

Use `null` for intentionally unset values:

- `crop_size: null`
- `pretrain: null`

Use YAML lists for sequences:

- `crop_min_size: [128, 128, 128]`
- `betas: [0.9, 0.999]`
- `multistep_milestones: [100, 200, 300]`

## One File for Multiple Commands

It is normal for one config to include preprocess, train, and test fields together.

- `mimose preprocess --config ...` reads preprocess keys
- `mimose train --config ...` reads train keys
- `mimose test --config ...` reads test keys

You do not need separate files unless that is easier for your workflow.

## Starting Points

Recommended examples:

- `src/brainchmark/data/configs/imfuse_18.yaml`
- `src/brainchmark/data/configs/imfuse_23.yaml`
- `src/brainchmark/data/configs/mmformer_18.yaml`
- `src/brainchmark/data/configs/mmformer_23.yaml`
- `src/brainchmark/data/configs/dcseg_18.yaml`
- `src/brainchmark/data/configs/dcseg_23.yaml`
- `src/brainchmark/data/configs/rfnet_18.yaml`
- `src/brainchmark/data/configs/rfnet_23.yaml`

## Related Docs

- [README.md](../README.md)
- [docs/preprocessing.md](preprocessing.md)
- [docs/training.md](training.md)
- [docs/testing.md](testing.md)
- [docs/extending.md](extending.md)
