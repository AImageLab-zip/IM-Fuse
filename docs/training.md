# Training

`brainchmark train` launches model training from CLI flags and/or a YAML config file.

This document is intentionally incomplete. It covers the current CLI surface and the parts that are already wired, but some training internals are still in flux.

## Current Status

Training support exists, but the configuration layer and the trainer implementation are not fully aligned yet.

At the moment:

- the CLI accepts optimizer and scheduler options
- YAML config values can be merged with CLI overrides
- optimizer and scheduler inputs are validated in the CLI/config layer
- the runtime trainer does not yet consume every advanced optimizer and scheduler setting end-to-end

If the same option is provided in both places, the CLI value should override the YAML value.

## Basic Usage

Run from YAML:

```bash
brainchmark train --config path/to/training.yaml
```

Run from CLI:

```bash
brainchmark train \
  --input-dir /path/to/preprocessed \
  --output-dir /path/to/run \
  --trainer imfuse \
  --dataname BRATS2018 \
  --optimizer radam \
  --scheduler poly \
  --lr 2e-4 \
  --num-epochs 1000
```

## Required Inputs

These values are currently required by the CLI layer:

- `input_dir`
- `output_dir`
- `trainer`
- `optimizer`
- `num_epochs`

In practice, `dataname` should also be treated as required for a real run, even though the CLI currently falls back to `BRATS2018`.

## YAML Example

```yaml
input_dir: /path/to/preprocessed
output_dir: /path/to/run

trainer: imfuse
dataname: BRATS2018

optimizer: radam
betas: [0.9, 0.999]
momentum: 0.9

scheduler: poly
poly_total_iters: 1000
poly_power: 0.9

lr: 0.0002
weight_decay: 0.00003
batch_size: 1
num_epochs: 1000
num_workers: 8
region_fusion_start_epoch: 0
seed: 999

train_transforms: "Compose([...])"
test_transforms: "Compose([...])"

debug: false
interleaved_tokenization: false
mamba_skip: false
first_skip: false

wandb_project: SegmentationMM
wandb_mode: online
```

## Optimizer Options

Supported optimizer values:

- `radam`
- `adamw`
- `adam`
- `sgd`

Relevant fields:

- `optimizer`
- `lr`
- `weight_decay`
- `betas`
- `momentum`

Notes:

- `betas` are used for `radam`, `adamw`, and `adam`
- `momentum` is used for `sgd`
- the CLI/config layer validates these combinations before training starts

## Scheduler Options

Supported built-in scheduler values:

- `poly`
- `cosine`
- `step`
- `multistep`
- `plateau`

Relevant fields:

- `scheduler`
- `poly_total_iters`
- `poly_power`
- `cosine_t_max`
- `cosine_eta_min`
- `step_step_size`
- `step_gamma`
- `multistep_milestones`
- `multistep_gamma`
- `plateau_mode`
- `plateau_factor`
- `plateau_patience`

Custom scheduler support is partially wired:

- custom scheduler names can be resolved from `brainchmark.training.scheduling`
- YAML and CLI can both provide `custom_scheduler_kwargs`
- runtime trainer integration is still incomplete

Example:

```yaml
scheduler: my_custom_scheduler
custom_scheduler_kwargs:
  warmup_steps: 100
  decay: 0.95
```

CLI form:

```bash
brainchmark train \
  --scheduler my_custom_scheduler \
  --custom-scheduler-kwarg warmup_steps=100 \
  --custom-scheduler-kwarg decay=0.95
```

## Other Training Options

The train command also accepts:

- `resume`
- `pretrain`
- `device`
- `batch_size`
- `num_workers`
- `region_fusion_start_epoch`
- `seed`
- `train_transforms`
- `test_transforms`
- `debug`
- `interleaved_tokenization`
- `mamba_skip`
- `first_skip`
- `wandb_project`
- `wandb_mode`

## Known Gaps

These parts still need cleanup or fuller documentation:

- exact end-to-end optimizer instantiation from `OptimizerConfig`
- exact end-to-end scheduler instantiation from `SchedulerConfig`
- expected YAML schema for every trainer-specific field
- resume/pretrain behavior details
- distributed training behavior
- metrics, checkpoints, and validation schedule
- transform expression format

## Next Steps

This doc should later be extended with:

- a complete YAML reference
- a full list of train CLI flags
- working examples for each scheduler kind
- trainer behavior details
- troubleshooting notes
