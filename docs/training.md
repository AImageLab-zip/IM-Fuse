# Training

`brainchmark train` launches config-driven training for the active BrainchMark package.

At the moment, the implemented training stack is centered on `IMFuseTrainer`, which is used for both:

- `imfuse`
- `mmformer`

The legacy model logic differs, but the active BrainchMark runtime currently shares the same dataset, loss, checkpointing, and trainer infrastructure across those models.

## Basic Usage

Run from YAML:

```bash
brainchmark train --config src/brainchmark/data/configs/imfuse_23.yaml
```

Run from CLI:

```bash
brainchmark train \
  --data-dir /path/to/preprocessed \
  --art-dir /path/to/run \
  --trainer imfuse \
  --model mmformer \
  --loss imfuse \
  --optimizer adam \
  --scheduler poly \
  --lr 2e-4 \
  --num-epochs 1000 \
  --dataset-type brats23
```

CLI values override YAML values when both are provided.

## Required Inputs

The training command currently requires:

- `data_dir`
- `art_dir`
- `trainer`
- `optimizer`
- `num_epochs`

In practice, you should also set these explicitly for a real run, even though some have defaults:

- `model`
- `dataset_type`

`split_file` is not strictly required on the CLI because it resolves to the packaged default split file when omitted.

## Reference Configs

The repo currently ships these reference training configs:

- `src/brainchmark/data/configs/imfuse_18.yaml`
- `src/brainchmark/data/configs/imfuse_23.yaml`
- `src/brainchmark/data/configs/mmformer_18.yaml`
- `src/brainchmark/data/configs/mmformer_23.yaml`

They are combined reference files that include preprocess, train, and test sections/fields. The train command reads the training-relevant keys and ignores the rest.

## Core Training Fields

Top-level training fields:

- `data_dir`
- `art_dir`
- `trainer`
- `model`
- `loss`
- `optimizer`
- `scheduler`
- `lr`
- `weight_decay`
- `batch_size`
- `num_epochs`
- `num_workers`
- `fp16`
- `resume`
- `pretrain`
- `seed`
- `wandb_project`
- `wandb_mode`
- `wandb_run_name`
- `dataset_type`

Structured extension points:

- `custom_model_kwargs`
- `custom_loss_kwargs`
- `custom_trainer_kwargs`

## Current Built-in Choices

Trainer values:

- `imfuse`

Model values:

- `imfuse`
- `mmformer`

Loss values:

- `imfuse`

Optimizer values:

- `radam`
- `adamw`
- `adam`
- `sgd`

Scheduler values:

- `poly`
- `cosine`
- `step`
- `multistep`
- `plateau`

Transform manager values:

- `imfuse`

## Trainer-Specific Runtime Options

The active IMFuse-style trainer currently consumes trainer kwargs such as:

- `iter_per_epoch`
- `region_fusion_start_epoch`
- `patch_size`
- `debug`
- `transform_kind`
- `train_masking_mode`
- `val_masking_mode`
- `split_file`

Example:

```yaml
custom_trainer_kwargs:
  iter_per_epoch: null
  region_fusion_start_epoch: 0
  patch_size: 128
  debug: false
  transform_kind: imfuse
  train_masking_mode: random
  val_masking_mode: validation
```

Current masking defaults:

- training: `random`
- validation: `validation`

## Model Kwargs

`custom_model_kwargs` are passed directly to the selected model class.

Current examples:

For `imfuse`:

```yaml
custom_model_kwargs:
  interleaved_tokenization: false
  mamba_skip: false
  num_cls: 4
```

For `mmformer`:

```yaml
custom_model_kwargs:
  num_cls: 4
```

You can also override these from the CLI:

```bash
brainchmark train \
  --model imfuse \
  --custom-model-kwargs interleaved_tokenization=True \
  --custom-model-kwargs mamba_skip=True
```

## WandB

Training supports:

- `wandb_project`
- `wandb_mode`
- `wandb_run_name`

`wandb_run_name` is optional and currently defaults to `training`.

Example:

```bash
brainchmark train \
  --config src/brainchmark/data/configs/mmformer_23.yaml \
  --wandb-run-name mmformer-ablation-01
```

The training launch panel also prints the resolved run name.

## Distributed Training

The CLI supports DDP relaunch through:

- `--distributed`
- `--nproc-per-node`

When `--distributed` is used outside an existing `torchrun` launch, BrainchMark relaunches itself through `torchrun`.

## Runtime Behavior

The active trainer currently provides:

- launch summary panel
- checkpoint directory setup
- checkpoint save/load
- optional pretrain loading
- WandB initialization/logging
- DDP wrapping
- Rich progress bars for train and validation
- VRAM progress-bar field
- CUDA OOM normalization into a clean CLI error

## Legacy Alignment Notes

The current configs try to mirror legacy defaults where they materially differ.

Examples:

- `imfuse_23.yaml`
  - `model: imfuse`
  - `optimizer: radam`
- `mmformer_23.yaml`
  - `model: mmformer`
  - `optimizer: adam`
  - `scheduler: poly`
  - `lr: 2e-4`
  - `weight_decay: 1e-4`

The loss implementation is shared because the legacy IMFuse and mmFormer loss files are effectively identical.

## Example YAML

```yaml
data_dir: /path/to/preprocessed
art_dir: /path/to/run
trainer: imfuse
model: mmformer
loss: imfuse

custom_model_kwargs:
  num_cls: 4

custom_trainer_kwargs:
  patch_size: 128
  transform_kind: imfuse
  train_masking_mode: random
  val_masking_mode: validation

optimizer: adam
betas: [0.9, 0.999]

scheduler: poly
poly_total_iters: 1000
poly_power: 0.9

lr: 0.0002
weight_decay: 0.0001
batch_size: 1
num_epochs: 1000
num_workers: 8
fp16: false
resume: false
pretrain: null
seed: 999

wandb_project: SegmentationMM
wandb_mode: online
wandb_run_name: training
dataset_type: brats23
```

## Related Docs

- [README.md](../README.md)
- [docs/testing.md](testing.md)
- [docs/extending.md](extending.md)
