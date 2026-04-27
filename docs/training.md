# Training

`brainchmark train` launches config-driven training for the active BrainchMark package.

The active training stack currently includes:

- `IMFuseTrainer` for:
  - `imfuse`
  - `mmformer`
  - `rfnet`
- `DCSegTrainer` for:
  - `dcseg`

The runtime still shares common checkpointing, DDP, WandB, and Rich progress infrastructure across trainers, but the DC-Seg path now has its own trainer and transform wiring.

## Basic Usage

Run from YAML:

```bash
brainchmark train --config imfuse_23.yaml
```
The CLI automatically resolves and autocompletes config files found in src/brainchmark/data/config. Absolute or relative paths for custom yaml files are supported too.
Run from CLI:

```bash
brainchmark train \
  --data-dir /path/to/preprocessed \
  --art-dir /path/to/artifacts_dir \
  --trainer dcseg \
  --model dcseg \
  --loss dcseg \
  --optimizer adam \
  --scheduler poly \
  --lr 2e-4 \
  --num-epochs 500 \
  --dataset-type brats23
```

CLI values override YAML values when both are provided.

For the overall BrainchMark YAML format, see [docs/yaml-config.md](yaml-config.md).
For very detailed extension notes on trainers, models, and runtime wiring, see [docs/components/README.md](components/README.md).


## Reference Configs

The repo currently ships these reference training configs:

- `src/brainchmark/data/configs/imfuse_18.yaml`
- `src/brainchmark/data/configs/imfuse_23.yaml`
- `src/brainchmark/data/configs/mmformer_18.yaml`
- `src/brainchmark/data/configs/mmformer_23.yaml`
- `src/brainchmark/data/configs/dcseg_18.yaml`
- `src/brainchmark/data/configs/dcseg_23.yaml`
- `src/brainchmark/data/configs/rfnet_18.yaml`
- `src/brainchmark/data/configs/rfnet_23.yaml`

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
- `dcseg`

Model values:

- `imfuse`
- `mmformer`
- `dcseg`
- `rfnet`

Loss values:

- `imfuse`
- `dcseg`

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
- `dcseg`
- `rfnet`

## Trainer-Specific Runtime Options

The active trainers consume overlapping but not identical trainer kwargs.

Common examples:

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

### DC-Seg-specific notes

The packaged DC-Seg configs are tuned to follow the maintained DC-Seg port rather than the earlier generic IMFuse defaults.

Current defaults in the shipped DC-Seg configs include:

- `trainer: dcseg`
- `model: dcseg`
- `loss: dcseg`
- `transform_kind: dcseg`
- `patch_size: 112`
- `poly_total_iters: 500`
- `fp16: false`
- `use_recon_loss: true`
- `use_reg_loss: true`
- `use_ana_contrastive: true`
- `use_mod_contrastive: true`

The DC-Seg inference path uses its own sliding-window `predict(...)` implementation inside the model, so training crop size and inference window size are not the same thing.

### RFNet-specific notes

The packaged RFNet configs reuse `IMFuseTrainer` and `IMFuseLoss`, because RFNet
returns the same fused, separate, and PRM prediction tuple during training.
RFNet uses `transform_kind: rfnet` and `patch_size: 80`, matching the legacy
RFNet crop and sliding-window size.

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

For `dcseg`:

```yaml
custom_model_kwargs:
  num_cls: 4
  fusion_type: RFM
```

For `rfnet`:

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

If W&B logging is enabled but there is no active login in the terminal, BrainchMark now fails with a clean CLI message telling you to run `wandb login` or disable W&B with `--wandb-mode disabled`.

## Distributed Training

Distributed relaunch can be configured either in YAML or from the CLI.

Runtime fields:

- `distributed`
- `nproc_per_node`

CLI flags:

- `--distributed`
- `--no-distributed`
- `--nproc-per-node`

When distributed mode is enabled outside an existing `torchrun` launch, BrainchMark relaunches itself through `torchrun`.


## Example YAML

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

distributed: true
nproc_per_node: 4

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
wandb_run_name: training
dataset_type: brats23
```

## Related Docs

- [README.md](../README.md)
- [docs/testing.md](testing.md)
- [docs/extending.md](extending.md)
