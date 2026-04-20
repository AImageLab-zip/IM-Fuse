# Extending BrainchMark

This document covers the active `brainchmark` package under `src/brainchmark`.

It intentionally does not treat `legacy/` as the extension surface. Legacy code is reference material; the maintained package API is under `src/brainchmark`.

## Active Extension Surface

Relevant areas:

- `src/brainchmark/cli.py`
- `src/brainchmark/preprocessing/`
- `src/brainchmark/models/`
- `src/brainchmark/datasets/`
- `src/brainchmark/training/`
- `src/brainchmark/testing/`
- `src/brainchmark/gui.py`

Typical flow:

1. Add the implementation in the correct package area.
2. Wire config/build logic if needed.
3. Expose it through the CLI or config resolver.
4. Update docs and example configs.

## Repo Reality

Current practical facts:

- preprocessing is mature and easy to extend
- the active trainer stack is usable, not just scaffold
- the model resolver is dynamic and discovers modules under `src/brainchmark/models`
- GUI behavior is driven by the Typer CLI, so simple CLI option changes usually surface there automatically
- testing is implemented and depends on models exposing `predict(images, mask)`

## Preprocessing

Relevant files:

- `src/brainchmark/preprocessing/config.py`
- `src/brainchmark/preprocessing/cropping.py`
- `src/brainchmark/preprocessing/clamping.py`
- `src/brainchmark/preprocessing/normalization.py`
- `src/brainchmark/preprocessing/pipeline.py`

### How it works

`build_crop_config`, `build_clamp_config`, and `build_norm_config` resolve callables by name.

Built-in names are backed by enums:

- `CropMode`
- `ClampMode`
- `NormMode`

Custom callable names are also allowed when they match functions in the relevant preprocessing module.

### Add a custom preprocessing function

Example for normalization:

1. Add a function to `src/brainchmark/preprocessing/normalization.py`.
2. Match the expected signature:

```python
def my_norm(images: np.ndarray, config: NormConfig) -> np.ndarray:
    ...
```

3. Use it from the CLI:

```bash
brainchmark preprocess --norm-mode my_norm
```

Expected signatures:

- `cropping.py`: `fn(images, seg, config) -> tuple[images, seg]`
- `clamping.py`: `fn(images, config) -> images`
- `normalization.py`: `fn(images, config) -> images`

### If you need new preprocessing parameters

1. Add the option to `brainchmark preprocess` in `src/brainchmark/cli.py`.
2. Merge it through `merge_cli_overrides(...)`.
3. Extend the relevant config dataclass in `src/brainchmark/preprocessing/config.py`.
4. Validate it in the corresponding `build_*_config(...)`.
5. Use it in the implementation.
6. Update `docs/preprocessing.md`.

## Models

Relevant files:

- `src/brainchmark/models/config.py`
- `src/brainchmark/models/abstract_model.py`
- `src/brainchmark/models/IMFuse.py`
- `src/brainchmark/models/mmformer.py`

### Current model resolution

Model selection is built from:

- `ModelKind`
- `ModelConfig`
- `build_model_config(...)`

The resolver scans Python modules under `src/brainchmark/models/` and matches callables by normalized name.

### Add a new model

1. Create a new module under `src/brainchmark/models/`.
2. Implement a model class there.
3. Make sure it inherits from `AbstractModel`.
4. Implement:
   - `forward(...)`
   - `predict(images, mask)`
5. If you want the model to be a first-class CLI enum value, add it to `ModelKind`.
6. Add or update a reference config under `src/brainchmark/data/configs/`.

### Model contract

For the active trainer/runtime, a model should:

- support training-time forward with `(images, mask)`
- support inference-time `predict(images, mask)`

For the IMFuse-style trainer path specifically, the training forward is expected to return:

- `fuse_pred`
- `sep_preds`
- `prm_preds`

That is why both active models currently conform to the IMFuse-style loss/trainer interface.

## Datasets

Relevant files:

- `src/brainchmark/datasets/base.py`
- `src/brainchmark/datasets/imfuse.py`
- `src/brainchmark/datasets/config.py`
- `src/brainchmark/data/splits/`

### Add a new dataset type

1. Add the enum value to `DatasetType`.
2. Extend dataset-specific path logic where needed:
   - preprocessing discovery in `src/brainchmark/preprocessing/pipeline.py`
   - split loading if needed
3. Add split files under `src/brainchmark/data/splits/`.
4. Add a new dataset loader module if the sample format differs.

### Masking support

The active IMFuse dataset path already supports configurable masking behavior through:

- `MaskingMode`
- predefined mask patterns
- dataset-level mask resolution

If you extend missing-modality behavior, the relevant places are:

- `src/brainchmark/datasets/base.py`
- `src/brainchmark/datasets/imfuse.py`
- trainer kwargs that select train/val masking modes

## Optimizers and Schedulers

Relevant file:

- `src/brainchmark/training/config.py`

This file contains the active pattern for configurable runtime objects:

- `OptimizerKind` + `OptimizerConfig` + `build_optimizer_config(...)`
- `SchedulerKind` + `SchedulerConfig` + `build_scheduler_config(...)`

### Add a new optimizer

1. Add a new enum value to `OptimizerKind`.
2. Add the implementation to `OPTIMIZER_DICT`.
3. Extend validation/config construction in `build_optimizer_config(...)`.
4. Update docs and example configs.

### Add a new scheduler

1. Add a new enum value to `SchedulerKind` if it is a built-in.
2. Extend `SCHEDULER_DICT`.
3. Extend validation/config construction in `build_scheduler_config(...)`.
4. Update docs and example configs.

## Trainers

Relevant files:

- `src/brainchmark/training/trainers/abstract_trainer.py`
- `src/brainchmark/training/trainers/base_trainer.py`
- `src/brainchmark/training/trainers/imfuse.py`
- `src/brainchmark/cli.py`

### Current trainer story

The active runtime is based on:

- `BaseTrainer` for generic runtime behavior
- `IMFuseTrainer` for the current concrete segmentation workflow

`IMFuseTrainer` is also currently reused for `mmformer`, because the active loss/data/runtime path still follows the IMFuse-style interface.

### Add a new trainer

1. Create a file under `src/brainchmark/training/trainers/`.
2. Subclass `BaseTrainer`.
3. Implement:
   - `train_epoch(self, epoch)`
   - `val_epoch(self, epoch)`
   - `build_datasets(self)`
4. Expose the trainer through `_resolve_trainer_class(...)` in `src/brainchmark/cli.py`.
5. Add a matching `TrainerKind` enum value if it should be a built-in choice.

### What BaseTrainer already provides

`BaseTrainer` currently handles:

- output/checkpoint directory setup
- checkpoint save/load
- DDP setup/cleanup
- rank-aware behavior
- WandB init/logging
- launch summary printing
- OOM normalization into clean CLI errors

## Testing

Relevant files:

- `src/brainchmark/testing/pipeline.py`
- `src/brainchmark/cli.py`

Testing currently assumes:

- CUDA is available
- the selected model inherits from `AbstractModel`
- the selected model implements `predict(images, mask)`

If you add a model and want `brainchmark test` to work, `predict(...)` is mandatory.

## CLI and GUI

The GUI is generated from the Typer CLI signature surface.

That means:

- new simple options in `cli.py` usually appear naturally in the GUI
- exotic types or unsupported annotations may need GUI follow-up work in `src/brainchmark/gui.py`

When you add new CLI options:

1. wire them into `merge_cli_overrides(...)`
2. document them
3. update example configs when appropriate

## Reference Configs

Reference configs live in:

- `src/brainchmark/data/configs/`

Current examples include:

- `imfuse_18.yaml`
- `imfuse_23.yaml`
- `mmformer_18.yaml`
- `mmformer_23.yaml`
- `preprocessing.yaml`

Whenever you add a meaningful built-in feature, update at least one reference config.
