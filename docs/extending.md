# Extending BrainchMark

This document explains how to extend the active `brainchmark` package under `src/brainchmark`.

It intentionally ignores `legacy/`. That folder is reference material at best and should not be treated as the extension path for the current codebase.

## Scope

The active extension surface lives here:

- `src/brainchmark/cli.py`
- `src/brainchmark/preprocessing/`
- `src/brainchmark/models/`
- `src/brainchmark/datasets/`
- `src/brainchmark/training/`
- `src/brainchmark/gui.py`

The general flow is:

1. Add implementation code in the right package.
2. Add config/build logic next to that implementation.
3. Expose it through `brainchmark.cli`.
4. Update docs and, if needed, the GUI-compatible CLI types.

## Repo Reality

Before extending anything, keep these current facts in mind:

- Preprocessing is the cleanest and most complete extension path.
- Model config is now separated into `src/brainchmark/models/config.py`.
- Trainer infrastructure exists, but the training runtime is still in transition.
- The CLI already builds `model_config`, `optimizer_config`, and `scheduler_config`, but trainer dispatch is not yet fully wired end to end.
- The GUI introspects the Typer CLI, so CLI changes can surface automatically there if the option types are simple enough.

That means:

- extending preprocessing is straightforward
- extending models is reasonable
- extending trainers is possible, but you may also need to finish some wiring in the active training path

## Extension Map

Use this decision table:

- new crop/clamp/normalization function: `src/brainchmark/preprocessing/*.py`
- new dataset enum or dataset loader: `src/brainchmark/datasets/`
- new model config or resolution rule: `src/brainchmark/models/config.py`
- new model implementation: `src/brainchmark/models/`
- new optimizer/scheduler config behavior: `src/brainchmark/training/config.py`
- new trainer base logic or reusable trainer behavior: `src/brainchmark/training/trainers/base_trainer.py`
- new concrete trainer: `src/brainchmark/training/trainers/`
- new CLI option or command: `src/brainchmark/cli.py`
- GUI support for a new CLI annotation shape: `src/brainchmark/gui.py`

## Preprocessing

The preprocessing extension model is module-based.

Relevant files:

- `src/brainchmark/preprocessing/config.py`
- `src/brainchmark/preprocessing/cropping.py`
- `src/brainchmark/preprocessing/clamping.py`
- `src/brainchmark/preprocessing/normalization.py`
- `src/brainchmark/preprocessing/pipeline.py`

### How it works

`build_crop_config`, `build_clamp_config`, and `build_norm_config` resolve callables by name from the corresponding module.

Built-in names are backed by enums:

- `CropMode`
- `ClampMode`
- `NormMode`

But custom names are also accepted now. The builder first checks known built-ins for special validation rules, then falls back to looking up a same-named callable in the module.

### Add a custom preprocessing function

Example for normalization:

1. Add a new function in `src/brainchmark/preprocessing/normalization.py`.
2. Match the expected signature:

```python
def my_norm(images: np.ndarray, config: NormConfig) -> np.ndarray:
    ...
```

3. Call it from the CLI with:

```bash
brainchmark preprocess --norm-mode my_norm
```

The same rule applies to:

- `cropping.py`: `fn(images, seg, config) -> tuple[images, seg]`
- `clamping.py`: `fn(images, config) -> images`
- `normalization.py`: `fn(images, config) -> images`

### When to add a new enum value

Add a new enum value only if the mode should become a first-class built-in with special validation behavior.

Examples:

- a crop mode that requires `crop_size`
- a clamp mode that requires percentile bounds
- a norm mode that requires mean/std

If your function can use the existing config object without extra CLI validation rules, you can just add the callable and use its name directly.

### If you need new parameters

If your preprocessing function needs new user-facing inputs:

1. add the option to `brainchmark preprocess` in `src/brainchmark/cli.py`
2. merge it with `merge_cli_overrides(...)`
3. extend the matching config dataclass in `src/brainchmark/preprocessing/config.py`
4. validate it in the appropriate `build_*_config(...)`
5. use it in your function implementation
6. document it in `docs/preprocessing.md`

## Models

Relevant files:

- `src/brainchmark/models/config.py`
- `src/brainchmark/models/__init__.py`
- `src/brainchmark/models/IMFuse.py`

### Current model architecture

Model selection is configured through:

- `ModelKind`
- `ModelConfig`
- `build_model_config(...)`

in `src/brainchmark/models/config.py`.

The CLI exposes `--model` and `--custom-model-kwargs`, and the training CLI builds a `model_config` object from them.

### Add a new model

1. Create a new module under `src/brainchmark/models/`.
2. Implement the model class there.
3. Update `src/brainchmark/models/config.py` so `_resolve_model(...)` can find it.
4. If the model should be a built-in CLI choice, add it to `ModelKind`.
5. If it needs custom constructor kwargs, pass them through `--custom-model-kwargs`.

### Important caveat

`src/brainchmark/models/__init__.py` still eagerly imports every Python file in the package. That means package-level model imports are heavy.

For extension work:

- prefer resolving the exact module or a registry in `models/config.py`
- do not assume importing `brainchmark.models` is cheap

If model extension becomes common, the clean next step is a `MODEL_REGISTRY` instead of package-wide eager imports.

## Datasets

Relevant files:

- `src/brainchmark/datasets/config.py`
- `src/brainchmark/datasets/imfuse.py`
- `src/brainchmark/data/splits/`

### Add a new dataset type

1. Add the enum value to `DatasetType` in `src/brainchmark/datasets/config.py`.
2. Extend any dataset-specific path logic that depends on dataset type:
   - preprocessing input discovery in `src/brainchmark/preprocessing/pipeline.py`
   - CLI help text if needed
3. Add or adapt split files under `src/brainchmark/data/splits/`.
4. If you need a new dataset loader, add a new dataset module under `src/brainchmark/datasets/`.

### Caution

`src/brainchmark/datasets/__init__.py` currently imports `IMFuseDataset` eagerly. Like models, that makes package imports heavier than they need to be.

For new work, prefer direct module imports in runtime code unless you intentionally want package-level export behavior.

## Optimizers and Schedulers

Relevant file:

- `src/brainchmark/training/config.py`

This file already contains the active pattern for configurable runtime objects:

- `OptimizerKind` + `OptimizerConfig` + `build_optimizer_config(...)`
- `SchedulerKind` + `SchedulerConfig` + `build_scheduler_config(...)`

### Add a new optimizer

1. Add a new enum value to `OptimizerKind`.
2. Add the implementation to `OPTIMIZER_DICT`.
3. Extend `build_optimizer_config(...)` validation and config construction.
4. Expose it in the CLI if the type annotation needs to change.

### Add a new scheduler

1. Add a new enum value to `SchedulerKind` if it is built-in.
2. Extend `SCHEDULER_DICT` if applicable.
3. Add validation and config generation in `build_scheduler_config(...)`.

## Trainers

Relevant files:

- `src/brainchmark/training/trainers/base_trainer.py`
- `src/brainchmark/training/trainers/imfuse.py`
- `src/brainchmark/training/trainers/__init__.py`
- `src/brainchmark/training/config.py`
- `src/brainchmark/cli.py`

### Current active trainer story

There are two parallel realities right now:

- `BaseTrainer` is the active generic abstraction being introduced
- `IMFuseTrainer` still contains an older, more self-contained runtime path

So when you add a trainer, decide first whether you are:

- extending the new generic path
- or porting an old trainer into the new path

The right long-term answer is the first one.

### Add a new trainer class

1. Create a file under `src/brainchmark/training/trainers/`.
2. Subclass `BaseTrainer`.
3. Implement:
   - `train_epoch(self, epoch)`
   - `val_epoch(self, epoch)`
4. Build and assign:
   - `self.model`
   - `self.optimizer`
   - `self.scheduler`
   - `self.train_loader`
   - `self.val_loader`
5. If you use DDP, call `self.wrap_model_for_distributed()` after model creation.

### What BaseTrainer already gives you

`BaseTrainer` already handles:

- output/checkpoint directory setup
- checkpoint save/load
- best-checkpoint tracking by validation loss
- DDP process-group setup and teardown
- per-rank device selection
- rank-aware checkpointing and logging
- metric reduction across ranks
- sampler `set_epoch(...)`

### Current trainer integration gap

Be explicit about this: the active CLI currently builds configs, but trainer dispatch is not yet finished end to end.

That means adding a trainer class alone is not enough. You will likely also need to:

1. add the trainer kind to the CLI enum
2. add any trainer-specific config plumbing
3. wire trainer instantiation into `brainchmark train`

If you want a simple extensible architecture, use:

- a trainer registry in `src/brainchmark/training/trainers/__init__.py`
- trainer name resolution in the CLI/runtime layer

## CLI

Relevant file:

- `src/brainchmark/cli.py`

The CLI is the user-facing extension surface and the source of truth for:

- Typer command signatures
- help panels
- YAML/CLI override merging
- config builder calls

### Add a new CLI option

1. Add the option to the command signature in `cli.py`.
2. Put it in the appropriate `rich_help_panel`.
3. Merge it through `merge_cli_overrides(...)`.
4. Feed it into the right config builder or runtime object.

### Lightweight enums in the CLI

The CLI currently duplicates lightweight enums like:

- `TrainerKind`
- `ModelKind`
- `OptimizerKind`
- `SchedulerKind`

This is intentional. It keeps CLI import time lighter and avoids pulling in the full runtime stack just to generate help or shell completions.

When adding a new built-in CLI choice, remember to update:

- the lightweight CLI enum in `src/brainchmark/cli.py`
- the real runtime enum/config logic in the owning module

## GUI

Relevant file:

- `src/brainchmark/gui.py`

The GUI introspects the Typer app and creates widgets from command signatures and option metadata.

This is useful when extending the CLI, because many GUI updates happen automatically if your option type is already supported.

### Good option types for automatic GUI support

The current GUI handles these shapes reasonably well:

- `Path`
- `bool`
- enums
- tuples of ints/floats
- lists of ints/floats
- simple strings

If you add a new option type that the GUI does not understand, you will need to extend the field generation logic in `gui.py`.

## Config and YAML

Relevant file:

- `src/brainchmark/utils/cli_overrides.py`

YAML loading is intentionally simple:

- top-level mapping only
- lists are converted to tuples
- CLI values override YAML values if not `None`

That means if you add new structured config:

- keep the YAML schema flat unless you also expand the loader logic
- or accept nested dicts and consume them manually in your runtime layer

## Recommended Extension Workflow

For any new feature:

1. Add the implementation in the owning module.
2. Add or extend config objects/builders.
3. Expose the feature in `cli.py`.
4. Make sure the GUI can still introspect the CLI signature.
5. Add or update docs in `docs/`.
6. Run at least:

```bash
python -m py_compile src/brainchmark/cli.py
python -m py_compile src/brainchmark/preprocessing/config.py
python -m py_compile src/brainchmark/models/config.py
python -m py_compile src/brainchmark/training/config.py
python -m py_compile src/brainchmark/training/trainers/base_trainer.py
```

## Minimal Examples

### New preprocessing normalization

Add to `src/brainchmark/preprocessing/normalization.py`:

```python
def percentile_unit(images, config):
    ...
```

Run:

```bash
brainchmark preprocess --norm-mode percentile_unit
```

### New model

Add:

- `src/brainchmark/models/my_model.py`

Then update:

- `src/brainchmark/models/config.py`

to resolve `my_model`, and optionally add it to `ModelKind`.

### New trainer

Add:

- `src/brainchmark/training/trainers/my_trainer.py`

Implement a `BaseTrainer` subclass, then wire it into:

- CLI trainer choice enum
- trainer registry or trainer dispatch logic

## What Not To Do

- Do not extend `legacy/` for current BrainchMark features.
- Do not hide heavy runtime imports in the CLI path unless you really have to.
- Do not add new user-facing config without updating docs.
- Do not rely on package `__init__.py` eager imports as your long-term extension mechanism.

## Future Cleanup That Would Make Extension Easier

These are the next architectural improvements worth making:

- replace eager model imports with a model registry
- replace eager dataset imports with a dataset registry
- finish trainer dispatch in the active training runtime
- move CLI lightweight enums to a dedicated low-import metadata module
- document the final YAML schema for training once the runtime stabilizes
