# Extending BrainchMark

This document gives a high-level map of how to extend the active `brainchmark` package under `src/brainchmark`.

The maintained extension surface is `src/brainchmark`. The `legacy/` directories are useful for reference and comparison, but they are not the package API you should extend.

If you need field-by-field or component-by-component instructions, use the detailed guides in [docs/components/README.md](components/README.md).

## Extension Flow

Most changes follow the same path:

1. add or modify the implementation in `src/brainchmark/...`
2. wire the new behavior into config/build logic
3. expose it through CLI and YAML if users need to select it
4. update reference configs under `src/brainchmark/data/configs/`
5. update the relevant docs

If your change introduces or changes YAML fields, start with [docs/yaml-config.md](yaml-config.md).

## Repo Areas

These are the main places you will touch:

- [src/brainchmark/cli.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/cli.py): CLI entrypoints, option definitions, command wiring
- [src/brainchmark/gui.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/gui.py): GUI surface generated from the CLI
- [src/brainchmark/enums.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/enums.py): shared enum choices exposed across config and CLI
- [src/brainchmark/utils/cli_overrides.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/utils/cli_overrides.py): CLI-over-YAML merge behavior
- [src/brainchmark/data/config_templates/](/homes/ocarpentiero/IM-Fuse/src/brainchmark/data/config_templates): shipped config templates copied by `brainchmark setup`
- [src/brainchmark/data/configs/](/homes/ocarpentiero/IM-Fuse/src/brainchmark/data/configs): local reference configs used by the package
- [src/brainchmark/data/splits/split.json](/homes/ocarpentiero/IM-Fuse/src/brainchmark/data/splits/split.json): packaged dataset split definition

## Preprocessing

The preprocessing stack lives in:

- [src/brainchmark/preprocessing/config.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/preprocessing/config.py)
- [src/brainchmark/preprocessing/cropping.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/preprocessing/cropping.py)
- [src/brainchmark/preprocessing/clamping.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/preprocessing/clamping.py)
- [src/brainchmark/preprocessing/normalization.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/preprocessing/normalization.py)
- [src/brainchmark/preprocessing/pipeline.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/preprocessing/pipeline.py)

This is where you extend:

- crop/clamp/normalization modes
- dataset discovery for preprocessing
- per-case pipeline behavior

Typical file route:

1. add the implementation in one of `cropping.py`, `clamping.py`, or `normalization.py`
2. validate and resolve it in `config.py`
3. make sure `pipeline.py` can call it correctly
4. expose new options from `cli.py` if needed

Detailed guide: [docs/components/preprocessing.md](components/preprocessing.md)

## Models

The model integration layer lives in:

- [src/brainchmark/models/abstract_model.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/models/abstract_model.py)
- [src/brainchmark/models/config.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/models/config.py)
- [src/brainchmark/models/IMFuse.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/models/IMFuse.py)
- [src/brainchmark/models/mmformer.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/models/mmformer.py)
- [src/brainchmark/models/dcseg.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/models/dcseg.py)

This is where you extend:

- new model families
- model selection logic
- inference contract through `predict(images, mask)`

Typical file route:

1. add a new module under `src/brainchmark/models/`
2. inherit from `AbstractModel`
3. wire selection through `models/config.py`
4. add enum support in `enums.py` if it should be a first-class built-in choice
5. make sure the chosen trainer understands the model output structure

Detailed guide: [docs/components/models.md](components/models.md)

## Losses

The loss layer lives in:

- [src/brainchmark/losses/config.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/losses/config.py)
- [src/brainchmark/losses/imfuse.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/losses/imfuse.py)
- [src/brainchmark/losses/dcseg.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/losses/dcseg.py)
- [src/brainchmark/losses/__init__.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/losses/__init__.py)

This is where you extend:

- new loss classes
- loss selection logic
- loss kwargs exposed from YAML or CLI

Typical file route:

1. implement the loss module under `src/brainchmark/losses/`
2. wire it through `losses/config.py`
3. add enum support in `enums.py` if needed
4. make sure the trainer passes the right tensors to it

The trainer/loss contract matters more than the loss file alone. If the model outputs or batch structure differ, you usually need trainer changes too.

## Datasets

The dataset layer lives in:

- [src/brainchmark/datasets/base.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/datasets/base.py)
- [src/brainchmark/datasets/imfuse.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/datasets/imfuse.py)
- [src/brainchmark/datasets/config.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/datasets/config.py)
- [src/brainchmark/data/splits/split.json](/homes/ocarpentiero/IM-Fuse/src/brainchmark/data/splits/split.json)

This is where you extend:

- sample loading
- masking behavior
- new dataset types
- split handling

Typical file route:

1. add enum support in `enums.py`
2. update dataset config or dataset classes
3. update preprocessing and testing pipeline assumptions if the sample format changes
4. add or update split files under `src/brainchmark/data/splits/`

Detailed guide: [docs/components/datasets.md](components/datasets.md)

## Training Runtime

The shared runtime configuration lives in:

- [src/brainchmark/training/config.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/training/config.py)
- [src/brainchmark/enums.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/enums.py)

This is where you extend:

- optimizers
- schedulers
- runtime object builders
- common train-time config validation

Typical file route:

1. add the enum in `enums.py`
2. register the implementation in `training/config.py`
3. validate new kwargs there
4. ensure the trainer stepping behavior matches the new scheduler or optimizer semantics

Detailed guide: [docs/components/runtime-config.md](components/runtime-config.md)

## Trainers and Transforms

The trainer stack lives in:

- [src/brainchmark/training/trainers/abstract_trainer.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/training/trainers/abstract_trainer.py)
- [src/brainchmark/training/trainers/base_trainer.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/training/trainers/base_trainer.py)
- [src/brainchmark/training/trainers/imfuse.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/training/trainers/imfuse.py)
- [src/brainchmark/training/trainers/dcseg.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/training/trainers/dcseg.py)
- [src/brainchmark/training/transforms/base_transforms.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/training/transforms/base_transforms.py)
- [src/brainchmark/training/transforms/imfuse.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/training/transforms/imfuse.py)
- [src/brainchmark/training/transforms/__init__.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/training/transforms/__init__.py)

This is where you extend:

- training loops
- validation behavior
- data augmentation and paired transforms
- family-specific runtime logic

Typical file route:

1. add or modify a concrete trainer under `training/trainers/`
2. wire trainer selection from `cli.py`
3. update transform managers if the trainer needs a different augmentation path
4. ensure the selected model and loss obey the trainer contract

Detailed guide: [docs/components/trainers.md](components/trainers.md)

## Testing

The evaluation path lives in:

- [src/brainchmark/testing/pipeline.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/testing/pipeline.py)
- [src/brainchmark/models/abstract_model.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/models/abstract_model.py)
- [src/brainchmark/cli.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/cli.py)

This is where you extend:

- checkpoint evaluation behavior
- result formatting
- model-specific inference integration

The critical contract is that test-time models must support `predict(images, mask)`. If a model only works through `forward(...)`, `brainchmark test` will not be enough on its own.

Detailed guide: [docs/components/testing.md](components/testing.md)

## CLI and GUI

The user-facing entrypoints live in:

- [src/brainchmark/cli.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/cli.py)
- [src/brainchmark/gui.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/gui.py)
- [src/brainchmark/utils/cli_overrides.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/utils/cli_overrides.py)
- [src/brainchmark/utils/cli_utils.py](/homes/ocarpentiero/IM-Fuse/src/brainchmark/utils/cli_utils.py)

This is where you extend:

- new command options
- command-level validation
- startup UX
- GUI exposure of CLI-backed features

Typical file route:

1. add the option or command in `cli.py`
2. merge CLI-over-YAML behavior in `utils/cli_overrides.py`
3. adjust helper formatting in `utils/cli_utils.py` if needed
4. verify the GUI still renders the new option correctly in `gui.py`

Detailed guide: [docs/components/cli-gui.md](components/cli-gui.md)

## Configs and Templates

The packaged reference configs live in:

- [src/brainchmark/data/config_templates/](/homes/ocarpentiero/IM-Fuse/src/brainchmark/data/config_templates)
- [src/brainchmark/data/configs/](/homes/ocarpentiero/IM-Fuse/src/brainchmark/data/configs)

When you add a new built-in model, trainer, loss, or workflow option, update the shipped configs too. Otherwise the code may be technically wired but still hard to discover or use correctly.

## Practical Advice

- Keep generic runtime logic in shared config or `BaseTrainer`; keep family-specific assumptions in concrete trainers.
- Do not change only one layer. Most extension work crosses implementation, config, CLI, and docs.
- If behavior is selected by string name, check `enums.py`, config builders, and YAML examples together.
- If you are porting from `legacy/`, verify not only the model body but also transforms, scheduler semantics, losses, and trainer behavior.

## Detailed Guides

Use these when you need the exhaustive version:

- [docs/components/README.md](components/README.md)
- [docs/components/preprocessing.md](components/preprocessing.md)
- [docs/components/models.md](components/models.md)
- [docs/components/datasets.md](components/datasets.md)
- [docs/components/runtime-config.md](components/runtime-config.md)
- [docs/components/trainers.md](components/trainers.md)
- [docs/components/testing.md](components/testing.md)
- [docs/components/cli-gui.md](components/cli-gui.md)
