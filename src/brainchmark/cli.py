# Standard library
from enum import StrEnum
from pathlib import Path
import sys
import time

# External dependencies
from click.shell_completion import CompletionItem
import typer

# Internal modules
from brainchmark.utils.cli_overrides import (
    CONFIGS_DIR,
    SPLITS_DIR,
    load_yaml_config,
    merge_cli_overrides,
    resolve_split_path,
)
from brainchmark.utils.cli_utils import require_preprocess_values

class DatasetType(StrEnum):
    BRATS18 = "brats18"
    BRATS23 = "brats23"


class CropMode(StrEnum):
    NONE = "none"
    CENTER = "center"
    NON_EMPTY = "non_empty"


class ClampMode(StrEnum):
    NONE = "none"
    SUBJECT = "subject"
    DATASET = "dataset"


class NormMode(StrEnum):
    NONE = "none"
    MIN_MAX = "min_max"
    SUBJECT_ZSCORE = "subject_zscore"
    DATASET_ZSCORE = "dataset_zscore"


class TrainerKind(StrEnum):
    IMFUSE = "imfuse"


class ModelKind(StrEnum):
    IMFUSE = "imfuse"


class OptimizerKind(StrEnum):
    RADAM = "radam"
    ADAMW = "adamw"
    SGD = "sgd"
    ADAM = "adam"


class SchedulerKind(StrEnum):
    POLY = "poly"
    COSINE = "cosine"
    STEP = "step"
    MULTISTEP = "multistep"
    PLATEAU = "plateau"


app = typer.Typer(help="BrainchMark CLI",rich_markup_mode="rich")


def _config_shell_complete(
    _ctx: typer.Context,
    _param: typer.CallbackParam,
    incomplete: str,
) -> list[CompletionItem]:
    suggestions: dict[str, CompletionItem] = {}

    for config_path in sorted(CONFIGS_DIR.glob("*.y*ml")):
        if config_path.name.startswith(incomplete):
            suggestions[config_path.name] = CompletionItem(
                config_path.name,
                help=str(CONFIGS_DIR),
            )

    if incomplete.startswith("/") or "/" in incomplete or incomplete.startswith("."):
        raw_path = Path(incomplete).expanduser()
        parent = raw_path if incomplete.endswith("/") else raw_path.parent
        prefix = "" if incomplete.endswith("/") else raw_path.name
        if parent.exists() and parent.is_dir():
            for candidate in sorted(parent.iterdir()):
                if candidate.suffix not in {".yaml", ".yml"}:
                    continue
                if not candidate.name.startswith(prefix):
                    continue
                suggestions[str(candidate)] = CompletionItem(str(candidate))

    return list(suggestions.values())


def _split_shell_complete(
    _ctx: typer.Context,
    _param: typer.CallbackParam,
    incomplete: str,
) -> list[CompletionItem]:
    suggestions: dict[str, CompletionItem] = {}

    for split_path in sorted(path for path in SPLITS_DIR.rglob("*") if path.is_file()):
        relative_name = split_path.relative_to(SPLITS_DIR).as_posix()
        if relative_name.startswith(incomplete):
            suggestions[relative_name] = CompletionItem(
                relative_name,
                help=str(SPLITS_DIR),
            )

    if incomplete.startswith("/") or "/" in incomplete or incomplete.startswith("."):
        raw_path = Path(incomplete).expanduser()
        parent = raw_path if incomplete.endswith("/") else raw_path.parent
        prefix = "" if incomplete.endswith("/") else raw_path.name
        if parent.exists() and parent.is_dir():
            for candidate in sorted(parent.iterdir()):
                if not candidate.is_file():
                    continue
                if not candidate.name.startswith(prefix):
                    continue
                suggestions[str(candidate)] = CompletionItem(str(candidate))

    return list(suggestions.values())


def _require_train_values(merged: dict[str, object], *required_keys: str) -> None:
    for key in required_keys:
        if merged.get(key) is None:
            raise typer.BadParameter(
                "missing value; provide it in the CLI or in --config",
                param_hint=f"--{key.replace('_', '-')}",
            )


def _merged_value(
    merged: dict[str, object],
    key: str,
    default: object,
) -> object:
    value = merged.get(key)
    return default if value is None else value


def _resolve_resume_checkpoint(merged: dict[str, object]) -> Path | None:
    resume = merged.get("resume")
    if not resume:
        return None

    output_dir = merged.get("output_dir")
    if output_dir is None:
        raise typer.BadParameter(
            "missing value; provide it in the CLI or in --config",
            param_hint="--output-dir",
        )
    output_dir = merged.get("output_dir")
    checkpoint_path = Path(output_dir) / 'checkpoints' /  "model_last.pth"
    if not checkpoint_path.is_file():
        raise typer.BadParameter(
            f"resume checkpoint not found at {checkpoint_path}",
            param_hint="--output-dir",
        )
    return checkpoint_path


def _resolve_trainer_class(trainer_kind: TrainerKind):
    from brainchmark.training.trainers import IMFuseTrainer

    trainer_map = {
        TrainerKind.IMFUSE: IMFuseTrainer,
    }
    try:
        return trainer_map[trainer_kind]
    except KeyError as exc:
        raise typer.BadParameter(
            f"Unsupported trainer: {trainer_kind}",
            param_hint="--trainer",
        ) from exc


@app.callback()
def main() -> None:
    """BrainchMark command group."""


'''@app.command()
def hello() -> None:
    """Simple test command."""
    message = "Hello from the AImageLab Team!"

    text = Text(message, style="bold magenta")
    text.stylize("bold cyan", 0, 5)      # "Hello"
    text.stylize("bold yellow", 15, 24)  # "AImageLab"

    console.print()
    console.print(text, justify="center")
    console.print()'''

@app.command("self-destruct")
def self_destruct(
    force: bool = typer.Option(False, "--force", help="Skip confirmation prompt")
):
    """
    Totally irreversible self-destruct sequence.
    """
    if not force:
        confirm = typer.confirm("Are you absolutely sure you want to self-destruct?")
        if not confirm:
            typer.echo("Aborted.")
            raise typer.Exit()
        confirm = typer.confirm("Are you ABSOLUTELY sure you want to DESTROY YOUR PC AND THIS REPO?")
        if not confirm:
            typer.echo("Aborted.")
            raise typer.Exit()

    typer.echo("Initializing self-destruct sequence...\n")

    for i in range(5, 0, -1):
        typer.echo(f"{i}...", nl=False)
        sys.stdout.flush()
        time.sleep(1)
        typer.echo("")

    text = \
"""———————————No brains?———————————
⠀⣞⢽⢪⢣⢣⢣⢫⡺⡵⣝⡮⣗⢷⢽⢽⢽⣮⡷⡽⣜⣜⢮⢺⣜⢷⢽⢝⡽⣝
⠸⡸⠜⠕⠕⠁⢁⢇⢏⢽⢺⣪⡳⡝⣎⣏⢯⢞⡿⣟⣷⣳⢯⡷⣽⢽⢯⣳⣫⠇
⠀⠀⢀⢀⢄⢬⢪⡪⡎⣆⡈⠚⠜⠕⠇⠗⠝⢕⢯⢫⣞⣯⣿⣻⡽⣏⢗⣗⠏⠀
⠀⠪⡪⡪⣪⢪⢺⢸⢢⢓⢆⢤⢀⠀⠀⠀⠀⠈⢊⢞⡾⣿⡯⣏⢮⠷⠁⠀⠀
⠀⠀⠀⠈⠊⠆⡃⠕⢕⢇⢇⢇⢇⢇⢏⢎⢎⢆⢄⠀⢑⣽⣿⢝⠲⠉⠀⠀⠀⠀
⠀⠀⠀⠀⠀⡿⠂⠠⠀⡇⢇⠕⢈⣀⠀⠁⠡⠣⡣⡫⣂⣿⠯⢪⠰⠂⠀⠀⠀⠀
⠀⠀⠀⠀⡦⡙⡂⢀⢤⢣⠣⡈⣾⡃⠠⠄⠀⡄⢱⣌⣶⢏⢊⠂⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢝⡲⣜⡮⡏⢎⢌⢂⠙⠢⠐⢀⢘⢵⣽⣿⡿⠁⠁⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠨⣺⡺⡕⡕⡱⡑⡆⡕⡅⡕⡜⡼⢽⡻⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣼⣳⣫⣾⣵⣗⡵⡱⡡⢣⢑⢕⢜⢕⡝⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⣴⣿⣾⣿⣿⣿⡿⡽⡑⢌⠪⡢⡣⣣⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⡟⡾⣿⢿⢿⢵⣽⣾⣼⣘⢸⢸⣞⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠁⠇⠡⠩⡫⢿⣝⡻⡮⣒⢽⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
—————————————————————————————
    """
    typer.echo(text)

@app.command()
def preprocess(
    config: Path | None = typer.Option(
        None,
        "--config",
        file_okay=True,
        dir_okay=False,
        shell_complete=_config_shell_complete,
        help="Path to a YAML config file.",
        rich_help_panel="Config",
    ),
    input_dir: Path = typer.Option(
        None,
        "--input-dir",
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        help="Directory containing the input data to preprocess.",
        rich_help_panel="Input/Output",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        help="Directory where preprocessed data will be written.",
        rich_help_panel="Input/Output",
    ),
    dataset_type: DatasetType = typer.Option(
        None,
        "--dataset-type",
        help="Input dataset type. Choose either brats18 or brats23",
        rich_help_panel="Input/Output",
    ),
    crop_mode: str = typer.Option(
        CropMode.NONE.value,
        "--crop-mode",
        help= "Cropping strategy. Built-ins: " + ", ".join(mode.value for mode in CropMode) + ". Custom function names from preprocessing.cropping are also accepted.",
        rich_help_panel="Cropping",
    ),
    crop_size: list[int] | None | None = typer.Option(
        None,
        "--crop-size",
        help="Center crop size as three integers: X Y Z.",
        rich_help_panel="Cropping",
    ),
    crop_min_size: list[int] | None | None = typer.Option(
        None,
        "--crop-min-size",
        help="Minimum non-empty crop size as three integers: X Y Z.",
        rich_help_panel="Cropping",
    ),
    clamp_mode: str = typer.Option(
        ClampMode.NONE.value,
        "--clamp-mode",
        help="Clamp mode. Built-ins: " + ", ".join(mode.value for mode in ClampMode) + ". Custom function names from preprocessing.clamping are also accepted.",
        rich_help_panel="Clamping",
    ),

    clamp_percentile: list[float] | None = typer.Option(
        None,
        "--clamp-percentile",
        help="Clamp percentiles as one or two floats: HIGH or LOW HIGH. If one value is given, LOW is assumed to be 0.",
        rich_help_panel="Clamping",
    ),

    clamp_min: list[int] | None = typer.Option(
        None,
        "--clamp-min",
        help="Minimum clamp values as one or four integers. If one value is given, it will be used for all modalities.",
        rich_help_panel="Clamping",

    ),

    clamp_max: list[int] | None = typer.Option(
        None,
        "--clamp-max",
        help="Maximum clamp values as one or four integers. If one value is given, it will be used for all modalities.",
        rich_help_panel="Clamping",

    ),

    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Automatically answer yes to prompts",
        rich_help_panel="Execution",
    ),
    norm_mode: str = typer.Option(
        NormMode.NONE.value,
        "--norm-mode",
        help="Normalization mode. Built-ins: " + ", ".join(mode.value for mode in NormMode) + ". Custom function names from preprocessing.normalization are also accepted.",
        rich_help_panel="Normalization",
    ),
    norm_min_max_range: tuple[float, float] | None = typer.Option(
        None,
        "--norm-min-max-range",
        help="Target min-max normalization range as two floats: MIN MAX.",
        rich_help_panel="Normalization",
    ),

    norm_mean: tuple[float, float, float, float] | None = typer.Option(
        None,
        "--norm-mean",
        help="Normalization means as four floats, one for each modality.",
        rich_help_panel="Normalization",
    ),

    norm_std: tuple[float, float, float, float] | None = typer.Option(
        None,
        "--norm-std",
        help="Normalization standard deviations as four floats, one for each modality.",
        rich_help_panel="Normalization",
    ),

) -> None:
    from brainchmark.preprocessing.config import (
        build_clamp_config,
        build_crop_config,
        build_norm_config,
    )
    from brainchmark.preprocessing.pipeline import run_preprocessing
    from brainchmark.datasets.config import DatasetType as PreprocessingDatasetType

    #TODO COMPLETE THE OVERRIDES, REMEMBER TO CHANGE ALL THE CALLS UNDERNEATH
    yaml_config = load_yaml_config(config)
    merged = merge_cli_overrides(
        yaml_config,
        input_dir=input_dir,
        output_dir=output_dir,
        dataset_type=dataset_type,
        crop_mode=crop_mode,
        crop_size=crop_size,
        crop_min_size=crop_min_size,
        clamp_mode=clamp_mode,
        clamp_percentile=clamp_percentile,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        norm_mode=norm_mode,
        norm_min_max_range=norm_min_max_range,
        norm_mean=norm_mean,
        norm_std=norm_std,
        yes=yes if yes else yaml_config.get("yes"),
    )
    require_preprocess_values(merged,"input_dir", "output_dir", "dataset_type")
    """Run dataset preprocessing."""
    crop_config = build_crop_config(
        crop_mode=str(merged.get("crop_mode", CropMode.NONE.value)),
        crop_size=merged.get("crop_size"),
        crop_min_size=merged.get("crop_min_size"),
    )
    clamp_config = build_clamp_config(
        clamp_mode=str(merged.get("clamp_mode", ClampMode.NONE.value)),
        clamp_percentile=merged.get("clamp_percentile"),
        clamp_min=merged.get("clamp_min"),
        clamp_max=merged.get("clamp_max"),
    )
    norm_config = build_norm_config(
        norm_mode=str(merged.get("norm_mode", NormMode.NONE.value)),
        norm_min_max_range=merged.get("norm_min_max_range"),
        norm_mean=merged.get("norm_mean"),
        norm_std=merged.get("norm_std"),
    )
    typer.echo(
        "Preprocessing data "
        f"from {merged.get('input_dir')} to {merged.get('output_dir')} "
        f"assuming a '{merged.get('dataset_type')}' configuration "
        f"with crop mode '{crop_config.fn.__name__}', "
        f"clamp mode '{clamp_config.fn.__name__}', "
        f"normalization mode '{norm_config.fn.__name__}'."
    )
    run_preprocessing(
        input_dir=Path(merged.get("input_dir")),
        output_dir=Path(merged.get("output_dir")),
        dataset_type=PreprocessingDatasetType(merged.get("dataset_type")),
        crop_config=crop_config,
        clamp_config=clamp_config,
        norm_config = norm_config,
        yes=yes,
    )


@app.command()
def train(
    config: Path | None = typer.Option(
        None,
        "--config",
        file_okay=True,
        dir_okay=False,
        shell_complete=_config_shell_complete,
        help="Path to a YAML config file.",
        rich_help_panel="Config",
    ),
    data_dir: Path | None = typer.Option(
        None,
        "--data-dir",
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        help="Directory containing the training data.",
        rich_help_panel="Input/Output",
    ),
    art_dir: Path | None = typer.Option(
        None,
        "--art-dir",
        file_okay=False,
        dir_okay=True,
        help="Directory where training artifacts will be written.",
        rich_help_panel="Input/Output",
    ),
    trainer: TrainerKind | None = typer.Option(
        None,
        "--trainer",
        help="Trainer implementation or preset to use.",
        rich_help_panel="Trainer",
    ),
    model: ModelKind | None = typer.Option(
        None,
        "--model",
        help="Model implementation or preset to use.",
        rich_help_panel="Model",
    ),
    loss: str | None = typer.Option(
        None,
        "--loss",
        help="Loss implementation or preset to use.",
        rich_help_panel="Loss",
    ),
    custom_model_kwargs: list[str] | None = typer.Option(
        None,
        "--custom-model-kwargs",
        help="Additional model kwargs in key=value form.",
        rich_help_panel="Model",
    ),
    custom_loss_kwargs: list[str] | None = typer.Option(
        None,
        "--custom-loss-kwargs",
        help="Additional loss kwargs in key=value form.",
        rich_help_panel="Loss",
    ),
    custom_trainer_kwargs: list[str] | None = typer.Option(
        None,
        "--custom-trainer-kwargs",
        "--costom--trainer--kwargs",
        help="Additional trainer kwargs in key=value form.",
        rich_help_panel="Trainer",
    ),
    split_file: Path | None = typer.Option(
        None,
        "--split-file",
        file_okay=True,
        dir_okay=False,
        shell_complete=_split_shell_complete,
        help="Split file path. Relative paths are resolved under brainchmark/data/splits.",
        rich_help_panel="Trainer",
        show_default="split.json",
    ),
    optimizer: OptimizerKind | None = typer.Option(
        None,
        "--optimizer",
        help="Optimizer name.",
        rich_help_panel="Optimization",
    ),
    betas: tuple[float, float] | None = typer.Option(
        None,
        "--betas",
        help="Optimizer betas as two floats: BETA1 BETA2.",
        rich_help_panel="Optimization",
    ),
    momentum: float | None = typer.Option(
        None,
        "--momentum",
        help="SGD momentum.",
        rich_help_panel="Optimization",
    ),
    scheduler: SchedulerKind | None = typer.Option(
        None,
        "--scheduler",
        help="Learning-rate scheduler name.",
        rich_help_panel="Scheduler",
    ),
    poly_total_iters: int | None = typer.Option(
        None,
        "--poly-total-iters",
        help="Total iterations for the polynomial scheduler. Defaults to num_epochs.",
        rich_help_panel="Scheduler",
    ),
    poly_power: float | None = typer.Option(
        None,
        "--poly-power",
        help="Power for the polynomial scheduler.",
        rich_help_panel="Scheduler",
    ),
    cosine_t_max: int | None = typer.Option(
        None,
        "--cosine-t-max",
        help="T_max for cosine annealing. Defaults to num_epochs.",
        rich_help_panel="Scheduler",
    ),
    cosine_eta_min: float | None = typer.Option(
        None,
        "--cosine-eta-min",
        help="Minimum learning rate for cosine annealing.",
        rich_help_panel="Scheduler",
    ),
    step_step_size: int | None = typer.Option(
        None,
        "--step-step-size",
        help="Step size for StepLR.",
        rich_help_panel="Scheduler",
    ),
    step_gamma: float | None = typer.Option(
        None,
        "--step-gamma",
        help="Decay factor for StepLR.",
        rich_help_panel="Scheduler",
    ),
    multistep_milestones: list[int] | None = typer.Option(
        None,
        "--multistep-milestones",
        help="Milestones for MultiStepLR. Repeat the option or pass multiple integers.",
        rich_help_panel="Scheduler",
    ),
    multistep_gamma: float | None = typer.Option(
        None,
        "--multistep-gamma",
        help="Decay factor for MultiStepLR.",
        rich_help_panel="Scheduler",
    ),
    plateau_mode: str | None = typer.Option(
        None,
        "--plateau-mode",
        help="Mode for ReduceLROnPlateau, usually min or max.",
        rich_help_panel="Scheduler",
    ),
    plateau_factor: float | None = typer.Option(
        None,
        "--plateau-factor",
        help="Decay factor for ReduceLROnPlateau.",
        rich_help_panel="Scheduler",
    ),
    plateau_patience: int | None = typer.Option(
        None,
        "--plateau-patience",
        help="Patience for ReduceLROnPlateau.",
        rich_help_panel="Scheduler",
    ),
    train_transforms: str | None = typer.Option(
        None,
        "--train-transforms",
        help="Training transform pipeline expression.",
        rich_help_panel="Data Pipeline",
    ),
    test_transforms: str | None = typer.Option(
        None,
        "--test-transforms",
        help="Validation/test transform pipeline expression.",
        rich_help_panel="Data Pipeline",
    ),
    lr: float | None = typer.Option(
        None,
        "--lr",
        help="Learning rate.",
        rich_help_panel="Optimization",
    ),
    num_epochs: int | None = typer.Option(
        None,
        "--num-epochs",
        help="Number of training epochs.",
        rich_help_panel="Training",
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        help="Mini-batch size.",
        rich_help_panel="Training",
    ),
    weight_decay: float | None = typer.Option(
        None,
        "--weight-decay",
        help="Weight decay.",
        rich_help_panel="Optimization",
    ),
    num_workers: int= typer.Option(
        8,
        "--num-workers",
        help="Number of dataloader workers.",
        rich_help_panel="Runtime",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume training from <output_dir>/model_last.pth.",
        rich_help_panel="Checkpointing",
    ),
    seed: int = typer.Option(
        69,
        "--seed",
        help="Random seed.",
        rich_help_panel="Runtime",
    ),
    pretrain: Path | None = typer.Option(
        None,
        "--pretrain",
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to pretrained weights to load before training.",
        rich_help_panel="Checkpointing",
    ),
    wandb_project: str | None = typer.Option(
        None,
        "--wandb-project",
        help="Weights & Biases project name.",
        rich_help_panel="Logging",
    ),
    wandb_mode: str | None = typer.Option(
        None,
        "--wandb-mode",
        help="Weights & Biases mode, for example online, offline, or disabled.",
        rich_help_panel="Logging",
    ),
    dataset_type: DatasetType = typer.Option(
        None,
        "--dataset-type",
        help="Input dataset type. Choose either brats18 or brats23",
        rich_help_panel="Input/Output",
    ),
) -> None:
    """Run training from CLI overrides and YAML configuration."""

    from brainchmark.training.config import (
        OptimizerKind as TrainingOptimizerKind,
        SchedulerKind as TrainingSchedulerKind,
        build_optimizer_config,
        build_scheduler_config,
        parse_kv_list,
    )
    from brainchmark.losses.config import build_loss_config

    from brainchmark.models.config import (
    ModelKind as TrainingModelKind,
    build_model_config
    )

    yaml_config = load_yaml_config(config)
    trainer_kwargs = parse_kv_list(custom_trainer_kwargs)
    merged = merge_cli_overrides(
        yaml_config,
        data_dir=data_dir,
        art_dir=art_dir,
        trainer=trainer,
        model=model,
        loss=loss,
        custom_model_kwargs=parse_kv_list(custom_model_kwargs),
        custom_loss_kwargs=parse_kv_list(custom_loss_kwargs),
        custom_trainer_kwargs=trainer_kwargs,
        split_file=split_file,
        optimizer=optimizer,
        betas=betas,
        momentum=momentum,
        scheduler=scheduler,
        poly_total_iters=poly_total_iters,
        poly_power=poly_power,
        cosine_t_max=cosine_t_max,
        cosine_eta_min=cosine_eta_min,
        step_step_size=step_step_size,
        step_gamma=step_gamma,
        multistep_milestones=multistep_milestones,
        multistep_gamma=multistep_gamma,
        plateau_mode=plateau_mode,
        plateau_factor=plateau_factor,
        plateau_patience=plateau_patience,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        lr=lr,
        num_epochs=num_epochs,
        batch_size=batch_size,
        weight_decay=weight_decay,
        num_workers=num_workers,
        resume=resume,
        pretrain=pretrain,
        seed=seed,
        wandb_project=wandb_project,
        wandb_mode=wandb_mode,
        dataset_type=dataset_type,
    )
    merged_trainer_kwargs = dict(merged.get("custom_trainer_kwargs") or {})
    split_file_value = merged.get("split_file")
    if split_file_value is None:
        split_file_value = merged_trainer_kwargs.get("split_file")
    resolved_split_file = resolve_split_path(split_file_value)
    merged_trainer_kwargs["split_file"] = str(resolved_split_file)
    merged["custom_trainer_kwargs"] = merged_trainer_kwargs
    merged["split_file"] = str(resolved_split_file)
    _require_train_values(
        merged,
        "data_dir",
        "art_dir",
        "trainer",
        "optimizer",
        "num_epochs",
    )
    model_kind = TrainingModelKind(merged.get("model", TrainingModelKind.IMFUSE))
    optimizer_kind = TrainingOptimizerKind(
        merged.get("optimizer", TrainingOptimizerKind.RADAM)
    )
    scheduler_kind = TrainingSchedulerKind(
        merged.get("scheduler", TrainingSchedulerKind.POLY)
    )
    model_config = build_model_config(
        model_kind=model_kind,
        model_kwargs=merged.get("custom_model_kwargs"),
    )
    loss_config = build_loss_config(
        loss_kind=merged.get("loss", "imfuse"),
        loss_kwargs=merged.get("custom_loss_kwargs"),
    )
    resolved_num_epochs = int(merged["num_epochs"])
    optimizer_config = build_optimizer_config(
        optimizer_kind=optimizer_kind,
        lr=float(_merged_value(merged, "lr", 2e-4)),
        weight_decay=float(_merged_value(merged, "weight_decay", 3e-5)),
        betas=tuple(merged["betas"]) if merged.get("betas") is not None else (0.9, 0.999),
        momentum=float(_merged_value(merged, "momentum", 0.9)),
    )
    scheduler_config = build_scheduler_config(
        scheduler_kind=scheduler_kind,
        poly_total_iters=int(merged["poly_total_iters"]) if merged.get("poly_total_iters") is not None else resolved_num_epochs,
        poly_power=float(_merged_value(merged, "poly_power", 0.9)),
        cosine_t_max=int(merged["cosine_t_max"]) if merged.get("cosine_t_max") is not None else resolved_num_epochs,
        cosine_eta_min=float(_merged_value(merged, "cosine_eta_min", 0.0)),
        step_step_size=int(merged["step_step_size"]) if merged.get("step_step_size") is not None else None,
        step_gamma=float(_merged_value(merged, "step_gamma", 0.1)),
        multistep_milestones=list(merged["multistep_milestones"]) if merged.get("multistep_milestones") is not None else None,
        multistep_gamma=float(_merged_value(merged, "multistep_gamma", 0.1)),
        plateau_mode=str(_merged_value(merged, "plateau_mode", "min")),
        plateau_factor=float(_merged_value(merged, "plateau_factor", 0.1)),
        plateau_patience=int(_merged_value(merged, "plateau_patience", 10)),
    )
    trainer_kind = TrainerKind(merged.get("trainer", TrainerKind.IMFUSE))
    trainer_class = _resolve_trainer_class(trainer_kind)
    trainer_instance = trainer_class(
        input_dir=Path(merged["data_dir"]),
        output_dir=Path(merged["art_dir"]),
        custom_trainer_kwargs=merged_trainer_kwargs,
        model_config=model_config,
        loss_config=loss_config,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        train_transforms=merged.get("train_transforms"),
        test_transforms=merged.get("test_transforms"),
        num_epochs=resolved_num_epochs,
        batch_size=int(merged.get("batch_size", 1)),
        num_workers=int(merged.get("num_workers", 8)),
        resume=bool(merged.get("resume", False)),
        seed=int(merged.get("seed", 69)) if merged.get("seed") is not None else None,
        pretrain=merged.get("pretrain"),
        wandb_project=merged.get("wandb_project"),
        wandb_mode=merged.get("wandb_mode"),
        dataset_type=merged.get("dataset_type")
    )
    trainer_instance.fit()
