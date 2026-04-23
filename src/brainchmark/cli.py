# Standard library
import os
from pathlib import Path
import sys
import time

# External dependencies
import typer
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

# Internal modules
from brainchmark import __version__
from brainchmark.cli_completion import config_shell_complete, split_shell_complete
from brainchmark.cli_display import (
    CONSOLE,
    prompt_optional_existing_directory,
    prompt_required_directory,
)
from brainchmark.cli_setup import (
    CONFIG_TEMPLATES_DIR,
    copy_config_templates,
    update_setup_config,
)
from brainchmark.cli_workflows import (
    build_preprocess_merged_config,
    build_train_merged_config,
    maybe_relaunch_distributed,
    run_preprocess_from_merged,
    run_train_from_merged,
)
from brainchmark.enums import (
    ClampMode,
    CropMode,
    DatasetType,
    ModelKind,
    NormMode,
    OptimizerKind,
    SchedulerKind,
    TransformKind,
    TrainerKind,
)
from brainchmark.utils.cli_overrides import CONFIGS_DIR, SPLITS_DIR, load_yaml_config, merge_cli_overrides, resolve_split_path

# Environment variables
os.environ["WANDB_SILENT"] = "true"

app = typer.Typer(help="BrainchMark CLI",rich_markup_mode="rich")


def _version_callback(value: bool) -> None:
    if not value:
        return

    typer.echo(f"BrainchMark {__version__}")
    raise typer.Exit()


def _require_values(merged: dict[str, object], *required_keys: str) -> None:
    for key in required_keys:
        if merged.get(key) is None:
            raise typer.BadParameter(
                "missing value; provide it in the CLI or in --config",
                param_hint=f"--{key.replace('_', '-')}",
            )


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


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show BrainchMark version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """BrainchMark command group."""


@app.command()
def version(
) -> None:
    """Show the installed BrainchMark version."""
    typer.echo(f"BrainchMark {__version__}")


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
def setup() -> None:
    """Copy template configs and patch only the local path fields."""
    CONSOLE.print(
        Panel(
            "[bold white]Configure the packaged BrainchMark YAML files.[/bold white]\n"
            "Leave a BraTS dataset path empty if you do not want to configure that dataset yet.",
            title="[bold green]BrainchMark Setup[/bold green]",
            border_style="green",
            expand=False,
        )
    )

    brats23_dir = prompt_optional_existing_directory(
        label="BraTS23",
        prompt="Directory containing BraTS23",
    )
    brats18_dir = prompt_optional_existing_directory(
        label="BraTS18",
        prompt="Directory containing BraTS18",
    )
    if brats23_dir is None and brats18_dir is None:
        raise typer.BadParameter(
            "at least one dataset directory must be provided",
            param_hint="brainchmark setup",
        )

    preprocessed_root_dir = prompt_required_directory(
        label="Preprocessed Root",
        prompt="Root directory for preprocessed datasets",
    )
    artifacts_root_dir = prompt_required_directory(
        label="Artifacts Root",
        prompt="Root directory for training artifacts",
    )

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(style="white")
    table.add_row("BraTS23", str(brats23_dir) if brats23_dir is not None else "null")
    table.add_row("BraTS18", str(brats18_dir) if brats18_dir is not None else "null")
    table.add_row("Preprocessed Root", str(preprocessed_root_dir))
    table.add_row("Artifacts Root", str(artifacts_root_dir))
    table.add_row("Templates", str(CONFIG_TEMPLATES_DIR))
    table.add_row("Configs", str(CONFIGS_DIR))
    table.add_row("Checkpoint Path", "<art_dir>/checkpoints/model_last.pth")
    table.add_row("Output Path", "<art_dir>/results.txt")

    CONSOLE.print(
        Panel(
            table,
            title="[bold green]Setup Plan[/bold green]",
            border_style="green",
            expand=False,
        )
    )

    if not Confirm.ask("Copy templates and rewrite local BrainchMark configs?", default=True):
        raise typer.Abort()

    updated_files = copy_config_templates()
    for config_path in updated_files:
        update_setup_config(
            config_path=config_path,
            brats18_dir=brats18_dir,
            brats23_dir=brats23_dir,
            preprocessed_root_dir=preprocessed_root_dir,
            artifacts_root_dir=artifacts_root_dir,
        )

    result_table = Table.grid(padding=(0, 2))
    result_table.add_column(style="bold cyan", no_wrap=True)
    result_table.add_column(style="white")
    result_table.add_row("Updated", str(len(updated_files)))
    result_table.add_row("Templates", str(CONFIG_TEMPLATES_DIR))
    result_table.add_row("BraTS23", str(brats23_dir) if brats23_dir is not None else "null")
    result_table.add_row("BraTS18", str(brats18_dir) if brats18_dir is not None else "null")
    result_table.add_row("Preprocessed Root", str(preprocessed_root_dir))
    result_table.add_row("Artifacts Root", str(artifacts_root_dir))
    result_table.add_row("Files", ", ".join(path.name for path in updated_files))

    CONSOLE.print(
        Panel(
            result_table,
            title="[bold green]Setup Complete[/bold green]",
            border_style="green",
            expand=False,
        )
    )

@app.command(
    help=(
        "Preprocess a BraTS-style dataset into the compressed `.npz` format "
        "used by BrainchMark training and testing."
    ),
    short_help="Preprocess a BraTS-style dataset.",
)
def preprocess(
    config: Path | None = typer.Option(
        None,
        "--config",
        file_okay=True,
        dir_okay=False,
        shell_complete=config_shell_complete,
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
    """Preprocess a BraTS-style dataset into BrainchMark `.npz` artifacts."""
    merged = build_preprocess_merged_config(
        config=config,
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
        yes=yes,
    )
    run_preprocess_from_merged(merged, console=CONSOLE, yes=yes)


@app.command()
def preprocess_train(
    config: Path | None = typer.Option(
        None,
        "--config",
        file_okay=True,
        dir_okay=False,
        shell_complete=config_shell_complete,
        help="Path to a YAML config file.",
        rich_help_panel="Config",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Automatically answer yes to preprocessing prompts",
        rich_help_panel="Execution",
    ),
    distributed: bool | None = typer.Option(
        None,
        "--distributed/--no-distributed",
        help="Relaunch training through torchrun for DDP.",
        rich_help_panel="Runtime",
    ),
    nproc_per_node: int | None = typer.Option(
        None,
        "--nproc-per-node",
        help="Processes to launch per node for distributed training. Defaults to the number of visible CUDA devices.",
        rich_help_panel="Runtime",
    ),
) -> None:
    """Run preprocessing first and then launch training using the same config."""
    preprocess_merged = build_preprocess_merged_config(
        config=config,
        input_dir=None,
        output_dir=None,
        dataset_type=None,
        crop_mode=CropMode.NONE.value,
        crop_size=None,
        crop_min_size=None,
        clamp_mode=ClampMode.NONE.value,
        clamp_percentile=None,
        clamp_min=None,
        clamp_max=None,
        norm_mode=NormMode.NONE.value,
        norm_min_max_range=None,
        norm_mean=None,
        norm_std=None,
        yes=yes,
    )
    run_preprocess_from_merged(preprocess_merged, console=CONSOLE, yes=yes)

    train_merged = build_train_merged_config(
        config=config,
        data_dir=None,
        art_dir=None,
        trainer=None,
        model=None,
        loss=None,
        custom_model_kwargs=None,
        custom_loss_kwargs=None,
        loss_num_classes=None,
        fuse_weight=None,
        sep_weight=None,
        prm_weight=None,
        loss_eps=None,
        log_clamp_min=None,
        custom_trainer_kwargs=None,
        split_file=None,
        optimizer=None,
        betas=None,
        momentum=None,
        scheduler=None,
        poly_total_iters=None,
        poly_power=None,
        cosine_t_max=None,
        cosine_eta_min=None,
        step_step_size=None,
        step_gamma=None,
        multistep_milestones=None,
        multistep_gamma=None,
        plateau_mode=None,
        plateau_factor=None,
        plateau_patience=None,
        transform_kind=None,
        lr=None,
        num_epochs=None,
        batch_size=None,
        weight_decay=None,
        num_workers=8,
        distributed=distributed,
        nproc_per_node=nproc_per_node,
        fp16=None,
        resume=False,
        seed=69,
        pretrain=None,
        wandb_project=None,
        wandb_mode=None,
        wandb_run_name=None,
        dataset_type=None,
    )
    maybe_relaunch_distributed(train_merged)
    run_train_from_merged(train_merged)


@app.command()
def train(
    config: Path | None = typer.Option(
        None,
        "--config",
        file_okay=True,
        dir_okay=False,
        shell_complete=config_shell_complete,
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
    loss_num_classes: int | None = typer.Option(
        None,
        "--loss-num-classes",
        help="Override the number of classes used by the loss.",
        rich_help_panel="Loss",
    ),
    fuse_weight: float | None = typer.Option(
        None,
        "--fuse-weight",
        help="Weight for the fused branch loss.",
        rich_help_panel="Loss",
    ),
    sep_weight: float | None = typer.Option(
        None,
        "--sep-weight",
        help="Weight for the separate branch loss.",
        rich_help_panel="Loss",
    ),
    prm_weight: float | None = typer.Option(
        None,
        "--prm-weight",
        help="Weight for the PRM branch loss.",
        rich_help_panel="Loss",
    ),
    loss_eps: float | None = typer.Option(
        None,
        "--loss-eps",
        help="Numerical stability epsilon used inside the loss.",
        rich_help_panel="Loss",
    ),
    log_clamp_min: float | None = typer.Option(
        None,
        "--log-clamp-min",
        help="Minimum probability clamp used before log in the weighted cross-entropy term.",
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
        shell_complete=split_shell_complete,
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
    transform_kind: TransformKind | None = typer.Option(
        None,
        "--transform-kind",
        help="Transform manager implementation to use.",
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
    distributed: bool | None = typer.Option(
        None,
        "--distributed/--no-distributed",
        help="Relaunch training through torchrun for DDP.",
        rich_help_panel="Runtime",
    ),
    nproc_per_node: int | None = typer.Option(
        None,
        "--nproc-per-node",
        help="Processes to launch per node for distributed training. Defaults to the number of visible CUDA devices.",
        rich_help_panel="Runtime",
    ),
    fp16: bool | None = typer.Option(
        None,
        "--fp16",
        help="Enable float16 mixed precision training on CUDA.",
        rich_help_panel="Runtime",
        is_flag=True,
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
    wandb_run_name: str | None = typer.Option(
        None,
        "--wandb-run-name",
        help="Optional Weights & Biases run name. Defaults to 'training'.",
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
    with CONSOLE.status("[bold cyan]Starting BrainchMark[/bold cyan]", spinner="dots") as status:
        status.update("[bold cyan]Starting BrainchMark[/bold cyan]  [dim]reading configuration[/dim]")
        merged = build_train_merged_config(
            config=config,
            data_dir=data_dir,
            art_dir=art_dir,
            trainer=trainer,
            model=model,
            loss=loss,
            custom_model_kwargs=parse_kv_list(custom_model_kwargs),
            custom_loss_kwargs=parse_kv_list(custom_loss_kwargs),
            loss_num_classes=loss_num_classes,
            fuse_weight=fuse_weight,
            sep_weight=sep_weight,
            prm_weight=prm_weight,
            loss_eps=loss_eps,
            log_clamp_min=log_clamp_min,
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
            transform_kind=transform_kind,
            lr=lr,
            num_epochs=num_epochs,
            batch_size=batch_size,
            weight_decay=weight_decay,
            num_workers=num_workers,
            fp16=fp16,
            resume=resume,
            pretrain=pretrain,
            seed=seed,
            wandb_project=wandb_project,
            wandb_mode=wandb_mode,
            wandb_run_name=wandb_run_name,
            dataset_type=dataset_type,
        )
        maybe_relaunch_distributed(merged)

        status.update("[bold cyan]Starting BrainchMark[/bold cyan]  [dim]building training objects[/dim]")
        status.update("[bold cyan]Starting BrainchMark[/bold cyan]  [dim]initializing trainer[/dim]")
        run_train_from_merged(merged)


@app.command()
def test(
    config: Path | None = typer.Option(
        None,
        "--config",
        file_okay=True,
        dir_okay=False,
        shell_complete=config_shell_complete,
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
        help="Directory containing the preprocessed test data.",
        rich_help_panel="Input/Output",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output-path",
        file_okay=True,
        dir_okay=False,
        help="Text file where mask-sweep test results will be written.",
        rich_help_panel="Input/Output",
    ),
    checkpoint_path: Path | None = typer.Option(
        None,
        "--checkpoint-path",
        file_okay=True,
        dir_okay=False,
        exists=True,
        readable=True,
        help="Checkpoint path to evaluate.",
        rich_help_panel="Checkpointing",
    ),
    model: ModelKind | None = typer.Option(
        None,
        "--model",
        help="Model implementation or preset to use.",
        rich_help_panel="Model",
    ),
    custom_model_kwargs: list[str] | None = typer.Option(
        None,
        "--custom-model-kwargs",
        help="Additional model kwargs in key=value form.",
        rich_help_panel="Model",
    ),
    split_file: Path | None = typer.Option(
        None,
        "--split-file",
        file_okay=True,
        dir_okay=False,
        shell_complete=split_shell_complete,
        help="Split file path. Relative paths are resolved under brainchmark/data/splits.",
        rich_help_panel="Data Pipeline",
        show_default="split.json",
    ),
    num_workers: int = typer.Option(
        8,
        "--num-workers",
        help="Number of dataloader workers.",
        rich_help_panel="Runtime",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed.",
        rich_help_panel="Runtime",
    ),
    dataset_type: DatasetType = typer.Option(
        None,
        "--dataset-type",
        help="Input dataset type. Choose either brats18 or brats23",
        rich_help_panel="Input/Output",
    ),
) -> None:
    """Run mask-sweep testing from CLI overrides and YAML configuration."""
    from brainchmark.models.config import (
        ModelKind as TestingModelKind,
        build_model_config,
    )
    from brainchmark.testing import run_testing
    from brainchmark.training.config import parse_kv_list

    yaml_config = load_yaml_config(config)
    merged = merge_cli_overrides(
        yaml_config,
        data_dir=data_dir,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        model=model,
        custom_model_kwargs=parse_kv_list(custom_model_kwargs),
        split_file=split_file,
        num_workers=num_workers,
        seed=seed,
        dataset_type=dataset_type,
    )
    resolved_split_file = resolve_split_path(merged.get("split_file"))
    merged["split_file"] = str(resolved_split_file)
    _require_values(
        merged,
        "data_dir",
        "output_path",
        "checkpoint_path",
        "dataset_type",
    )

    model_kind = TestingModelKind(merged.get("model", TestingModelKind.IMFUSE))
    model_config = build_model_config(
        model_kind=model_kind,
        model_kwargs=merged.get("custom_model_kwargs"),
    )
    output_file = run_testing(
        data_dir=Path(merged["data_dir"]),
        output_path=Path(merged["output_path"]),
        checkpoint_path=Path(merged["checkpoint_path"]),
        dataset_type=DatasetType(merged["dataset_type"]),
        model_class=model_config.model_class,
        model_kwargs=model_config.kwargs,
        split_file=resolved_split_file,
        num_workers=int(merged.get("num_workers", 8)),
        seed=int(merged.get("seed", 42)),
    )
    typer.echo(f"Test report written to {output_file}")
