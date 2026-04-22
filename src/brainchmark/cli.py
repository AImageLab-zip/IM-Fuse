# Standard library
import os
from pathlib import Path
import re
import shutil
import sys
import time

# External dependencies
from click.shell_completion import CompletionItem
import typer
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.completion import PathCompleter
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

# Internal modules
from brainchmark import __version__
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
from brainchmark.updater import UpdateError, update_checkout
from brainchmark.versioning import check_for_updates
from brainchmark.utils.cli_overrides import (
    CONFIGS_DIR,
    SPLITS_DIR,
    load_yaml_config,
    merge_cli_overrides,
    resolve_split_path,
)
from brainchmark.utils.cli_utils import require_preprocess_values

# Environment variables
os.environ["WANDB_SILENT"] = "true"

app = typer.Typer(help="BrainchMark CLI",rich_markup_mode="rich")
CONSOLE = Console()
PATH_COMPLETER = PathCompleter(expanduser=True)
CONFIG_TEMPLATES_DIR = CONFIGS_DIR.parent / "config_templates"


def _version_callback(value: bool) -> None:
    if not value:
        return

    typer.echo(f"BrainchMark {__version__}")
    raise typer.Exit()


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


def _require_values(merged: dict[str, object], *required_keys: str) -> None:
    _require_train_values(merged, *required_keys)


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


def _distributed_launch_active() -> bool:
    return "LOCAL_RANK" in os.environ or int(os.environ.get("WORLD_SIZE", "1")) > 1


def _infer_nproc_per_node() -> int:
    import torch

    num_devices = torch.cuda.device_count()
    if num_devices <= 0:
        raise typer.BadParameter(
            "unable to infer --nproc-per-node because no CUDA devices are visible",
            param_hint="--nproc-per-node",
        )
    return num_devices


def _relaunch_with_torchrun(nproc_per_node: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        str(nproc_per_node),
        sys.argv[0],
        *sys.argv[1:],
    ]
    raise typer.Exit(os.spawnvp(os.P_WAIT, sys.executable, command))


def _resolve_trainer_class(trainer_kind: TrainerKind):
    from brainchmark.training.trainers import DCSegTrainer, IMFuseTrainer

    trainer_map = {
        TrainerKind.IMFUSE: IMFuseTrainer,
        TrainerKind.DCSEG: DCSegTrainer,
    }
    try:
        return trainer_map[trainer_kind]
    except KeyError as exc:
        raise typer.BadParameter(
            f"Unsupported trainer: {trainer_kind}",
            param_hint="--trainer",
        ) from exc


def _expand_user_path(raw_value: str) -> Path:
    return Path(raw_value).expanduser().resolve(strict=False)


def _prompt_path(prompt: str) -> str:
    return pt_prompt(
        f"{prompt}: ",
        completer=PATH_COMPLETER,
        complete_while_typing=True,
    ).strip()


def _prompt_optional_existing_directory(
    *,
    label: str,
    prompt: str,
) -> Path | None:
    while True:
        raw_value = _prompt_path(prompt)
        if not raw_value:
            return None

        candidate = _expand_user_path(raw_value)
        if candidate.is_dir():
            return candidate

        CONSOLE.print(
            Panel(
                f"[bold red]{label}[/bold red]\n{candidate} is not an existing directory.",
                title="[bold red]Invalid Directory[/bold red]",
                border_style="red",
                expand=False,
            )
        )


def _prompt_required_directory(
    *,
    label: str,
    prompt: str,
) -> Path:
    while True:
        raw_value = _prompt_path(prompt)
        if not raw_value:
            CONSOLE.print(
                Panel(
                    f"[bold red]{label}[/bold red] is required.",
                    title="[bold red]Missing Value[/bold red]",
                    border_style="red",
                    expand=False,
                )
            )
            continue

        return _expand_user_path(raw_value)


def _config_run_tag(config_path: Path) -> str:
    return config_path.stem.replace("_", "")


def _dataset_input_dir_for_config(
    config_name: str,
    *,
    brats18_dir: Path | None,
    brats23_dir: Path | None,
) -> str | None:
    if config_name.endswith("_18.yaml"):
        return str(brats18_dir) if brats18_dir is not None else None
    if config_name.endswith("_23.yaml"):
        return str(brats23_dir) if brats23_dir is not None else None
    return None


def _replace_yaml_line(
    content: str,
    *,
    key: str,
    value: str | None,
) -> str:
    replacement = f"{key}: {'null' if value is None else value}"
    pattern = re.compile(rf"^{re.escape(key)}:\s*.*$", re.MULTILINE)
    updated, count = pattern.subn(replacement, content, count=1)
    if count != 1:
        raise typer.BadParameter(
            f"Could not update '{key}' in config content",
            param_hint="brainchmark setup",
        )
    return updated


def _update_setup_config(
    *,
    config_path: Path,
    brats18_dir: Path | None,
    brats23_dir: Path | None,
    preprocessed_root_dir: Path,
    artifacts_root_dir: Path,
) -> None:
    content = config_path.read_text(encoding="utf-8")
    run_tag = _config_run_tag(config_path)
    preprocessed_dir = preprocessed_root_dir / f"{run_tag}-preprocessed"
    artifacts_dir = artifacts_root_dir / run_tag
    content = _replace_yaml_line(
        content,
        key="input_dir",
        value=_dataset_input_dir_for_config(
            config_path.name,
            brats18_dir=brats18_dir,
            brats23_dir=brats23_dir,
        ),
    )
    content = _replace_yaml_line(content, key="output_dir", value=str(preprocessed_dir))
    if config_path.name != "preprocessing.yaml":
        content = _replace_yaml_line(content, key="data_dir", value=str(preprocessed_dir))
        content = _replace_yaml_line(content, key="art_dir", value=str(artifacts_dir))
        content = _replace_yaml_line(
            content,
            key="checkpoint_path",
            value=str(artifacts_dir / "checkpoints" / "model_last.pth"),
        )
        content = _replace_yaml_line(
            content,
            key="output_path",
            value=str(artifacts_dir / "results.txt"),
        )
    config_path.write_text(content, encoding="utf-8")


def _copy_config_templates() -> list[Path]:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    template_paths = sorted(CONFIG_TEMPLATES_DIR.glob("*.y*ml"))
    if not template_paths:
        raise typer.BadParameter(
            f"No config templates found under {CONFIG_TEMPLATES_DIR}",
            param_hint="brainchmark setup",
        )

    copied_paths: list[Path] = []
    for template_path in template_paths:
        destination = CONFIGS_DIR / template_path.name
        shutil.copyfile(template_path, destination)
        copied_paths.append(destination)
    return copied_paths


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
    check_update: bool = typer.Option(
        False,
        "--check-update",
        help="Check GitHub for a newer BrainchMark release.",
    ),
    force_refresh: bool = typer.Option(
        False,
        "--force-refresh",
        help="Ignore the cached GitHub result for this check.",
    ),
) -> None:
    """Show the installed BrainchMark version."""
    typer.echo(f"BrainchMark {__version__}")

    if not check_update:
        return

    result = check_for_updates(force_refresh=force_refresh)
    if result.error is not None:
        typer.echo("Could not check GitHub for updates.")
        return

    if result.latest_version is None:
        typer.echo("No published BrainchMark release was found on GitHub.")
        return

    source_label = result.source or "GitHub"
    if result.update_available:
        typer.echo(
            f"Update available from {source_label}: {result.latest_version} "
            f"(current: {result.current_version})"
        )
        return

    typer.echo(
        f"Up to date with latest {source_label}: {result.latest_version}"
    )


@app.command()
def update(
    branch: str | None = typer.Option(
        None,
        "--branch",
        help="Remote branch to update from. Defaults to the current branch.",
    ),
    remote: str = typer.Option(
        "origin",
        "--remote",
        help="Git remote to pull from.",
    ),
    skip_install: bool = typer.Option(
        False,
        "--skip-install",
        help="Skip running install.sh after pulling new commits.",
    ),
) -> None:
    """Fast-forward the local checkout from Git and optionally refresh the environment."""
    try:
        result = update_checkout(
            branch=branch,
            remote=remote,
            run_install=not skip_install,
        )
    except UpdateError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if result.switched_branch:
        typer.echo(f"Switched to branch: {result.branch}")

    if result.changed:
        typer.echo(
            f"Updated {result.branch}: {result.previous_commit[:12]} -> "
            f"{result.current_commit[:12]}"
        )
        if result.install_ran:
            typer.echo("Dependency refresh completed with install.sh.")
        else:
            typer.echo("Skipped dependency refresh.")
        return

    typer.echo(f"Already up to date on {result.branch} ({result.current_commit[:12]}).")


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

    brats23_dir = _prompt_optional_existing_directory(
        label="BraTS23",
        prompt="Directory containing BraTS23",
    )
    brats18_dir = _prompt_optional_existing_directory(
        label="BraTS18",
        prompt="Directory containing BraTS18",
    )
    if brats23_dir is None and brats18_dir is None:
        raise typer.BadParameter(
            "at least one dataset directory must be provided",
            param_hint="brainchmark setup",
        )

    preprocessed_root_dir = _prompt_required_directory(
        label="Preprocessed Root",
        prompt="Root directory for preprocessed datasets",
    )
    artifacts_root_dir = _prompt_required_directory(
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

    updated_files = _copy_config_templates()
    for config_path in updated_files:
        _update_setup_config(
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
    """Preprocess a BraTS-style dataset into BrainchMark `.npz` artifacts."""
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
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(style="white")
    table.add_row("Dataset", str(merged.get("dataset_type")))
    table.add_row("Crop", crop_config.fn.__name__)
    table.add_row("Clamp", clamp_config.fn.__name__)
    table.add_row("Normalize", norm_config.fn.__name__)
    table.add_row("Input", str(merged.get("input_dir")))
    table.add_row("Output", str(merged.get("output_dir")))

    CONSOLE.print(
        Panel(
            table,
            title="[bold green]Preprocessing Start[/bold green]",
            border_style="green",
            expand=False,
        )
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
    distributed: bool = typer.Option(
        False,
        "--distributed",
        help="Relaunch training through torchrun for DDP.",
        rich_help_panel="Runtime",
        is_flag=True,
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

    if distributed and not _distributed_launch_active():
        resolved_nproc_per_node = nproc_per_node
        if resolved_nproc_per_node is None:
            resolved_nproc_per_node = _infer_nproc_per_node()
        if resolved_nproc_per_node <= 0:
            raise typer.BadParameter(
                "--nproc-per-node must be > 0",
                param_hint="--nproc-per-node",
            )
        _relaunch_with_torchrun(resolved_nproc_per_node)

    with CONSOLE.status("[bold cyan]Starting BrainchMark[/bold cyan]", spinner="dots") as status:
        status.update("[bold cyan]Starting BrainchMark[/bold cyan]  [dim]loading training modules[/dim]")
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
            build_model_config,
        )

        status.update("[bold cyan]Starting BrainchMark[/bold cyan]  [dim]reading configuration[/dim]")
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
        merged_loss_kwargs = dict(merged.get("custom_loss_kwargs") or {})
        explicit_loss_kwargs = {
            "num_classes": merged.get("loss_num_classes"),
            "fuse_weight": merged.get("fuse_weight"),
            "sep_weight": merged.get("sep_weight"),
            "prm_weight": merged.get("prm_weight"),
            "eps": merged.get("loss_eps"),
            "log_clamp_min": merged.get("log_clamp_min"),
        }
        for key, value in explicit_loss_kwargs.items():
            if value is not None:
                merged_loss_kwargs[key] = value
        merged["custom_loss_kwargs"] = merged_loss_kwargs
        merged_trainer_kwargs = dict(merged.get("custom_trainer_kwargs") or {})
        split_file_value = merged.get("split_file")
        if split_file_value is None:
            split_file_value = merged_trainer_kwargs.get("split_file")
        resolved_split_file = resolve_split_path(split_file_value)
        merged_trainer_kwargs["split_file"] = str(resolved_split_file)
        if merged.get("transform_kind") is not None:
            merged_trainer_kwargs["transform_kind"] = merged["transform_kind"]
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

        status.update("[bold cyan]Starting BrainchMark[/bold cyan]  [dim]building training objects[/dim]")
        trainer_kind = TrainerKind(merged.get("trainer", TrainerKind.IMFUSE))
        default_model = TrainingModelKind.DCSEG if trainer_kind is TrainerKind.DCSEG else TrainingModelKind.IMFUSE
        default_loss = "dcseg" if trainer_kind is TrainerKind.DCSEG else "imfuse"
        model_kind = TrainingModelKind(merged.get("model", default_model))
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
            loss_kind=merged.get("loss", default_loss),
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
        trainer_class = _resolve_trainer_class(trainer_kind)

        status.update("[bold cyan]Starting BrainchMark[/bold cyan]  [dim]initializing trainer[/dim]")
        trainer_instance = trainer_class(
            input_dir=Path(merged["data_dir"]),
            output_dir=Path(merged["art_dir"]),
            custom_trainer_kwargs=merged_trainer_kwargs,
            model_config=model_config,
            loss_config=loss_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            num_epochs=resolved_num_epochs,
            batch_size=int(merged.get("batch_size", 1)),
            num_workers=int(merged.get("num_workers", 8)),
            fp16=bool(merged.get("fp16", False)),
            resume=bool(merged.get("resume", False)),
            seed=int(merged.get("seed", 69)) if merged.get("seed") is not None else None,
            pretrain=merged.get("pretrain"),
            wandb_project=merged.get("wandb_project"),
            wandb_mode=merged.get("wandb_mode"),
            wandb_run_name=merged.get("wandb_run_name"),
            dataset_type=merged.get("dataset_type")
        )

    trainer_instance.fit()


@app.command()
def test(
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
        shell_complete=_split_shell_complete,
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
