# Standard library
from pathlib import Path

# External dependencies
from rich.console import Console
from rich.text import Text
import typer

# Internal modules
from brainchmark.utils.cli_utils import require_preprocess_values
from brainchmark.preprocessing.config import (
    ClampMode,
    CropMode,
    NormMode,
    build_clamp_config,
    build_crop_config,
    build_norm_config,
)
from brainchmark.datasets.config import DatasetType
from brainchmark.preprocessing.pipeline import run_preprocessing
from brainchmark.training import IMFuseTrainer
from brainchmark.training.config import (
    DEFAULT_TEST_TRANSFORMS,
    DEFAULT_TRAIN_TRANSFORMS,
    IMFuseTrainingConfig,
    OptimizerKind,
    SchedulerKind,
    TrainerKind,
    WandbConfig,
)
from brainchmark.utils.cli_overrides import load_yaml_config, merge_cli_overrides

app = typer.Typer(help="BrainchMark CLI")
console = Console()


def _require_train_values(merged: dict[str, object], *required_keys: str) -> None:
    for key in required_keys:
        if merged.get(key) is None:
            raise typer.BadParameter(
                "missing value; provide it in the CLI or in --config",
                param_hint=f"--{key.replace('_', '-')}",
            )


@app.callback()
def main() -> None:
    """BrainchMark command group."""


@app.command()
def hello() -> None:
    """Simple test command."""
    message = "Hello from the AImageLab Team!"

    text = Text(message, style="bold magenta")
    text.stylize("bold cyan", 0, 5)      # "Hello"
    text.stylize("bold yellow", 15, 24)  # "AImageLab"

    console.print()
    console.print(text, justify="center")
    console.print()


@app.command()
def preprocess(
    config: Path | None = typer.Option(
        None,
        "--config",
        file_okay=True,
        dir_okay=False,
        exists=True,
        readable=True,
        help="Path to a YAML config file.",
    ),
    input_dir: Path = typer.Option(
        None,
        "--input-dir",
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        help="Directory containing the input data to preprocess.",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        help="Directory where preprocessed data will be written.",
    ),
    dataset_type: DatasetType = typer.Option(
        None,
        "--dataset-type",
        help="Input dataset type. Choose either brats18 or brats23",
    ),
    crop_mode: CropMode = typer.Option(
        CropMode.NONE,
        "--crop-mode",
        help= "Cropping strategy. One of: " + ", ".join(mode.value for mode in CropMode) + ".",
    ),
    crop_size: list[int] | None | None = typer.Option(
        None,
        "--crop-size",
        help="Center crop size as three integers: X Y Z.",
    ),
    crop_min_size: list[int] | None | None = typer.Option(
        None,
        "--crop-min-size",
        help="Minimum non-empty crop size as three integers: X Y Z.",
    ),
    clamp_mode: ClampMode = typer.Option(
        ClampMode.NONE,
        "--clamp-mode",
        help="Clamp mode. One of: " + ", ".join(mode.value for mode in ClampMode) + ".",
    ),

    clamp_percentile: list[float] | None = typer.Option(
        None,
        "--clamp-percentile",
        help="Clamp percentiles as one or two floats: HIGH or LOW HIGH. If one value is given, LOW is assumed to be 0.",
    ),

    clamp_min: list[int] | None = typer.Option(
        None,
        "--clamp-min",
        help="Minimum clamp values as one or four integers. If one value is given, it will be used for all modalities.",

    ),

    clamp_max: list[int] | None = typer.Option(
        None,
        "--clamp-max",
        help="Maximum clamp values as one or four integers. If one value is given, it will be used for all modalities.",

    ),

    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Automatically answer yes to prompts",
    ),
    norm_mode: NormMode = typer.Option(
        NormMode.NONE,
        "--norm-mode",
        help="Normalization mode. One of: " + ", ".join(mode.value for mode in NormMode) + ".",
    ),
    norm_min_max_range: tuple[float, float] | None = typer.Option(
        None,
        "--norm-min-max-range",
        help="Target min-max normalization range as two floats: MIN MAX.",
    ),

    norm_mean: tuple[float, float, float, float] | None = typer.Option(
        None,
        "--norm-mean",
        help="Normalization means as four floats, one for each modality.",
    ),

    norm_std: tuple[float, float, float, float] | None = typer.Option(
        None,
        "--norm-std",
        help="Normalization standard deviations as four floats, one for each modality.",
    ),

) -> None:
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
        crop_mode=CropMode(merged.get("crop_mode", CropMode.NONE)),
        crop_size=merged.get("crop_size"),
        crop_min_size=merged.get("crop_min_size"),
    )
    clamp_config = build_clamp_config(
        clamp_mode=ClampMode(merged.get("clamp_mode", ClampMode.NONE)),
        clamp_percentile=merged.get("clamp_percentile"),
        clamp_min=merged.get("clamp_min"),
        clamp_max=merged.get("clamp_max"),
    )
    norm_config = build_norm_config(
        norm_mode=NormMode(merged.get("norm_mode", NormMode.NONE)),
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
        dataset_type=merged.get("dataset_type"),
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
        exists=True,
        readable=True,
        help="Path to a YAML config file.",
    ),
    input_dir: Path | None = typer.Option(
        None,
        "--input-dir",
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        help="Directory containing the training data.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        help="Directory where training artifacts will be written.",
    ),
    trainer: TrainerKind | None = typer.Option(
        None,
        "--trainer",
        help="Trainer implementation or preset to use.",
    ),
    dataname: str | None = typer.Option(
        None,
        "--dataname",
        help="Dataset name, for example BRATS2018 or BRATS2023.",
    ),
    optimizer: OptimizerKind | None = typer.Option(
        None,
        "--optimizer",
        help="Optimizer name.",
    ),
    scheduler: SchedulerKind | None = typer.Option(
        None,
        "--scheduler",
        help="Learning-rate scheduler name.",
    ),
    train_transforms: str | None = typer.Option(
        None,
        "--train-transforms",
        help="Training transform pipeline expression.",
    ),
    test_transforms: str | None = typer.Option(
        None,
        "--test-transforms",
        help="Validation/test transform pipeline expression.",
    ),
    lr: float | None = typer.Option(
        None,
        "--lr",
        help="Learning rate.",
    ),
    num_epochs: int | None = typer.Option(
        None,
        "--num-epochs",
        help="Number of training epochs.",
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        help="Mini-batch size.",
    ),
    weight_decay: float | None = typer.Option(
        None,
        "--weight-decay",
        help="Weight decay.",
    ),
    region_fusion_start_epoch: int | None = typer.Option(
        None,
        "--region-fusion-start-epoch",
        help="Epoch at which fused-region loss starts contributing.",
    ),
    num_workers: int | None = typer.Option(
        None,
        "--num-workers",
        help="Number of dataloader workers.",
    ),
    resume: Path | None = typer.Option(
        None,
        "--resume",
        file_okay=True,
        dir_okay=False,
        exists=True,
        readable=True,
        help="Checkpoint path to resume training from.",
    ),
    pretrain: Path | None = typer.Option(
        None,
        "--pretrain",
        file_okay=True,
        dir_okay=False,
        exists=True,
        readable=True,
        help="Checkpoint path to use as pretrained initialization.",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Random seed.",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Device identifier, for example cpu, cuda, or cuda:0.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Run a shortened debug training loop.",
    ),
    interleaved_tokenization: bool = typer.Option(
        False,
        "--interleaved-tokenization",
        help="Enable interleaved tokenization in IMFuse.",
    ),
    mamba_skip: bool = typer.Option(
        False,
        "--mamba-skip",
        help="Enable mamba-based skip fusion.",
    ),
    first_skip: bool = typer.Option(
        False,
        "--first-skip",
        help="Use the first-skip IMFuse variant instead of the no-1-skip variant.",
    ),
    wandb_project: str | None = typer.Option(
        None,
        "--wandb-project",
        help="Weights & Biases project name.",
    ),
    wandb_mode: str | None = typer.Option(
        None,
        "--wandb-mode",
        help="Weights & Biases mode, for example online, offline, or disabled.",
    ),
) -> None:
    """Run training from CLI overrides and YAML configuration."""
    yaml_config = load_yaml_config(config)
    merged = merge_cli_overrides(
        yaml_config,
        input_dir=input_dir,
        output_dir=output_dir,
        trainer=trainer,
        dataname=dataname,
        optimizer=optimizer,
        scheduler=scheduler,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        lr=lr,
        num_epochs=num_epochs,
        batch_size=batch_size,
        weight_decay=weight_decay,
        region_fusion_start_epoch=region_fusion_start_epoch,
        num_workers=num_workers,
        resume=resume,
        pretrain=pretrain,
        seed=seed,
        device=device,
        debug=debug if debug else yaml_config.get("debug"),
        interleaved_tokenization=interleaved_tokenization if interleaved_tokenization else yaml_config.get("interleaved_tokenization"),
        mamba_skip=mamba_skip if mamba_skip else yaml_config.get("mamba_skip"),
        first_skip=first_skip if first_skip else yaml_config.get("first_skip"),
        wandb_project=wandb_project,
        wandb_mode=wandb_mode,
    )
    _require_train_values(
        merged,
        "input_dir",
        "output_dir",
        "trainer",
        "optimizer",
        "num_epochs",
    )
    train_config = IMFuseTrainingConfig(
        input_dir=Path(merged["input_dir"]),
        output_dir=Path(merged["output_dir"]),
        trainer=TrainerKind(merged.get("trainer", TrainerKind.IMFUSE)),
        dataname=str(merged.get("dataname", "BRATS2018")),
        optimizer=OptimizerKind(merged.get("optimizer", OptimizerKind.RADAM)),
        scheduler=SchedulerKind(merged.get("scheduler", SchedulerKind.POLY)),
        batch_size=int(merged.get("batch_size", 1)),
        lr=float(merged.get("lr", 2e-4)),
        weight_decay=float(merged.get("weight_decay", 3e-5)),
        num_epochs=int(merged["num_epochs"]),
        num_workers=int(merged.get("num_workers", 8)),
        region_fusion_start_epoch=int(merged.get("region_fusion_start_epoch", 0)),
        seed=int(merged.get("seed", 999)),
        resume=Path(merged["resume"]) if merged.get("resume") is not None else None,
        pretrain=Path(merged["pretrain"]) if merged.get("pretrain") is not None else None,
        debug=bool(merged.get("debug", False)),
        interleaved_tokenization=bool(merged.get("interleaved_tokenization", False)),
        mamba_skip=bool(merged.get("mamba_skip", False)),
        first_skip=bool(merged.get("first_skip", False)),
        device=str(merged["device"]) if merged.get("device") is not None else None,
        train_transforms=str(merged.get("train_transforms")) if merged.get("train_transforms") is not None else DEFAULT_TRAIN_TRANSFORMS,
        test_transforms=str(merged.get("test_transforms")) if merged.get("test_transforms") is not None else DEFAULT_TEST_TRANSFORMS,
        wandb=WandbConfig(
            enabled=str(merged.get("wandb_mode", "online")) != "disabled",
            project=str(merged.get("wandb_project", "SegmentationMM")),
            mode=str(merged.get("wandb_mode", "online")),
        ),
    )

    typer.echo(
        "Training "
        f"{train_config.trainer.value} on {train_config.dataname} "
        f"from {train_config.input_dir} into {train_config.output_dir} "
        f"with optimizer={train_config.optimizer.value}, "
        f"scheduler={train_config.scheduler.value}, "
        f"lr={train_config.lr}, epochs={train_config.num_epochs}."
    )
    IMFuseTrainer(train_config).fit()
