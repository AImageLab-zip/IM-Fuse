from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import typer

from brainchmark.cli_runtime import (
    distributed_launch_active,
    infer_nproc_per_node,
    relaunch_with_torchrun,
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
from brainchmark.utils.cli_overrides import (
    load_yaml_config,
    merge_cli_overrides,
    resolve_split_path,
)
from brainchmark.utils.cli_utils import require_preprocess_values


def build_preprocess_merged_config(
    *,
    config: Path | None,
    input_dir: Path | None,
    output_dir: Path | None,
    dataset_type: DatasetType | None,
    crop_mode: str,
    crop_size: list[int] | None,
    crop_min_size: list[int] | None,
    clamp_mode: str,
    clamp_percentile: list[float] | None,
    clamp_min: list[int] | None,
    clamp_max: list[int] | None,
    norm_mode: str,
    norm_min_max_range: tuple[float, float] | None,
    norm_mean: tuple[float, float, float, float] | None,
    norm_std: tuple[float, float, float, float] | None,
    yes: bool,
) -> dict[str, object]:
    yaml_config = load_yaml_config(config)
    return merge_cli_overrides(
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


def run_preprocess_from_merged(
    merged: dict[str, object],
    *,
    console: Console,
    yes: bool,
) -> None:
    from brainchmark.preprocessing.config import (
        build_clamp_config,
        build_crop_config,
        build_norm_config,
    )
    from brainchmark.preprocessing.pipeline import run_preprocessing
    from brainchmark.datasets.config import DatasetType as PreprocessingDatasetType

    require_preprocess_values(merged, "input_dir", "output_dir", "dataset_type")
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

    console.print(
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
        norm_config=norm_config,
        yes=bool(merged.get("yes", yes)),
    )


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


def build_train_merged_config(
    *,
    config: Path | None,
    data_dir: Path | None,
    art_dir: Path | None,
    trainer: TrainerKind | None,
    model: ModelKind | None,
    loss: str | None,
    custom_model_kwargs: list[str] | None,
    custom_loss_kwargs: list[str] | None,
    loss_num_classes: int | None,
    fuse_weight: float | None,
    sep_weight: float | None,
    prm_weight: float | None,
    loss_eps: float | None,
    log_clamp_min: float | None,
    custom_trainer_kwargs: list[str] | None,
    split_file: Path | None,
    optimizer: OptimizerKind | None,
    betas: tuple[float, float] | None,
    momentum: float | None,
    scheduler: SchedulerKind | None,
    poly_total_iters: int | None,
    poly_power: float | None,
    cosine_t_max: int | None,
    cosine_eta_min: float | None,
    step_step_size: int | None,
    step_gamma: float | None,
    multistep_milestones: list[int] | None,
    multistep_gamma: float | None,
    plateau_mode: str | None,
    plateau_factor: float | None,
    plateau_patience: int | None,
    transform_kind: TransformKind | None,
    lr: float | None,
    num_epochs: int | None,
    batch_size: int | None,
    weight_decay: float | None,
    num_workers: int,
    distributed: bool | None,
    nproc_per_node: int | None,
    fp16: bool | None,
    resume: bool,
    seed: int,
    pretrain: Path | None,
    wandb_project: str | None,
    wandb_mode: str | None,
    wandb_run_name: str | None,
    dataset_type: DatasetType | None,
) -> dict[str, object]:
    from brainchmark.training.config import parse_kv_list

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
        distributed=distributed,
        nproc_per_node=nproc_per_node,
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
    return merged


def maybe_relaunch_distributed(merged: dict[str, object]) -> None:
    if not bool(merged.get("distributed")) or distributed_launch_active():
        return

    resolved_nproc_per_node = merged.get("nproc_per_node")
    if resolved_nproc_per_node is None:
        resolved_nproc_per_node = infer_nproc_per_node()
    resolved_nproc_per_node = int(resolved_nproc_per_node)
    if resolved_nproc_per_node <= 0:
        raise typer.BadParameter(
            "--nproc-per-node must be > 0",
            param_hint="--nproc-per-node",
        )
    relaunch_with_torchrun(resolved_nproc_per_node)


def run_train_from_merged(merged: dict[str, object]) -> None:
    from brainchmark.training.config import (
        OptimizerKind as TrainingOptimizerKind,
        SchedulerKind as TrainingSchedulerKind,
        build_optimizer_config,
        build_scheduler_config,
    )
    from brainchmark.losses.config import build_loss_config
    from brainchmark.models.config import (
        ModelKind as TrainingModelKind,
        build_model_config,
    )
    from brainchmark.training.trainers import DCSegTrainer, IMFuseTrainer

    _require_train_values(
        merged,
        "data_dir",
        "art_dir",
        "trainer",
        "optimizer",
        "num_epochs",
    )

    trainer_kind = TrainerKind(merged.get("trainer", TrainerKind.IMFUSE))
    trainer_map = {
        TrainerKind.IMFUSE: IMFuseTrainer,
        TrainerKind.DCSEG: DCSegTrainer,
    }
    try:
        trainer_class = trainer_map[trainer_kind]
    except KeyError as exc:
        raise typer.BadParameter(
            f"Unsupported trainer: {trainer_kind}",
            param_hint="--trainer",
        ) from exc

    default_model = (
        TrainingModelKind.DCSEG
        if trainer_kind is TrainerKind.DCSEG
        else TrainingModelKind.IMFUSE
    )
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
    trainer_instance = trainer_class(
        input_dir=Path(merged["data_dir"]),
        output_dir=Path(merged["art_dir"]),
        custom_trainer_kwargs=dict(merged.get("custom_trainer_kwargs") or {}),
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
        dataset_type=merged.get("dataset_type"),
    )
    trainer_instance.fit()
