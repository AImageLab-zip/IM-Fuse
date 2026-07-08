from pathlib import Path
import re

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import typer

from mimose.cli_runtime import (
    distributed_launch_active,
    infer_nproc_per_node,
    relaunch_with_torchrun,
)
from mimose.enums import (
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
from mimose.utils.cli_overrides import (
    apply_run_suffix,
    load_yaml_config,
    merge_cli_overrides,
    resolve_split_path,
)
from mimose.utils.cli_utils import require_preprocess_values


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
    from mimose.preprocessing.config import (
        build_clamp_config,
        build_crop_config,
        build_norm_config,
    )
    from mimose.preprocessing.pipeline import run_preprocessing

    require_preprocess_values(merged, "input_dir", "output_dir")
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


def _require_existing_data_dir(merged: dict[str, object]) -> None:
    data_dir = merged.get("data_dir")
    if data_dir is None:
        return

    resolved = Path(data_dir)
    if not resolved.is_dir():
        raise typer.BadParameter(
            f"dataset directory not found: {resolved}",
            param_hint="--data-dir",
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
    custom_trainer_kwargs: list[str] | None,
    split_file: Path | None,
    optimizer: OptimizerKind | None,
    betas: tuple[float, float] | None,
    momentum: float | None,
    nesterov: bool | None,
    amsgrad: bool | None,
    scheduler: SchedulerKind | None,
    poly_total_iters: int | None,
    poly_power: float | None,
    warmuppoly_warmup_iters: int | None,
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
    validation_every: int | None,
    batch_size: int | None,
    weight_decay: float | None,
    num_workers: int,
    distributed: bool | None,
    nproc_per_node: int | None,
    fp16: bool | None,
    compile: bool | None,
    resume: bool,
    try_resume: bool,
    seed: int,
    pretrain: Path | None,
    wandb_project: str | None,
    wandb_mode: str | None,
    wandb_run_name: str | None,
    dataset_type: DatasetType | None,
    push_to_hf: bool | None,
    hf_repo: str | None,
    run_suffix: str | None = None,
) -> dict[str, object]:
    from mimose.training.config import parse_kv_list

    yaml_config = load_yaml_config(config)
    model_kwargs = (
        parse_kv_list(custom_model_kwargs)
        if custom_model_kwargs is not None
        else None
    )
    loss_kwargs = (
        parse_kv_list(custom_loss_kwargs)
        if custom_loss_kwargs is not None
        else None
    )
    trainer_kwargs = (
        parse_kv_list(custom_trainer_kwargs)
        if custom_trainer_kwargs is not None
        else None
    )
    merged = merge_cli_overrides(
        yaml_config,
        data_dir=data_dir,
        art_dir=art_dir,
        trainer=trainer,
        model=model,
        loss=loss,
        custom_model_kwargs=model_kwargs,
        custom_loss_kwargs=loss_kwargs,
        custom_trainer_kwargs=trainer_kwargs,
        split_file=split_file,
        optimizer=optimizer,
        betas=betas,
        momentum=momentum,
        nesterov=nesterov,
        amsgrad=amsgrad,
        scheduler=scheduler,
        poly_total_iters=poly_total_iters,
        poly_power=poly_power,
        warmuppoly_warmup_iters=warmuppoly_warmup_iters,
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
        validation_every=validation_every,
        batch_size=batch_size,
        weight_decay=weight_decay,
        num_workers=num_workers,
        distributed=distributed,
        nproc_per_node=nproc_per_node,
        fp16=fp16,
        compile=compile,
        resume=resume,
        try_resume=try_resume,
        pretrain=pretrain,
        seed=seed,
        wandb_project=wandb_project,
        wandb_mode=wandb_mode,
        wandb_run_name=wandb_run_name,
        dataset_type=dataset_type,
        push_to_hf=push_to_hf,
        hf_repo=hf_repo,
        run_suffix=run_suffix,
    )
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
    merged["push_to_hf"] = bool(merged.get("push_to_hf", False) or merged.get("hf_repo"))
    if merged["push_to_hf"] and merged.get("hf_repo") is None:
        raise typer.BadParameter(
            "missing value; provide it in the CLI or in --config",
            param_hint="--hf-repo",
        )
    _require_existing_data_dir(merged)
    apply_run_suffix(merged)
    return merged


def _validate_hf_repo_id(hf_repo: str, *, param_hint: str = "--hf-repo") -> None:
    if re.match(r"^https?://", hf_repo.strip()):
        raise typer.BadParameter(
            "expected a Hugging Face repo id like 'namespace/repo', not a full URL",
            param_hint=param_hint,
        )


def build_push_merged_config(
    *,
    config: Path | None,
    art_dir: Path | None,
    checkpoint_path: Path | None,
    trainer: TrainerKind | None,
    model: ModelKind | None,
    custom_model_kwargs: list[str] | None,
    custom_trainer_kwargs: list[str] | None,
    num_workers: int,
    seed: int,
    wandb_run_name: str | None,
    dataset_type: DatasetType | None,
    hf_repo: str | None,
    run_suffix: str | None = None,
) -> dict[str, object]:
    merged = build_train_merged_config(
        config=config,
        data_dir=None,
        art_dir=art_dir,
        trainer=trainer,
        model=model,
        loss=None,
        custom_model_kwargs=custom_model_kwargs,
        custom_loss_kwargs=None,
        custom_trainer_kwargs=custom_trainer_kwargs,
        split_file=None,
        optimizer=None,
        betas=None,
        momentum=None,
        nesterov=None,
        amsgrad=None,
        scheduler=None,
        poly_total_iters=None,
        poly_power=None,
        warmuppoly_warmup_iters=None,
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
        validation_every=None,
        batch_size=None,
        weight_decay=None,
        num_workers=num_workers,
        distributed=False,
        nproc_per_node=None,
        fp16=None,
        compile=None,
        resume=False,
        try_resume=False,
        seed=seed,
        pretrain=None,
        wandb_project=None,
        wandb_mode=None,
        wandb_run_name=wandb_run_name,
        dataset_type=dataset_type,
        push_to_hf=True,
        hf_repo=hf_repo,
        run_suffix=run_suffix,
    )
    if checkpoint_path is not None:
        merged["checkpoint_path"] = str(checkpoint_path)
    _require_train_values(
        merged,
        "art_dir",
        "trainer",
        "model",
        "data_dir",
        "hf_repo",
        "wandb_run_name",
    )
    _validate_hf_repo_id(str(merged["hf_repo"]))
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
    trainer_instance = _build_trainer_instance_from_merged(merged)
    trainer_instance.fit()


def _resolve_push_checkpoint_path(merged: dict[str, object]) -> Path:
    checkpoint_path = Path(
        merged.get("checkpoint_path")
        or (Path(merged["art_dir"]) / "checkpoints" / "final_weights_only.safetensors")
    )
    if not checkpoint_path.is_file():
        raise typer.BadParameter(
            f"checkpoint not found at {checkpoint_path}",
            param_hint="--checkpoint-path",
        )
    return checkpoint_path


def run_push_from_merged(merged: dict[str, object]) -> Path:
    trainer_instance = _build_trainer_instance_from_merged(merged)
    checkpoint_path = _resolve_push_checkpoint_path(merged)
    return trainer_instance.push_checkpoint_to_hf(checkpoint_path)


# Seeds trained by sbatcher_18.sh / sbatcher_23.sh for every model (seed 67 is a
# standalone extra run and is intentionally excluded from the bulk push too).
PUSH_ALL_SEEDS: tuple[int, ...] = (0, 42, 69)

# Seed whose checkpoint is re-uploaded a second time under the run's unsuffixed
# name, so testers can grab a default checkpoint without knowing which seed to ask for.
PUSH_DEFAULT_SEED = 0


def run_push_all_seeds_from_config(
    *,
    config: Path | None,
    art_dir: Path | None,
    trainer: TrainerKind | None,
    model: ModelKind | None,
    custom_model_kwargs: list[str] | None,
    custom_trainer_kwargs: list[str] | None,
    num_workers: int,
    wandb_run_name: str | None,
    dataset_type: DatasetType | None,
    hf_repo: str | None,
) -> list[Path]:
    export_dirs: list[Path] = []
    default_checkpoint_path: Path | None = None

    for seed in PUSH_ALL_SEEDS:
        merged = build_push_merged_config(
            config=config,
            art_dir=art_dir,
            checkpoint_path=None,
            trainer=trainer,
            model=model,
            custom_model_kwargs=custom_model_kwargs,
            custom_trainer_kwargs=custom_trainer_kwargs,
            num_workers=num_workers,
            seed=seed,
            wandb_run_name=wandb_run_name,
            dataset_type=dataset_type,
            hf_repo=hf_repo,
            run_suffix=f"seed{seed}",
        )
        export_dirs.append(run_push_from_merged(merged))
        if seed == PUSH_DEFAULT_SEED:
            default_checkpoint_path = _resolve_push_checkpoint_path(merged)

    assert default_checkpoint_path is not None
    default_merged = build_push_merged_config(
        config=config,
        art_dir=art_dir,
        checkpoint_path=default_checkpoint_path,
        trainer=trainer,
        model=model,
        custom_model_kwargs=custom_model_kwargs,
        custom_trainer_kwargs=custom_trainer_kwargs,
        num_workers=num_workers,
        seed=PUSH_DEFAULT_SEED,
        wandb_run_name=wandb_run_name,
        dataset_type=dataset_type,
        hf_repo=hf_repo,
        run_suffix=None,
    )
    export_dirs.append(run_push_from_merged(default_merged))
    return export_dirs


def _build_trainer_instance_from_merged(merged: dict[str, object]):
    from mimose.training.config import (
        OptimizerKind as TrainingOptimizerKind,
        SchedulerKind as TrainingSchedulerKind,
        build_optimizer_config,
        build_scheduler_config,
    )
    from mimose.losses.config import build_loss_config
    from mimose.models.config import (
        ModelKind as TrainingModelKind,
        build_model_config,
    )
    from mimose.training.trainers import (
        DCSegTrainer,
        IMFuseTrainer,
        IMS2TransTrainer,
        LCKDTrainer,
        M3AETrainer,
        MaMTrainer,
        MIFPNTrainer,
        MSTKDTrainer,
        ReverseTrainer,
        RobustSegTrainer,
        ShaSpecTrainer,
        SRMNetTrainer,
        UHVEDTrainer,
    )
    from mimose.training.transforms import build_transform_manager

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
        TrainerKind.UHVED: UHVEDTrainer,
        TrainerKind.ROBUSTSEG: RobustSegTrainer,
        TrainerKind.SHASPEC: ShaSpecTrainer,
        TrainerKind.M3AE: M3AETrainer,
        TrainerKind.MAM: MaMTrainer,
        TrainerKind.SRMNET: SRMNetTrainer,
        TrainerKind.IMS2TRANS: IMS2TransTrainer,
        TrainerKind.MSTKDNET: MSTKDTrainer,
        TrainerKind.MIFPN: MIFPNTrainer,
        TrainerKind.REVERSE: ReverseTrainer,
        TrainerKind.LCKD: LCKDTrainer,
    }
    try:
        trainer_class = trainer_map[trainer_kind]
    except KeyError as exc:
        raise typer.BadParameter(
            f"Unsupported trainer: {trainer_kind}",
            param_hint="--trainer",
        ) from exc

    default_model_map = {
        TrainerKind.DCSEG: TrainingModelKind.DCSEG,
        TrainerKind.UHVED: TrainingModelKind.UHVED,
        TrainerKind.ROBUSTSEG: TrainingModelKind.ROBUSTSEG,
        TrainerKind.SHASPEC: TrainingModelKind.SHASPEC,
        TrainerKind.M3AE: TrainingModelKind.M3AE,
        TrainerKind.MAM: TrainingModelKind.MAM,
        TrainerKind.SRMNET: TrainingModelKind.SRMNET,
        TrainerKind.IMS2TRANS: TrainingModelKind.IMS2TRANS,
        TrainerKind.MSTKDNET: TrainingModelKind.MSTKDNET,
        TrainerKind.MIFPN: TrainingModelKind.MIFPN,
        TrainerKind.REVERSE: TrainingModelKind.REVERSE,
        TrainerKind.LCKD: TrainingModelKind.LCKD,
    }
    default_loss_map = {
        TrainerKind.UHVED: "uhved",
        TrainerKind.ROBUSTSEG: "robustseg",
        TrainerKind.SHASPEC: "shaspec",
        TrainerKind.M3AE: "m3ae",
        TrainerKind.MAM: "mam",
        TrainerKind.SRMNET: "srmnet",
        TrainerKind.IMS2TRANS: "ims2trans",
        TrainerKind.MSTKDNET: "mstkdnet",
        TrainerKind.MIFPN: "mifpn",
        TrainerKind.REVERSE: "reverse",
        TrainerKind.LCKD: "lckd",
    }
    default_model = default_model_map.get(trainer_kind, TrainingModelKind.IMFUSE)
    default_loss = default_loss_map.get(trainer_kind, "imfuse")
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
        nesterov=bool(_merged_value(merged, "nesterov", False)),
        amsgrad=bool(merged["amsgrad"]) if merged.get("amsgrad") is not None else None,
    )
    scheduler_config = build_scheduler_config(
        scheduler_kind=scheduler_kind,
        poly_total_iters=int(merged["poly_total_iters"]) if merged.get("poly_total_iters") is not None else resolved_num_epochs,
        poly_power=float(_merged_value(merged, "poly_power", 0.9)),
        warmuppoly_warmup_iters=int(_merged_value(merged, "warmuppoly_warmup_iters", 100)),
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
    trainer_kwargs = dict(merged.get("custom_trainer_kwargs") or {})
    default_transform_kind = TransformKind.IMFUSE
    resolved_transform_kind = TransformKind(
        trainer_kwargs.get(
            "transform_kind",
            merged.get("transform_kind", default_transform_kind),
        )
    )
    if (
        model_kind == TrainingModelKind.TINYMIMOSA
        and resolved_transform_kind != TransformKind.TINYMIMOSA
    ):
        raise typer.BadParameter(
            "TinyMimosa requires transform_kind=tinymimosa in config/overrides.\n"
            f"resolved model={model_kind.value}\n"
            f"resolved transform_kind={resolved_transform_kind.value}\n"
            f"custom_trainer_kwargs={trainer_kwargs}",
            param_hint="--transform-kind",
        )
    trainer_kwargs["transform_kind"] = resolved_transform_kind.value
    patch_size = trainer_kwargs.get("patch_size")
    crop_size = (int(patch_size),) * 3 if patch_size is not None else None
    transform_manager = build_transform_manager(
        resolved_transform_kind,
        model_kwargs=model_config.kwargs,
        crop_size=crop_size,
    )
    trainer_instance = trainer_class(
        input_dir=Path(merged["data_dir"]),
        output_dir=Path(merged["art_dir"]),
        custom_trainer_kwargs=trainer_kwargs,
        model_config=model_config,
        loss_config=loss_config,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        transform_manager=transform_manager,
        num_epochs=resolved_num_epochs,
        validation_every=int(_merged_value(merged, "validation_every", 1)),
        batch_size=int(merged.get("batch_size", 1)),
        num_workers=int(merged.get("num_workers", 8)),
        fp16=bool(merged.get("fp16", False)),
        compile=bool(merged.get("compile", False)),
        resume=bool(merged.get("resume", False)),
        try_resume=bool(merged.get("try_resume", False)),
        seed=int(merged.get("seed", 69)) if merged.get("seed") is not None else None,
        pretrain=merged.get("pretrain"),
        wandb_project=merged.get("wandb_project"),
        wandb_mode=merged.get("wandb_mode"),
        wandb_run_name=merged.get("wandb_run_name"),
        dataset_type=merged.get("dataset_type"),
        push_to_hf=bool(merged.get("push_to_hf", False)),
        hf_repo=merged.get("hf_repo"),
    )
    return trainer_instance
