from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Type, Any
import ast

import brainchmark.training.scheduling as custom_scheduling

import typer

from torch.optim import Adam,RAdam,AdamW,SGD, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    MultiStepLR,
    PolynomialLR,
    ReduceLROnPlateau,
    StepLR,
)

DEFAULT_TRAIN_TRANSFORMS = (
    "Compose([RandCrop3D((128,128,128)), RandomRotion(10), "
    "RandomIntensityChange((0.1,0.1)), RandomFlip(0), "
    "NumpyType((np.float32, np.int64)),])"
)
DEFAULT_TEST_TRANSFORMS = "Compose([NumpyType((np.float32, np.int64)),])"

class OptimizerKind(StrEnum):
    RADAM = "radam"
    ADAMW = "adamw"
    SGD = "sgd"
    ADAM = "adam"


OPTIMIZER_DICT = {
    OptimizerKind.RADAM: RAdam,
    OptimizerKind.ADAMW: AdamW,
    OptimizerKind.ADAM:Adam,
    OptimizerKind.SGD:SGD
}

@dataclass(frozen=True)
class OptimizerConfig:
    optim_class: Type[Optimizer]
    lr: float
    weight_decay: float
    betas: tuple[float, float] | None = None
    momentum: float | None = None
    eps: float | None = None

def build_optimizer_config(
    optimizer_kind: OptimizerKind,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] | None,
    momentum: float,
) -> OptimizerConfig:
    if lr <= 0:
        raise ValueError("lr must be > 0")
    if weight_decay < 0:
        raise ValueError("weight_decay must be >= 0")
    if optimizer_kind is OptimizerKind.SGD and momentum < 0:
        raise ValueError("momentum must be >= 0")

    if optimizer_kind in {OptimizerKind.RADAM, OptimizerKind.ADAMW, OptimizerKind.ADAM}:
        if betas is None or len(betas) != 2:
            raise ValueError("betas must contain exactly two values")

        beta1, beta2 = betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("betas must be in the range [0, 1)")

        return OptimizerConfig(
            optim_class=OPTIMIZER_DICT[optimizer_kind],
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=1e-6,
        )
    if optimizer_kind is OptimizerKind.SGD:
        return OptimizerConfig(
            optim_class=OPTIMIZER_DICT[optimizer_kind],
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    raise typer.BadParameter(
        f"Unsupported optimizer: {optimizer_kind}",
        param_hint="--optimizer",
    )


class SchedulerKind(StrEnum):
    POLY = "poly"
    COSINE = "cosine"
    STEP = "step"
    MULTISTEP = "multistep"
    PLATEAU = "plateau"


@dataclass(frozen=True)
class SchedulerConfig:
    scheduler_class: type[LRScheduler] | type
    kwargs: dict[str, Any] = field(default_factory=dict)


SCHEDULER_DICT = {
    SchedulerKind.POLY: PolynomialLR,
    SchedulerKind.COSINE: CosineAnnealingLR,
    SchedulerKind.STEP: StepLR,
    SchedulerKind.MULTISTEP: MultiStepLR,
    SchedulerKind.PLATEAU: ReduceLROnPlateau,
}


def parse_kv_list(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise typer.BadParameter(
                f"Invalid value '{item}', expected key=value format.",
            )
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()

        try:
            out[key] = ast.literal_eval(value)
        except Exception:
            out[key] = value

    return out


def _resolve_custom_scheduler(module: ModuleType, scheduler_name: str) -> type:
    candidates = (
        scheduler_name,
        scheduler_name.upper(),
        scheduler_name.capitalize(),
        "".join(part.capitalize() for part in scheduler_name.split("_")),
    )

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)

        scheduler_class = getattr(module, candidate, None)
        if isinstance(scheduler_class, type):
            return scheduler_class

    raise typer.BadParameter(
        f"Unsupported scheduler: {scheduler_name}",
        param_hint="--scheduler",
    )


def build_scheduler_config(
    scheduler_kind: SchedulerKind | str,
    *,
    poly_total_iters: int | None,
    poly_power: float,
    cosine_t_max: int | None,
    cosine_eta_min: float,
    step_step_size: int | None,
    step_gamma: float,
    multistep_milestones: list[int] | None,
    multistep_gamma: float,
    plateau_mode: str,
    plateau_factor: float,
    plateau_patience: int,
    custom_scheduler_kwargs: dict[str, Any] | None = None,
) -> SchedulerConfig:
    if isinstance(scheduler_kind, SchedulerKind):
        scheduler_name = scheduler_kind.value
    else:
        scheduler_name = str(scheduler_kind).strip()

    try:
        known_scheduler = SchedulerKind(scheduler_name)
    except ValueError:
        known_scheduler = None

    if known_scheduler is SchedulerKind.POLY:
        if poly_total_iters is None:
            raise typer.BadParameter(
                "--poly-total-iters is required for poly",
                param_hint="--poly-total-iters",
            )
        return SchedulerConfig(
            scheduler_class=SCHEDULER_DICT[SchedulerKind.POLY],
            kwargs={
                "total_iters": poly_total_iters,
                "power": poly_power,
            },
        )

    if known_scheduler is SchedulerKind.COSINE:
        if cosine_t_max is None:
            raise typer.BadParameter(
                "--cosine-t-max is required for cosine",
                param_hint="--cosine-t-max",
            )
        return SchedulerConfig(
            scheduler_class=SCHEDULER_DICT[SchedulerKind.COSINE],
            kwargs={
                "T_max": cosine_t_max,
                "eta_min": cosine_eta_min,
            },
        )

    if known_scheduler is SchedulerKind.STEP:
        if step_step_size is None:
            raise typer.BadParameter(
                "--step-step-size is required for step",
                param_hint="--step-step-size",
            )
        return SchedulerConfig(
            scheduler_class=SCHEDULER_DICT[SchedulerKind.STEP],
            kwargs={
                "step_size": step_step_size,
                "gamma": step_gamma,
            },
        )

    if known_scheduler is SchedulerKind.MULTISTEP:
        if not multistep_milestones:
            raise typer.BadParameter(
                "--multistep-milestones is required for multistep",
                param_hint="--multistep-milestones",
            )
        return SchedulerConfig(
            scheduler_class=SCHEDULER_DICT[SchedulerKind.MULTISTEP],
            kwargs={
                "milestones": multistep_milestones,
                "gamma": multistep_gamma,
            },
        )

    if known_scheduler is SchedulerKind.PLATEAU:
        return SchedulerConfig(
            scheduler_class=SCHEDULER_DICT[SchedulerKind.PLATEAU],
            kwargs={
                "mode": plateau_mode,
                "factor": plateau_factor,
                "patience": plateau_patience,
            },
        )

    scheduler_class = _resolve_custom_scheduler(custom_scheduling, scheduler_name)
    return SchedulerConfig(
        scheduler_class=scheduler_class,
        kwargs=custom_scheduler_kwargs or {},
    )


class TrainerKind(StrEnum):
    IMFUSE = "imfuse"




@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = True
    project: str = "SegmentationMM"
    run_name: str | None = "training"
    mode: str = "online"
    resume: str = "allow"
