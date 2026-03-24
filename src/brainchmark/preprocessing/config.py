from dataclasses import dataclass,field
from enum import StrEnum
import typer
from typing import Callable
import importlib

class CropMode(StrEnum):
    NONE = "none"
    CENTER = "center"
    NON_EMPTY = "non_empty"

@dataclass(frozen=True)
class CropConfig:
    fn: Callable
    size: tuple[int, int, int] | None = None
    min_size: tuple[int, int, int] | None = None

def build_crop_config(
    crop_mode: CropMode,
    crop_size: tuple[int, int, int] | None,
    crop_min_size: tuple[int, int, int] | None,
) -> CropConfig:
    """Validate crop-related CLI options and return a structured config."""
    module = importlib.import_module("brainchmark.preprocessing.cropping")

    fn = getattr(module, crop_mode.value, None)

    if fn is None or not callable(fn):
        raise typer.BadParameter(
            f"Unknown crop mode {crop_mode!r}: function {crop_mode!r} "
            f"was not found in module {module.__name__!r}.",
            param_hint='--crop-mode'
        )

    if crop_mode is CropMode.NONE:
        if crop_size is not None:
            raise typer.BadParameter(
                "Do not pass --crop-size with --crop-mode none.",
                param_hint="--crop-size",
            )
        if crop_min_size is not None:
            raise typer.BadParameter(
                "Do not pass --crop-min-size with --crop-mode none.",
                param_hint="--crop-min-size",
            )

    elif crop_mode is CropMode.CENTER:
        if crop_size is None:
            raise typer.BadParameter(
                f"--crop-size is required when --crop-mode {CropMode.CENTER}.",
                param_hint='--crop-size'
            )
        if crop_min_size is not None:
            raise typer.BadParameter(
                f"--crop-min-size is not used with --crop-mode {CropMode.CENTER}.",
                param_hint='--crop-size'
            )

    elif crop_mode is CropMode.NON_EMPTY:
        if crop_min_size is None:
            raise typer.BadParameter(
                "--crop-min-size is required when --crop-mode non-empty.",
                param_hint='--crop-min-size'
            )
        if crop_size is not None:
            raise typer.BadParameter(
                "--crop-size is not used with --crop-mode non-empty.",
                param_hint='--crop-size'
            )

    return CropConfig(fn = fn,min_size=crop_min_size,size=crop_size)

@dataclass(frozen=True)
class ClampConfig:
    fn: Callable
    percentile: tuple[float,float] | None = None
    min: tuple[float, float, float,float] | None = None
    max: tuple[float, float, float, float] | None = None

class ClampMode(StrEnum):
    NONE = "none"
    SUBJECT = "subject"
    DATASET = "dataset"

def build_clamp_config(
    clamp_mode: ClampMode,
    clamp_percentile: tuple[float, float] | None,
    clamp_min: tuple[float, float, float, float] | None,
    clamp_max: tuple[float, float, float, float] | None,
)->ClampConfig:
    module = importlib.import_module("brainchmark.preprocessing.clamping")

    fn = getattr(module, clamp_mode.value, None)

    if fn is None or not callable(fn):
        raise typer.BadParameter(
            f"Unknown crop mode {clamp_mode!r}: function {clamp_mode!r} "
            f"was not found in module {module.__name__!r}."
        )

    if clamp_mode == ClampMode.NONE:
        if clamp_percentile is not None:
            raise typer.BadParameter(
                "must be omitted when --clamp-mode is 'none'",
                param_hint="--clamp-percentile",
            )
        if clamp_min is not None:
            raise typer.BadParameter(
                "must be omitted when --clamp-mode is 'none'",
                param_hint="--clamp-min",
            )
        if clamp_max is not None:
            raise typer.BadParameter(
                "must be omitted when --clamp-mode is 'none'",
                param_hint="--clamp-max",
            )
    elif clamp_mode == ClampMode.SUBJECT:
        if clamp_percentile is None:
            raise typer.BadParameter(
                "is required when --clamp-mode is 'subject'",
                param_hint="--clamp-percentile",
            )
        if clamp_min is not None:
            raise typer.BadParameter(
                "must be omitted when --clamp-mode is 'subject'",
                param_hint="--clamp-min",
            )
        if clamp_max is not None:
            raise typer.BadParameter(
                "must be omitted when --clamp-mode is 'subject'",
                param_hint="--clamp-max",
            )
        if clamp_percentile[0] >= clamp_percentile[1]:
            raise typer.BadParameter(
                "LOW must be strictly smaller than HIGH.",
                param_hint="--clamp-percentile",
            )
        if clamp_percentile[0] < 0:
            raise typer.BadParameter(
                "LOW must be greater than or equal to 0.",
                param_hint="--clamp-percentile",
            )
        if clamp_percentile[0] > 100:
            raise typer.BadParameter(
                "LOW must be less than or equal to 100.",
                param_hint="--clamp-percentile",
            )
        if clamp_percentile[1] < 0:
            raise typer.BadParameter(
                "HIGH must be greater than or equal to 0.",
                param_hint="--clamp-percentile",
            )
        if clamp_percentile[1] > 100:
            raise typer.BadParameter(
                "HIGH must be less than or equal to 100.",
                param_hint="--clamp-percentile",
            )

    elif clamp_mode == ClampMode.DATASET:
        if clamp_percentile is not None:
            raise typer.BadParameter(
                f"must be omitted when --clamp-mode is '{ClampMode.DATASET}'",
                param_hint="--clamp-percentile",
            )
        if clamp_min is None:
            raise typer.BadParameter(
                f"is required when --clamp-mode is '{ClampMode.DATASET}'",
                param_hint="--clamp-min",
            )
        if clamp_max is None:
            raise typer.BadParameter(
                f"is required when --clamp-mode is '{ClampMode.DATASET}'",
                param_hint="--clamp-max",
            )
        for i, (min_value, max_value) in enumerate(zip(clamp_min, clamp_max)):
            if min_value >= max_value:
                raise typer.BadParameter(
                    f"value at index {i} must be strictly smaller than the corresponding "
                    f"value in --clamp-max, got {min_value} >= {max_value}",
                    param_hint="--clamp-min",
                )
    return ClampConfig(fn=fn,percentile=clamp_percentile,min=clamp_min,max=clamp_max)

class NormMode(StrEnum):
    NONE = "none" #sussy
    MIN_MAX = "min_max" #this will need a --min-max-range param
    SUBJECT_ZSCORE = "subject_zscore"
    DATASET_ZSCORE = "dataset_zscore"

@dataclass(frozen=True)
class NormConfig:
    fn: Callable
    min_max_range: tuple[float, float] | None = None
    mean: tuple[float, float, float,float] | None = None
    std: tuple[float, float, float,float] | None = None

def build_norm_config(
    norm_mode: NormMode,
    norm_min_max_range: tuple[float, float] | None,
    norm_mean: tuple[float, float, float, float] | None,
    norm_std: tuple[float, float, float, float] | None,
)->NormConfig:
    module = importlib.import_module("brainchmark.preprocessing.normalization")

    fn = getattr(module, norm_mode.value, None)

    if fn is None or not callable(fn):
        raise typer.BadParameter(
            f"Unknown crop mode {norm_mode!r}: function {norm_mode!r} "
            f"was not found in module {module.__name__!r}."
        )
    if norm_mode == NormMode.NONE:
        if norm_min_max_range is not None:
            raise typer.BadParameter(
                f"must be omitted when --norm-mode is '{NormMode.NONE}'",
                param_hint="--norm-min-max-range",
            )
        if norm_mean is not None:
            raise typer.BadParameter(
                f"must be omitted when --norm-mode is '{NormMode.NONE}'",
                param_hint="--norm-mean",
            )
        if norm_std is not None:
            raise typer.BadParameter(
                f"must be omitted when --norm-mode is '{NormMode.NONE}'",
                param_hint="--norm-std",
            )

    elif norm_mode == NormMode.MIN_MAX:
        if norm_min_max_range is None:
            raise typer.BadParameter(
                f"is required when --norm-mode is '{NormMode.NONE}'",
                param_hint="--norm-min-max-range",
            )
        if norm_mean is not None:
            raise typer.BadParameter(
                "must be omitted when --norm-mode is 'min_max'",
                param_hint="--norm-mean",
            )
        if norm_std is not None:
            raise typer.BadParameter(
                "must be omitted when --norm-mode is 'min_max'",
                param_hint="--norm-std",
            )
        if norm_min_max_range[0] >= norm_min_max_range[1]:
            raise typer.BadParameter(
                "MIN must be strictly smaller than MAX.",
                param_hint="--norm-min-max-range",
            )

    elif norm_mode == NormMode.SUBJECT_ZSCORE:
        if norm_min_max_range is not None:
            raise typer.BadParameter(
                f"must be omitted when --norm-mode is '{NormMode.SUBJECT_ZSCORE}'",
                param_hint="--norm-min-max-range",
            )
        if norm_mean is not None:
            raise typer.BadParameter(
                f"must be omitted when --norm-mode is '{NormMode.SUBJECT_ZSCORE}'",
                param_hint="--norm-mean",
            )
        if norm_std is not None:
            raise typer.BadParameter(
                f"must be omitted when --norm-mode is '{NormMode.SUBJECT_ZSCORE}'",
                param_hint="--norm-std",
            )

    elif norm_mode == NormMode.DATASET_ZSCORE:
        if norm_min_max_range is not None:
            raise typer.BadParameter(
                f"must be omitted when --norm-mode is '{NormMode.DATASET_ZSCORE}'",
                param_hint="--norm-min-max-range",
            )
        if norm_mean is None:
            raise typer.BadParameter(
                f"is required when --norm-mode is '{NormMode.DATASET_ZSCORE}'",
                param_hint="--norm-mean",
            )
        if norm_std is None:
            raise typer.BadParameter(
                f"is required when --norm-mode is '{NormMode.DATASET_ZSCORE}'",
                param_hint="--norm-std",
            )
        for i, std_value in enumerate(norm_std):
            if std_value <= 0:
                raise typer.BadParameter(
                    f"value at index {i} must be strictly greater than 0, got {std_value}",
                    param_hint="--norm-std",
                )
    return NormConfig(fn=fn,min_max_range=norm_min_max_range,mean=norm_mean,std=norm_std)
