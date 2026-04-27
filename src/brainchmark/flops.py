from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from rich.table import Table

from brainchmark.models.config import build_model_config
from brainchmark.utils.cli_overrides import CONFIGS_DIR, load_yaml_config


@dataclass(frozen=True)
class PreparedMeasurement:
    name: str
    images: torch.Tensor | None
    mask: torch.Tensor | None
    input_shape: tuple[int, ...] | None
    note: str = ""


@dataclass(frozen=True)
class FlopsMeasurement:
    name: str
    input_shape: tuple[int, ...] | None
    macs: float | None
    flops: float | None
    params: int
    note: str = ""


@dataclass(frozen=True)
class FlopsReport:
    model_name: str
    model_class_name: str
    device: str
    sample_path: str
    sample_shape: tuple[int, ...]
    patch_size: int | None
    measurements: tuple[FlopsMeasurement, ...]


class _FlopsWrapper(nn.Module):
    def __init__(self, model: nn.Module, *, use_predict: bool) -> None:
        super().__init__()
        self.model = model
        self.use_predict = use_predict

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.use_predict:
            return self.model.predict(images, mask)  # type: ignore[attr-defined]
        return self.model(images, mask)


def run_flops_analysis(
    *,
    config_path: Path | None,
    model_name: str | None,
    custom_model_kwargs: dict[str, Any] | None,
    device: str = "auto",
) -> FlopsReport:
    yaml_config = load_yaml_config(config_path)
    resolved_model_name = resolve_model_name(yaml_config, model_name)
    model_kwargs = merge_model_kwargs(yaml_config, custom_model_kwargs)
    model_config = build_model_config(
        model_kind=resolved_model_name,
        model_kwargs=model_kwargs,
    )
    torch_device = resolve_device(device)
    model = model_config.model_class(**model_config.kwargs).to(torch_device)
    model.eval()
    if hasattr(model, "is_training"):
        model.is_training = False

    sample_path, dummy_images = load_dummy_brain()
    dummy_images = dummy_images.to(device=torch_device, dtype=torch.float32)
    patch_size = detect_input_patch_size(model)
    prepared_inputs = prepare_measurement_inputs(
        dummy_images,
        patch_size=patch_size,
        device=torch_device,
    )
    params = count_parameters(model)
    measurements: list[FlopsMeasurement] = []
    for prepared in prepared_inputs:
        if prepared.images is None or prepared.mask is None:
            measurements.append(
                FlopsMeasurement(
                    name=prepared.name,
                    input_shape=prepared.input_shape,
                    macs=None,
                    flops=None,
                    params=params,
                    note=prepared.note,
                )
            )
            continue

        use_predict = prepared.name == "predict"
        if use_predict and not hasattr(model, "predict"):
            measurements.append(
                FlopsMeasurement(
                    name=prepared.name,
                    input_shape=prepared.input_shape,
                    macs=None,
                    flops=None,
                    params=params,
                    note="skipped: model does not implement predict(images, mask)",
                )
            )
            continue

        macs = measure_macs(
            model,
            images=prepared.images,
            mask=prepared.mask,
            use_predict=use_predict,
        )
        measurements.append(
            FlopsMeasurement(
                name=prepared.name,
                input_shape=prepared.input_shape,
                macs=macs,
                flops=macs * 2,
                params=params,
                note=prepared.note,
            )
        )

    return FlopsReport(
        model_name=resolved_model_name,
        model_class_name=model_config.model_class.__name__,
        device=str(torch_device),
        sample_path=str(sample_path),
        sample_shape=tuple(dummy_images.shape),
        patch_size=patch_size,
        measurements=tuple(measurements),
    )


def resolve_model_name(
    yaml_config: dict[str, Any],
    cli_model_name: str | None,
) -> str:
    model_name = (
        cli_model_name
        if cli_model_name is not None
        else yaml_config.get("model")
    )
    if model_name is None or not str(model_name).strip():
        raise typer.BadParameter(
            "provide --model or set 'model' in --config",
            param_hint="--model",
        )
    return str(model_name).strip()


def resolve_2023_config_paths(configs_dir: Path = CONFIGS_DIR) -> tuple[Path, ...]:
    config_paths = tuple(sorted(configs_dir.glob("*_23.yaml")))
    if not config_paths:
        raise typer.BadParameter(
            f"no 2023 configs found under {configs_dir}",
            param_hint="--all",
        )
    return config_paths


def validate_flops_selection(
    *,
    all_configs: bool,
    config_path: Path | None,
    model_name: str | None,
) -> None:
    if not all_configs:
        return
    if config_path is not None or model_name is not None:
        raise typer.BadParameter(
            "--all cannot be combined with --config or --model",
            param_hint="--all",
        )


def merge_model_kwargs(
    yaml_config: dict[str, Any],
    cli_model_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    model_kwargs = dict(yaml_config.get("custom_model_kwargs") or {})
    model_kwargs.update(cli_model_kwargs or {})
    return model_kwargs


def resolve_device(device: str) -> torch.device:
    normalized = device.strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise typer.BadParameter(
            "device must be one of: auto, cpu, cuda",
            param_hint="--device",
        )
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise typer.BadParameter(
            "CUDA was requested but is not available",
            param_hint="--device",
        )
    return torch.device(normalized)


def load_dummy_brain() -> tuple[Path, torch.Tensor]:
    resource = files("brainchmark").joinpath("data/dummy_brain.npz")
    with as_file(resource) as sample_path:
        if not sample_path.is_file():
            raise RuntimeError(f"dummy brain sample not found at {sample_path}")
        with np.load(sample_path) as data:
            if "images" not in data:
                raise RuntimeError(f"{sample_path} must contain an 'images' array")
            images = np.asarray(data["images"], dtype=np.float32)

    if images.ndim == 4:
        images = images[None, ...]
    if images.ndim != 5:
        raise RuntimeError(
            "dummy brain images must have shape [M, H, W, D] or [B, M, H, W, D], "
            f"got {tuple(images.shape)}"
        )
    return sample_path, torch.from_numpy(np.ascontiguousarray(images))


def detect_input_patch_size(model: nn.Module) -> int | None:
    value = getattr(model, "input_patch_size", None)
    if value is None:
        module = __import__(model.__class__.__module__, fromlist=["input_patch_size"])
        value = getattr(module, "input_patch_size", None)
    if value is None:
        return None
    try:
        patch_size = int(value)
    except (TypeError, ValueError):
        return None
    return patch_size if patch_size > 0 else None


def prepare_measurement_inputs(
    images: torch.Tensor,
    *,
    patch_size: int | None,
    device: torch.device,
) -> tuple[PreparedMeasurement, ...]:
    images = images.to(device=device, dtype=torch.float32)
    if patch_size is None:
        return (
            PreparedMeasurement(
                name="forward",
                images=images,
                mask=all_modalities_mask(images.size(0), images.size(1), device),
                input_shape=tuple(images.shape),
            ),
            PreparedMeasurement(
                name="predict",
                images=None,
                mask=None,
                input_shape=None,
                note="skipped: no input_patch_size detected",
            ),
        )

    patch_images = center_crop_or_pad_volume(
        images,
        (patch_size, patch_size, patch_size),
    )
    return (
        PreparedMeasurement(
            name="forward",
            images=patch_images,
            mask=all_modalities_mask(
                patch_images.size(0),
                patch_images.size(1),
                device,
            ),
            input_shape=tuple(patch_images.shape),
        ),
        PreparedMeasurement(
            name="predict",
            images=images,
            mask=all_modalities_mask(images.size(0), images.size(1), device),
            input_shape=tuple(images.shape),
        ),
    )


def all_modalities_mask(
    batch_size: int,
    num_modalities: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.ones(
        batch_size,
        num_modalities,
        dtype=torch.bool,
        device=device,
    )


def center_crop_or_pad_volume(
    images: torch.Tensor,
    spatial_size: tuple[int, int, int],
) -> torch.Tensor:
    if images.ndim != 5:
        raise ValueError(f"expected [B, M, H, W, D], got {tuple(images.shape)}")

    slices: list[slice] = [slice(None), slice(None)]
    for current, target in zip(images.shape[-3:], spatial_size, strict=True):
        if current <= target:
            slices.append(slice(None))
            continue
        start = (current - target) // 2
        slices.append(slice(start, start + target))

    cropped = images[tuple(slices)]
    pad: list[int] = []
    spatial_pairs = list(zip(cropped.shape[-3:], spatial_size, strict=True))
    for current, target in reversed(spatial_pairs):
        missing = max(target - current, 0)
        before = missing // 2
        after = missing - before
        pad.extend((before, after))
    if any(pad):
        cropped = F.pad(cropped, pad)
    return cropped


def measure_macs(
    model: nn.Module,
    *,
    images: torch.Tensor,
    mask: torch.Tensor,
    use_predict: bool,
) -> float:
    try:
        from ptflops import get_model_complexity_info
    except ImportError as exc:
        raise RuntimeError(
            "ptflops>=0.7.5 is required for 'brainchmark flops'. "
            "Install project dependencies again before running this command."
        ) from exc

    wrapper = _FlopsWrapper(model, use_predict=use_predict)
    wrapper.eval()

    def input_constructor(_: tuple[int, ...]) -> dict[str, torch.Tensor]:
        return {"images": images, "mask": mask}

    with torch.no_grad():
        macs, _ = get_model_complexity_info(
            wrapper,
            tuple(images.shape[1:]),
            input_constructor=input_constructor,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
            backend="aten",
        )
    return float(macs)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_flops_table(report: FlopsReport) -> Table:
    table = Table(
        title=(
            f"FLOPs for {report.model_name} "
            f"({report.model_class_name}, device={report.device})"
        )
    )
    table.add_column("Path", style="bold cyan")
    table.add_column("Input Shape")
    table.add_column("MACs", justify="right")
    table.add_column("FLOPs", justify="right")
    table.add_column("Params", justify="right")
    table.add_column("Note")

    for measurement in report.measurements:
        table.add_row(
            measurement.name,
            _format_shape(measurement.input_shape),
            _format_count(measurement.macs),
            _format_count(measurement.flops),
            _format_count(float(measurement.params)),
            measurement.note,
        )

    patch_size = str(report.patch_size) if report.patch_size is not None else "n/a"
    table.caption = (
        f"sample={report.sample_path} shape={_format_shape(report.sample_shape)} "
        f"patch_size={patch_size}; FLOPs are reported as 2 * MACs."
    )
    return table


def _format_shape(shape: tuple[int, ...] | None) -> str:
    if shape is None:
        return "-"
    return "[" + ", ".join(str(dim) for dim in shape) + "]"


def _format_count(value: float | None) -> str:
    if value is None:
        return "-"
    units = (
        (1e12, "T"),
        (1e9, "G"),
        (1e6, "M"),
        (1e3, "K"),
    )
    for factor, suffix in units:
        if abs(value) >= factor:
            return f"{value / factor:.3f}{suffix}"
    return f"{value:.0f}"
