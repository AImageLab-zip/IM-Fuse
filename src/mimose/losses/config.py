from dataclasses import dataclass, field
import importlib
from pathlib import Path
from typing import Any

import typer

from mimose.enums import LossKind


@dataclass(frozen=True)
class LossConfig:
    loss_class: Any
    kwargs: dict[str, Any] = field(default_factory=dict)


def build_loss_config(
    loss_kind: LossKind | str,
    loss_kwargs: dict[str, Any] | None = None,
) -> LossConfig:
    if isinstance(loss_kind, LossKind):
        resolved_loss_name = loss_kind.value
    else:
        resolved_loss_name = str(loss_kind).strip()

    if not resolved_loss_name:
        raise typer.BadParameter(
            "Loss name cannot be empty",
            param_hint="--loss",
        )

    loss_class = _resolve_loss(resolved_loss_name)
    resolved_kwargs = _normalize_loss_kwargs(resolved_loss_name, loss_kwargs or {})
    return LossConfig(
        loss_class=loss_class,
        kwargs=resolved_kwargs,
    )


def _resolve_loss(loss_name: str) -> Any:
    target_name = _normalize_name(loss_name)
    package_dir = Path(__file__).resolve().parent

    for module_path in sorted(package_dir.glob("*.py")):
        if module_path.stem in {"__init__", "config"}:
            continue

        module = importlib.import_module(f"mimose.losses.{module_path.stem}")
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue

            attr = getattr(module, attr_name)
            if getattr(attr, "__module__", None) != module.__name__:
                continue
            if not callable(attr):
                continue
            if _matches_name(target_name, attr_name):
                return attr

    raise typer.BadParameter(
        f"Unsupported loss: {loss_name}",
        param_hint="--loss",
    )


def _normalize_loss_kwargs(loss_name: str, loss_kwargs: dict[str, Any]) -> dict[str, Any]:
    target_name = _normalize_name(loss_name)
    kwargs = dict(loss_kwargs)

    if target_name == "imfuse":
        if kwargs.get("num_classes") is not None:
            num_classes = int(kwargs["num_classes"])
            if num_classes <= 0:
                raise typer.BadParameter("loss num_classes must be > 0", param_hint="--loss-num-classes")
            kwargs["num_classes"] = num_classes

        for key, param_hint in (
            ("fuse_weight", "--fuse-weight"),
            ("sep_weight", "--sep-weight"),
            ("prm_weight", "--prm-weight"),
        ):
            if kwargs.get(key) is not None:
                value = float(kwargs[key])
                if value < 0:
                    raise typer.BadParameter(f"{key} must be >= 0", param_hint=param_hint)
                kwargs[key] = value

        if kwargs.get("eps") is not None:
            eps = float(kwargs["eps"])
            if eps <= 0:
                raise typer.BadParameter("loss eps must be > 0", param_hint="--loss-eps")
            kwargs["eps"] = eps

        if kwargs.get("log_clamp_min") is not None:
            log_clamp_min = float(kwargs["log_clamp_min"])
            if not 0 < log_clamp_min <= 1:
                raise typer.BadParameter(
                    "log_clamp_min must be in the range (0, 1]",
                    param_hint="--log-clamp-min",
                )
            kwargs["log_clamp_min"] = log_clamp_min

    if target_name == "a2fseg":
        if kwargs.get("num_classes") is not None:
            num_classes = int(kwargs["num_classes"])
            if num_classes <= 0:
                raise typer.BadParameter("loss num_classes must be > 0", param_hint="--loss-num-classes")
            kwargs["num_classes"] = num_classes

        for key, param_hint in (
            ("fuse_weight", "--fuse-weight"),
            ("sep_weight", "--sep-weight"),
            ("fusion_ds_weight", "--fusion-ds-weight"),
        ):
            if kwargs.get(key) is not None:
                value = float(kwargs[key])
                if value < 0:
                    raise typer.BadParameter(f"{key} must be >= 0", param_hint=param_hint)
                kwargs[key] = value

        if kwargs.get("eps") is not None:
            eps = float(kwargs["eps"])
            if eps <= 0:
                raise typer.BadParameter("loss eps must be > 0", param_hint="--loss-eps")
            kwargs["eps"] = eps

        if kwargs.get("log_clamp_min") is not None:
            log_clamp_min = float(kwargs["log_clamp_min"])
            if not 0 < log_clamp_min <= 1:
                raise typer.BadParameter(
                    "log_clamp_min must be in the range (0, 1]",
                    param_hint="--log-clamp-min",
                )
            kwargs["log_clamp_min"] = log_clamp_min

    if target_name == "clrs":
        if kwargs.get("num_classes") is not None:
            num_classes = int(kwargs["num_classes"])
            if num_classes <= 0:
                raise typer.BadParameter("loss num_classes must be > 0", param_hint="--loss-num-classes")
            kwargs["num_classes"] = num_classes

        for key, param_hint in (
            ("coe_specloss", "--coe-specloss"),
            ("coe_consist", "--coe-consist"),
            ("temperature", "--temperature"),
        ):
            if kwargs.get(key) is not None:
                value = float(kwargs[key])
                if value < 0:
                    raise typer.BadParameter(f"{key} must be >= 0", param_hint=param_hint)
                kwargs[key] = value

        if kwargs.get("eps") is not None:
            eps = float(kwargs["eps"])
            if eps <= 0:
                raise typer.BadParameter("loss eps must be > 0", param_hint="--loss-eps")
            kwargs["eps"] = eps

        if kwargs.get("log_clamp_min") is not None:
            log_clamp_min = float(kwargs["log_clamp_min"])
            if not 0 < log_clamp_min <= 1:
                raise typer.BadParameter(
                    "log_clamp_min must be in the range (0, 1]",
                    param_hint="--log-clamp-min",
                )
            kwargs["log_clamp_min"] = log_clamp_min

    if target_name in {"tinymimosa"}:
        for key, param_hint in (
            ("dice_weight", "--dice-weight"),
            ("ce_weight", "--ce-weight"),
        ):
            if kwargs.get(key) is not None:
                value = float(kwargs[key])
                if value < 0:
                    raise typer.BadParameter(f"{key} must be >= 0", param_hint=param_hint)
                kwargs[key] = value

        if kwargs.get("eps") is not None:
            eps = float(kwargs["eps"])
            if eps <= 0:
                raise typer.BadParameter("loss eps must be > 0", param_hint="--loss-eps")
            kwargs["eps"] = eps

    return kwargs


def _matches_name(target_name: str, candidate_name: str) -> bool:
    normalized_candidate = _normalize_name(candidate_name)
    return (
        normalized_candidate == target_name
        or normalized_candidate.removesuffix("loss") == target_name
        or target_name.removesuffix("loss") == normalized_candidate
    )


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())
