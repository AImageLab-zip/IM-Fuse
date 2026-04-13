from dataclasses import dataclass, field
from enum import StrEnum
import importlib
from pathlib import Path
from typing import Any

import typer


class LossKind(StrEnum):
    IMFUSE = "imfuse"


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
    return LossConfig(
        loss_class=loss_class,
        kwargs=loss_kwargs or {},
    )


def _resolve_loss(loss_name: str) -> Any:
    target_name = _normalize_name(loss_name)
    package_dir = Path(__file__).resolve().parent

    for module_path in sorted(package_dir.glob("*.py")):
        if module_path.stem in {"__init__", "config"}:
            continue

        module = importlib.import_module(f"brainchmark.losses.{module_path.stem}")
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


def _matches_name(target_name: str, candidate_name: str) -> bool:
    normalized_candidate = _normalize_name(candidate_name)
    return (
        normalized_candidate == target_name
        or normalized_candidate.removesuffix("loss") == target_name
        or target_name.removesuffix("loss") == normalized_candidate
    )


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())
