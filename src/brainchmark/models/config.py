from dataclasses import dataclass, field
from enum import StrEnum
import importlib
from typing import Any

import typer


class ModelKind(StrEnum):
    IMFUSE = "imfuse"


@dataclass(frozen=True)
class ModelConfig:
    model_class: Any
    kwargs: dict[str, Any] = field(default_factory=dict)


def _resolve_model(model_name: str) -> Any:
    candidates = (
        model_name,
        model_name.upper(),
        model_name.capitalize(),
        "".join(part.capitalize() for part in model_name.split("_")),
    )
    module = importlib.import_module("brainchmark.models.IMFuse")

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)

        model_class = getattr(module, candidate, None)
        if model_class is not None:
            return model_class

    raise typer.BadParameter(
        f"Unsupported model: {model_name}",
        param_hint="--model",
    )


def build_model_config(
    model_kind: ModelKind | str,
    model_kwargs: dict[str, Any] | None = None,
) -> ModelConfig:
    if isinstance(model_kind, ModelKind):
        resolved_model_name = model_kind.value
    else:
        resolved_model_name = str(model_kind).strip()

    if not resolved_model_name:
        raise typer.BadParameter(
            "Model name cannot be empty",
            param_hint="--model",
        )

    model_class = _resolve_model(resolved_model_name)
    return ModelConfig(
        model_class=model_class,
        kwargs=model_kwargs or {},
    )
