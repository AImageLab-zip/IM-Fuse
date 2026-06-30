import ast
import yaml
from pathlib import Path
from typing import Any
import typer
CONFIG_NONE = "none"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CONFIGS_DIR = DATA_DIR / "configs"
SPLITS_DIR = DATA_DIR / "splits"


def _resolve_config_path(config_path: Path) -> Path:
    if config_path.is_absolute() or config_path.exists():
        return config_path

    candidate = CONFIGS_DIR / config_path
    if candidate.exists():
        return candidate

    raise typer.BadParameter(
        "config file not found. Use an absolute path, a valid relative path, "
        f"or a filename under {CONFIGS_DIR}",
        param_hint="--config",
    )


def resolve_split_path(split_path: Path | str | None) -> Path:
    if split_path is None:
        return SPLITS_DIR / "split.json"

    candidate = Path(split_path)
    if candidate.is_absolute() or candidate.exists():
        return candidate

    resolved = SPLITS_DIR / candidate
    if resolved.exists():
        return resolved

    raise typer.BadParameter(
        "split file not found. Use an absolute path, a valid relative path, "
        f"or a filename under {SPLITS_DIR}",
        param_hint="--split-file",
    )


def load_yaml_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}

    resolved_path = _resolve_config_path(config_path)
    with resolved_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data = _normalize_config_value(data)
    if not isinstance(data, dict):
        raise typer.BadParameter("YAML config must contain a mapping at the top level.", param_hint="--config")

    return data


def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_config_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return tuple(_normalize_config_value(inner) for inner in value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("(", "[", "{")) and stripped.endswith((")", "]", "}")):
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                return value
            return _normalize_config_value(parsed)
    return value

def merge_cli_overrides(
    yaml_config: dict[str, Any],
    **cli_values: Any,
) -> dict[str, Any]:
    merged = dict(yaml_config)

    for key, value in cli_values.items():
        if value is not None and value != CONFIG_NONE:
            merged[key] = value

    return merged
