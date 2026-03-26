import yaml
from pathlib import Path
from typing import Any
import typer
CONFIG_NONE = "none"
def load_yaml_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for key,value in data.items():
        if isinstance(value,list):
            data[key] = tuple(value)
    if not isinstance(data, dict):
        raise typer.BadParameter("YAML config must contain a mapping at the top level.", param_hint="--config")

    return data

def merge_cli_overrides(
    yaml_config: dict[str, Any],
    **cli_values: Any,
) -> dict[str, Any]:
    merged = dict(yaml_config)

    for key, value in cli_values.items():
        if value is not None and value != CONFIG_NONE:
            merged[key] = value

    return merged