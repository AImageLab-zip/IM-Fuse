import re
import shutil
from pathlib import Path

import typer

from mimose.utils.cli_overrides import CONFIGS_DIR


CONFIG_TEMPLATES_DIR = CONFIGS_DIR.parent / "config_templates"


def config_run_tag(config_path: Path) -> str:
    return config_path.stem.replace("_", "")


def dataset_input_dir_for_config(
    config_name: str,
    *,
    brats18_dir: Path | None,
    brats23_dir: Path | None,
) -> str | None:
    if config_name.endswith("_18.yaml"):
        return str(brats18_dir) if brats18_dir is not None else None
    if config_name.endswith("_23.yaml"):
        return str(brats23_dir) if brats23_dir is not None else None
    return None


def replace_yaml_line(
    content: str,
    *,
    key: str,
    value: str | None,
) -> str:
    replacement = f"{key}: {'null' if value is None else value}"
    pattern = re.compile(rf"^{re.escape(key)}:\s*.*$", re.MULTILINE)
    updated, count = pattern.subn(replacement, content, count=1)
    if count != 1:
        raise typer.BadParameter(
            f"Could not update '{key}' in config content",
            param_hint="mimose setup",
        )
    return updated


def update_setup_config(
    *,
    config_path: Path,
    brats18_dir: Path | None,
    brats23_dir: Path | None,
    preprocessed_root_dir: Path,
    artifacts_root_dir: Path,
) -> None:
    content = config_path.read_text(encoding="utf-8")
    run_tag = config_run_tag(config_path)
    preprocessed_dir = preprocessed_root_dir / f"{run_tag}-preprocessed"
    artifacts_dir = artifacts_root_dir / run_tag
    content = replace_yaml_line(
        content,
        key="input_dir",
        value=dataset_input_dir_for_config(
            config_path.name,
            brats18_dir=brats18_dir,
            brats23_dir=brats23_dir,
        ),
    )
    content = replace_yaml_line(content, key="output_dir", value=str(preprocessed_dir))
    if config_path.name != "preprocessing.yaml":
        content = replace_yaml_line(content, key="data_dir", value=str(preprocessed_dir))
        content = replace_yaml_line(content, key="art_dir", value=str(artifacts_dir))
        content = replace_yaml_line(
            content,
            key="checkpoint_path",
            value=str(artifacts_dir / "checkpoints" / "model_last.pth"),
        )
        content = replace_yaml_line(
            content,
            key="output_path",
            value=str(artifacts_dir / "results.txt"),
        )
    config_path.write_text(content, encoding="utf-8")


def copy_config_templates() -> list[Path]:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    template_paths = sorted(CONFIG_TEMPLATES_DIR.glob("*.y*ml"))
    if not template_paths:
        raise typer.BadParameter(
            f"No config templates found under {CONFIG_TEMPLATES_DIR}",
            param_hint="mimose setup",
        )

    copied_paths: list[Path] = []
    for template_path in template_paths:
        destination = CONFIGS_DIR / template_path.name
        shutil.copyfile(template_path, destination)
        copied_paths.append(destination)
    return copied_paths
