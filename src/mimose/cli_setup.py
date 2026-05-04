import re
import shutil
from pathlib import Path

import typer

from mimose.utils.cli_overrides import CONFIGS_DIR


CONFIG_TEMPLATES_DIR = CONFIGS_DIR.parent / "config_templates"
HF_REPO_PLACEHOLDER_PREFIXES = ("/path/to/", "<", "__")


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


def upsert_yaml_line(
    content: str,
    *,
    key: str,
    value: str | None,
) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:\s*.*$", re.MULTILINE)
    if pattern.search(content):
        return replace_yaml_line(content, key=key, value=value)

    if content and not content.endswith("\n"):
        content += "\n"
    return content + f"{key}: {'null' if value is None else value}\n"


def get_yaml_line_value(content: str, *, key: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(key)}:\s*(.*)$", re.MULTILINE)
    match = pattern.search(content)
    if match is None:
        return None
    return match.group(1).strip()


def is_placeholder_value(value: str | None) -> bool:
    if value is None:
        return False
    cleaned = value.strip().strip("\"'")
    if cleaned in {"", "null", "none", "~"}:
        return False
    return cleaned.startswith(HF_REPO_PLACEHOLDER_PREFIXES)


def update_setup_config(
    *,
    config_path: Path,
    brats18_dir: Path | None,
    brats23_dir: Path | None,
    preprocessed_root_dir: Path,
    artifacts_root_dir: Path,
    hf_repo: str | None = None,
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
            value=str(artifacts_dir / "checkpoints" / "final_weights_only.safetensors"),
        )
        content = replace_yaml_line(
            content,
            key="output_path",
            value=str(artifacts_dir / "results.txt"),
        )
        template_hf_repo = get_yaml_line_value(content, key="hf_repo")
        if is_placeholder_value(template_hf_repo):
            content = upsert_yaml_line(content, key="push_to_hf", value="true" if hf_repo else "false")
            content = upsert_yaml_line(content, key="hf_repo", value=hf_repo)
    config_path.write_text(content, encoding="utf-8")


def templates_require_hf_repo_prompt() -> bool:
    for template_path in sorted(CONFIG_TEMPLATES_DIR.glob("*.y*ml")):
        content = template_path.read_text(encoding="utf-8")
        if is_placeholder_value(get_yaml_line_value(content, key="hf_repo")):
            return True
    return False


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
