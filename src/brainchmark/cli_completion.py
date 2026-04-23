from pathlib import Path

from click.shell_completion import CompletionItem
import typer

from brainchmark.utils.cli_overrides import CONFIGS_DIR, SPLITS_DIR


def config_shell_complete(
    _ctx: typer.Context,
    _param: typer.CallbackParam,
    incomplete: str,
) -> list[CompletionItem]:
    suggestions: dict[str, CompletionItem] = {}

    for config_path in sorted(CONFIGS_DIR.glob("*.y*ml")):
        if config_path.name.startswith(incomplete):
            suggestions[config_path.name] = CompletionItem(
                config_path.name,
                help=str(CONFIGS_DIR),
            )

    if incomplete.startswith("/") or "/" in incomplete or incomplete.startswith("."):
        raw_path = Path(incomplete).expanduser()
        parent = raw_path if incomplete.endswith("/") else raw_path.parent
        prefix = "" if incomplete.endswith("/") else raw_path.name
        if parent.exists() and parent.is_dir():
            for candidate in sorted(parent.iterdir()):
                if candidate.suffix not in {".yaml", ".yml"}:
                    continue
                if not candidate.name.startswith(prefix):
                    continue
                suggestions[str(candidate)] = CompletionItem(str(candidate))

    return list(suggestions.values())


def split_shell_complete(
    _ctx: typer.Context,
    _param: typer.CallbackParam,
    incomplete: str,
) -> list[CompletionItem]:
    suggestions: dict[str, CompletionItem] = {}

    for split_path in sorted(path for path in SPLITS_DIR.rglob("*") if path.is_file()):
        relative_name = split_path.relative_to(SPLITS_DIR).as_posix()
        if relative_name.startswith(incomplete):
            suggestions[relative_name] = CompletionItem(
                relative_name,
                help=str(SPLITS_DIR),
            )

    if incomplete.startswith("/") or "/" in incomplete or incomplete.startswith("."):
        raw_path = Path(incomplete).expanduser()
        parent = raw_path if incomplete.endswith("/") else raw_path.parent
        prefix = "" if incomplete.endswith("/") else raw_path.name
        if parent.exists() and parent.is_dir():
            for candidate in sorted(parent.iterdir()):
                if not candidate.is_file():
                    continue
                if not candidate.name.startswith(prefix):
                    continue
                suggestions[str(candidate)] = CompletionItem(str(candidate))

    return list(suggestions.values())
