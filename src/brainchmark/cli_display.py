from pathlib import Path

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.completion import PathCompleter
from rich.console import Console
from rich.panel import Panel


CONSOLE = Console()
PATH_COMPLETER = PathCompleter(expanduser=True)


def expand_user_path(raw_value: str) -> Path:
    return Path(raw_value).expanduser().resolve(strict=False)


def prompt_path(prompt: str) -> str:
    return pt_prompt(
        f"{prompt}: ",
        completer=PATH_COMPLETER,
        complete_while_typing=True,
    ).strip()


def prompt_optional_existing_directory(
    *,
    label: str,
    prompt: str,
) -> Path | None:
    while True:
        raw_value = prompt_path(prompt)
        if not raw_value:
            return None

        candidate = expand_user_path(raw_value)
        if candidate.is_dir():
            return candidate

        CONSOLE.print(
            Panel(
                f"[bold red]{label}[/bold red]\n{candidate} is not an existing directory.",
                title="[bold red]Invalid Directory[/bold red]",
                border_style="red",
                expand=False,
            )
        )


def prompt_required_directory(
    *,
    label: str,
    prompt: str,
) -> Path:
    while True:
        raw_value = prompt_path(prompt)
        if not raw_value:
            CONSOLE.print(
                Panel(
                    f"[bold red]{label}[/bold red] is required.",
                    title="[bold red]Missing Value[/bold red]",
                    border_style="red",
                    expand=False,
                )
            )
            continue

        return expand_user_path(raw_value)
