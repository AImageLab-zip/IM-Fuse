from __future__ import annotations

import json
import os
from pathlib import Path
import time
import tomllib
from typing import Any, Callable
from urllib.request import urlopen

from packaging.version import InvalidVersion, Version

from brainchmark import __version__

REMOTE_PYPROJECT_URL = (
    "https://raw.githubusercontent.com/AImageLab-zip/IM-Fuse/stable/pyproject.toml"
)
UPDATE_CHECK_INTERVAL_SECONDS = 7 * 24 * 60 * 60


def _default_state_path() -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "brainchmark" / "update_check.json"


def _load_state(state_path: Path) -> dict[str, Any]:
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_state(state_path: Path, state: dict[str, Any]) -> None:
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        return


def _is_check_due(state: dict[str, Any], now: float) -> bool:
    last_checked_at = state.get("last_checked_at")
    if not isinstance(last_checked_at, (int, float)):
        return True
    return now - float(last_checked_at) >= UPDATE_CHECK_INTERVAL_SECONDS


def _parse_version(version_string: str) -> Version | None:
    try:
        return Version(version_string)
    except InvalidVersion:
        return None


def _extract_version_from_pyproject(pyproject_text: str) -> str | None:
    try:
        payload = tomllib.loads(pyproject_text)
    except tomllib.TOMLDecodeError:
        return None

    project = payload.get("project")
    if not isinstance(project, dict):
        return None

    version = project.get("version")
    return version if isinstance(version, str) else None


def _fetch_remote_version(
    remote_pyproject_url: str = REMOTE_PYPROJECT_URL,
) -> str | None:
    try:
        with urlopen(remote_pyproject_url, timeout=2.0) as response:
            pyproject_text = response.read().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    return _extract_version_from_pyproject(pyproject_text)


def maybe_notify_about_update(
    *,
    now: float | None = None,
    state_path: Path | None = None,
    fetch_remote_version: Callable[[], str | None] | None = None,
    echo: Callable[[str], None] | None = None,
) -> None:
    current_time = time.time() if now is None else now
    resolved_state_path = _default_state_path() if state_path is None else state_path
    state = _load_state(resolved_state_path)

    if not _is_check_due(state, current_time):
        return

    resolved_fetch_remote_version = _fetch_remote_version
    if fetch_remote_version is not None:
        resolved_fetch_remote_version = fetch_remote_version

    try:
        latest_version = resolved_fetch_remote_version()
    except Exception:
        latest_version = None

    state["last_checked_at"] = current_time
    if latest_version is not None:
        state["latest_version"] = latest_version
    _save_state(resolved_state_path, state)

    current = _parse_version(__version__)
    latest = _parse_version(latest_version) if latest_version is not None else None
    if current is None or latest is None or latest <= current:
        return

    resolved_echo = echo
    if resolved_echo is None:
        import typer

        resolved_echo = typer.echo

    resolved_echo(
        f"Update available: MiMoSe {latest_version} is available on the "
        f"stable branch (installed: {__version__})."
    )
