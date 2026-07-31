"""Runner configuration, loaded from a `.env` file at the repo root.

See `.env.example` for the two variables this reads:
`MIMOSE_RUNNER_PORT` (what port to serve the website on) and
`MIMOSE_RUNNER_RAW_BRATS_DIR` (path to the already-unpacked raw BRaTS root,
mounted read-only into the preprocessing container).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# src/runner/config.py -> src/runner -> src -> repo root. Assumes this package
# is used from a checkout of this repo (e.g. an editable install), same as
# docker/*/Dockerfile already assume being built from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(REPO_ROOT / ".env")

PORT_ENV_VAR = "MIMOSE_RUNNER_PORT"
RAW_BRATS_DIR_ENV_VAR = "MIMOSE_RUNNER_RAW_BRATS_DIR"


class ConfigError(RuntimeError):
    """Raised when the runner's `.env` is missing or invalid."""


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ConfigError(
            f"{name} is not set -- copy .env.example to .env at the repo root and fill it in."
        )
    return value


def get_port() -> int:
    value = _require_env(PORT_ENV_VAR)
    try:
        return int(value)
    except ValueError:
        raise ConfigError(f"{PORT_ENV_VAR}={value!r} is not a valid integer port") from None


def get_raw_brats_dir() -> Path:
    raw_dir = Path(_require_env(RAW_BRATS_DIR_ENV_VAR)).expanduser().resolve()
    if not raw_dir.is_dir():
        raise ConfigError(f"{RAW_BRATS_DIR_ENV_VAR}={raw_dir} is not a directory")
    return raw_dir
