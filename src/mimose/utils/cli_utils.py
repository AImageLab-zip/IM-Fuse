import typer
from typing import Any
import os
import shutil
import sys
import subprocess

def require_preprocess_values(merged: dict[str, Any], *required_keys: str) -> None:
    for key in required_keys:
        value = merged.get(key)
        if value is None:
            raise typer.BadParameter(
                "missing value; provide it in the CLI or in --config",
                param_hint=f"--{key.replace('_', '-')}",
            )

def maybe_relaunch_with_torchrun() -> None:
    if "--distributed" not in sys.argv:
        return

    if "RANK" in os.environ:
        return

    torchrun = shutil.which("torchrun")
    if torchrun is None:
        raise RuntimeError("torchrun is not available")

    cmd = [
        torchrun,
        "--nproc-per-node=4",
        "-m",
        "mimose.cli",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(cmd))