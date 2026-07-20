"""Thin wrapper around the `ssh` CLI for running commands on a remote SLURM
login node. No paramiko/fabric dependency -- just subprocess, matching what's
already available on this cluster's nodes.
"""
from __future__ import annotations

import subprocess


class SSHCommandError(RuntimeError):
    def __init__(self, host: str, command: str, returncode: int, stdout: str, stderr: str) -> None:
        super().__init__(f"ssh {host} {command!r} exited {returncode}: {stderr.strip() or stdout.strip()}")
        self.host = host
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class SSHRunner:
    """Runs remote commands via `ssh <host> <command>`, batch-mode (no password
    prompts, no interactive host-key prompts -- the target host's key must
    already be trusted and key-based auth already set up for the account this
    runs as)."""

    def __init__(self, host: str, *, ssh_options: tuple[str, ...] = ("-o", "BatchMode=yes")) -> None:
        self.host = host
        self.ssh_options = ssh_options

    def run(self, command: str, *, timeout: float = 60.0) -> str:
        completed = subprocess.run(
            ["ssh", *self.ssh_options, self.host, command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0:
            raise SSHCommandError(self.host, command, completed.returncode, completed.stdout, completed.stderr)
        return completed.stdout
