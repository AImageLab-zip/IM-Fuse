from __future__ import annotations

from pathlib import Path
import subprocess

from typer.testing import CliRunner

from mimose import cli


def test_update_runs_install_script(monkeypatch) -> None:
    runner = CliRunner()
    calls: list[tuple[list[str], Path]] = []

    monkeypatch.setattr(cli, "maybe_notify_about_update", lambda: None)
    monkeypatch.setattr(cli, "_repo_root", lambda: Path("/tmp/mimose-repo"))

    def fake_is_file(self: Path) -> bool:
        return str(self) == "/tmp/mimose-repo/install.sh"

    def fake_run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        calls.append((cmd, cwd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = runner.invoke(cli.app, ["update"])

    assert result.exit_code == 0
    assert calls == [
        (
            ["bash", "/tmp/mimose-repo/install.sh"],
            Path("/tmp/mimose-repo"),
        )
    ]
