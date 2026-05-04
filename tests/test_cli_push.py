from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from mimose import cli
import mimose.cli_workflows as workflows


def _write_push_config(path: Path, *, hf_repo: str = "owner/repo") -> None:
    path.write_text(
        "\n".join(
            [
                "data_dir: /tmp/data",
                "art_dir: /tmp/artifacts",
                "trainer: imfuse",
                "model: imfuse",
                "optimizer: radam",
                "num_epochs: 1",
                "num_workers: 8",
                "wandb_run_name: imfuse23_training",
                f"hf_repo: {hf_repo}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_push_uses_config_only_inputs(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "push.yaml"
    _write_push_config(config_path)

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        workflows,
        "run_push_from_merged",
        lambda merged: captured.update(merged) or (tmp_path / "export"),
    )

    result = runner.invoke(cli.app, ["push", "--config", str(config_path)])

    assert result.exit_code == 0
    assert captured["hf_repo"] == "owner/repo"
    assert captured["wandb_run_name"] == "imfuse23_training"
    assert captured["trainer"] == "imfuse"
    assert captured["model"] == "imfuse"


def test_push_cli_overrides_config_values(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "push.yaml"
    checkpoint_path = tmp_path / "manual.safetensors"
    checkpoint_path.write_bytes(b"weights")
    _write_push_config(config_path, hf_repo="owner/repo")

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        workflows,
        "run_push_from_merged",
        lambda merged: captured.update(merged) or (tmp_path / "export"),
    )

    result = runner.invoke(
        cli.app,
        [
            "push",
            "--config",
            str(config_path),
            "--hf-repo",
            "other/repo",
            "--wandb-run-name",
            "override_run",
            "--checkpoint-path",
            str(checkpoint_path),
        ],
    )

    assert result.exit_code == 0
    assert captured["hf_repo"] == "other/repo"
    assert captured["wandb_run_name"] == "override_run"
    assert captured["checkpoint_path"] == str(checkpoint_path)


def test_push_requires_hf_repo(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "push.yaml"
    _write_push_config(config_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("hf_repo: owner/repo\n", ""),
        encoding="utf-8",
    )

    result = runner.invoke(cli.app, ["push", "--config", str(config_path)])

    assert result.exit_code == 2
    assert "--hf-repo" in result.output


def test_push_rejects_url_form_hf_repo(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "push.yaml"
    _write_push_config(config_path, hf_repo="https://huggingface.co/owner/repo")

    result = runner.invoke(cli.app, ["push", "--config", str(config_path)])

    assert result.exit_code == 2
    assert "not a full URL" in result.output


def test_push_reports_missing_default_checkpoint(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "push.yaml"
    art_dir = tmp_path / "artifacts"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_path.write_text(
        "\n".join(
            [
                f"data_dir: {data_dir}",
                f"art_dir: {art_dir}",
                "trainer: imfuse",
                "model: imfuse",
                "optimizer: radam",
                "num_epochs: 1",
                "num_workers: 8",
                "wandb_run_name: imfuse23_training",
                "hf_repo: owner/repo",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        workflows,
        "_build_trainer_instance_from_merged",
        lambda merged: object(),
    )

    result = runner.invoke(cli.app, ["push", "--config", str(config_path)])

    assert result.exit_code == 2
    assert "--checkpoint-path" in result.output
