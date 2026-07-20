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


def test_push_all_seeds_pushes_each_seed_plus_unsuffixed_default(
    monkeypatch, tmp_path: Path
) -> None:
    runner = CliRunner()
    config_path = tmp_path / "push.yaml"
    _write_push_config(config_path)

    calls: list[dict[str, object]] = []

    def fake_run_push_from_merged(merged: dict[str, object]) -> Path:
        calls.append(dict(merged))
        return tmp_path / f"export-{merged.get('run_suffix')}"

    monkeypatch.setattr(workflows, "run_push_from_merged", fake_run_push_from_merged)
    monkeypatch.setattr(
        workflows, "_resolve_push_checkpoint_path", lambda merged: tmp_path / "seed0.safetensors"
    )

    result = runner.invoke(cli.app, ["push", "--config", str(config_path), "--all-seeds"])

    assert result.exit_code == 0, result.output
    run_suffixes = [call.get("run_suffix") for call in calls]
    assert run_suffixes == ["seed0", "seed42", "seed69", None]

    seeded_calls = {call.get("run_suffix"): call for call in calls}
    assert seeded_calls["seed0"]["art_dir"] == "/tmp/artifacts_seed0"
    assert seeded_calls["seed0"]["wandb_run_name"] == "imfuse23_training_seed0"
    assert seeded_calls["seed42"]["art_dir"] == "/tmp/artifacts_seed42"
    assert seeded_calls["seed69"]["art_dir"] == "/tmp/artifacts_seed69"

    default_call = seeded_calls[None]
    assert default_call["art_dir"] == "/tmp/artifacts"
    assert default_call["wandb_run_name"] == "imfuse23_training"
    assert default_call["checkpoint_path"] == str(tmp_path / "seed0.safetensors")


def test_push_all_seeds_rejects_run_suffix(tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "push.yaml"
    _write_push_config(config_path)

    result = runner.invoke(
        cli.app,
        ["push", "--config", str(config_path), "--all-seeds", "--run-suffix", "seed0"],
    )

    assert result.exit_code == 2
    assert "--run-suffix" in result.output


def test_push_all_seeds_rejects_checkpoint_path(tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "push.yaml"
    checkpoint_path = tmp_path / "manual.safetensors"
    checkpoint_path.write_bytes(b"weights")
    _write_push_config(config_path)

    result = runner.invoke(
        cli.app,
        [
            "push",
            "--config",
            str(config_path),
            "--all-seeds",
            "--checkpoint-path",
            str(checkpoint_path),
        ],
    )

    assert result.exit_code == 2
    assert "--checkpoint-path" in result.output


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
