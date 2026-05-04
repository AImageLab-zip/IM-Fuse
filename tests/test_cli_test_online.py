from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from mimose import cli
from mimose.models import config as model_config
import mimose.testing as testing


def test_test_online_downloads_hf_checkpoint_into_art_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    data_dir = tmp_path / "data"
    art_dir = tmp_path / "artifacts"
    output_path = tmp_path / "results.txt"
    data_dir.mkdir()

    downloads: list[tuple[str, str, Path]] = []
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "maybe_notify_about_update", lambda: None)
    monkeypatch.setattr(
        cli,
        "_download_hf_checkpoint",
        lambda repo, run_name, destination: (
            downloads.append((repo, run_name, destination)),
            destination.parent.mkdir(parents=True, exist_ok=True),
            destination.write_bytes(b"ckpt"),
        ),
    )
    monkeypatch.setattr(
        model_config,
        "build_model_config",
        lambda **kwargs: model_config.ModelConfig(model_class=object, kwargs={}),
    )
    monkeypatch.setattr(
        testing,
        "run_testing",
        lambda **kwargs: captured.update(kwargs) or kwargs["output_path"],
    )

    result = runner.invoke(
        cli.app,
        [
            "test",
            "--data-dir",
            str(data_dir),
            "--output-path",
            str(output_path),
            "--dataset-type",
            "brats23",
            "--art-dir",
            str(art_dir),
            "--hf-repo",
            "owner/repo",
            "--hf-run-name",
            "imfuse23_training",
            "--online",
        ],
    )

    expected_checkpoint = (
        art_dir
        / "checkpoints"
        / "hf"
        / "owner--repo"
        / "imfuse23_training"
        / "final_weights_only.safetensors"
    )
    assert result.exit_code == 0
    assert downloads == [("owner/repo", "imfuse23_training", expected_checkpoint)]
    assert captured["checkpoint_path"] == expected_checkpoint


def test_test_online_reuses_cached_hf_checkpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    data_dir = tmp_path / "data"
    art_dir = tmp_path / "artifacts"
    output_path = tmp_path / "results.txt"
    cached_checkpoint = (
        art_dir
        / "checkpoints"
        / "hf"
        / "owner--repo"
        / "imfuse23_training"
        / "final_weights_only.safetensors"
    )
    data_dir.mkdir()
    cached_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    cached_checkpoint.write_bytes(b"cached")

    downloads: list[tuple[str, str, Path]] = []
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "maybe_notify_about_update", lambda: None)
    monkeypatch.setattr(
        cli,
        "_download_hf_checkpoint",
        lambda repo, run_name, destination: downloads.append((repo, run_name, destination)),
    )
    monkeypatch.setattr(
        model_config,
        "build_model_config",
        lambda **kwargs: model_config.ModelConfig(model_class=object, kwargs={}),
    )
    monkeypatch.setattr(
        testing,
        "run_testing",
        lambda **kwargs: captured.update(kwargs) or kwargs["output_path"],
    )

    result = runner.invoke(
        cli.app,
        [
            "test",
            "--data-dir",
            str(data_dir),
            "--output-path",
            str(output_path),
            "--dataset-type",
            "brats23",
            "--art-dir",
            str(art_dir),
            "--hf-repo",
            "owner/repo",
            "--hf-run-name",
            "imfuse23_training",
            "--online",
        ],
    )

    assert result.exit_code == 0
    assert downloads == []
    assert captured["checkpoint_path"] == cached_checkpoint


def test_test_online_requires_hf_repo(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    data_dir = tmp_path / "data"
    art_dir = tmp_path / "artifacts"
    output_path = tmp_path / "results.txt"
    data_dir.mkdir()

    monkeypatch.setattr(cli, "maybe_notify_about_update", lambda: None)

    result = runner.invoke(
        cli.app,
        [
            "test",
            "--data-dir",
            str(data_dir),
            "--output-path",
            str(output_path),
            "--dataset-type",
            "brats23",
            "--art-dir",
            str(art_dir),
            "--hf-run-name",
            "imfuse23_training",
            "--online",
        ],
    )

    assert result.exit_code == 2
    assert "--hf-repo" in result.output


def test_test_online_requires_hf_run_name(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    data_dir = tmp_path / "data"
    art_dir = tmp_path / "artifacts"
    output_path = tmp_path / "results.txt"
    data_dir.mkdir()

    monkeypatch.setattr(cli, "maybe_notify_about_update", lambda: None)

    result = runner.invoke(
        cli.app,
        [
            "test",
            "--data-dir",
            str(data_dir),
            "--output-path",
            str(output_path),
            "--dataset-type",
            "brats23",
            "--art-dir",
            str(art_dir),
            "--hf-repo",
            "owner/repo",
            "--online",
        ],
    )

    assert result.exit_code == 2
    assert "--hf-run-name" in result.output


def test_test_uses_final_weights_only_from_art_dir_when_checkpoint_path_is_omitted(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    data_dir = tmp_path / "data"
    art_dir = tmp_path / "artifacts"
    output_path = tmp_path / "results.txt"
    final_weights_only = art_dir / "checkpoints" / "final_weights_only.safetensors"
    data_dir.mkdir()
    final_weights_only.parent.mkdir(parents=True, exist_ok=True)
    final_weights_only.write_bytes(b"weights")

    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "maybe_notify_about_update", lambda: None)
    monkeypatch.setattr(
        model_config,
        "build_model_config",
        lambda **kwargs: model_config.ModelConfig(model_class=object, kwargs={}),
    )
    monkeypatch.setattr(
        testing,
        "run_testing",
        lambda **kwargs: captured.update(kwargs) or kwargs["output_path"],
    )

    result = runner.invoke(
        cli.app,
        [
            "test",
            "--data-dir",
            str(data_dir),
            "--output-path",
            str(output_path),
            "--dataset-type",
            "brats23",
            "--art-dir",
            str(art_dir),
        ],
    )

    assert result.exit_code == 0
    assert captured["checkpoint_path"] == final_weights_only


def test_test_requires_final_weights_only_checkpoint_in_art_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    data_dir = tmp_path / "data"
    art_dir = tmp_path / "artifacts"
    output_path = tmp_path / "results.txt"
    data_dir.mkdir()

    monkeypatch.setattr(cli, "maybe_notify_about_update", lambda: None)

    result = runner.invoke(
        cli.app,
        [
            "test",
            "--data-dir",
            str(data_dir),
            "--output-path",
            str(output_path),
            "--dataset-type",
            "brats23",
            "--art-dir",
            str(art_dir),
        ],
    )

    assert result.exit_code == 2
    assert "final_weights_only.safetensors" in result.output
    assert "--art-dir" in result.output
