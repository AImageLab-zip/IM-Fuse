from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from brainchmark import cli
from brainchmark.models import config as model_config
import brainchmark.testing as testing


def test_test_online_downloads_checkpoint_into_art_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    data_dir = tmp_path / "data"
    art_dir = tmp_path / "artifacts"
    output_path = tmp_path / "results.txt"
    data_dir.mkdir()

    downloads: list[tuple[str, Path]] = []
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "maybe_notify_about_update", lambda: None)
    monkeypatch.setattr(
        cli,
        "_download_checkpoint",
        lambda link, destination: (
            downloads.append((link, destination)),
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
            "--checkpoint-link",
            "https://example.com/model_last.pth",
            "--online",
        ],
    )

    expected_checkpoint = art_dir / "checkpoints" / "online_checkpoint.pth"
    assert result.exit_code == 0
    assert downloads == [("https://example.com/model_last.pth", expected_checkpoint)]
    assert captured["checkpoint_path"] == expected_checkpoint


def test_test_online_reuses_cached_checkpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    data_dir = tmp_path / "data"
    art_dir = tmp_path / "artifacts"
    output_path = tmp_path / "results.txt"
    cached_checkpoint = art_dir / "checkpoints" / "online_checkpoint.pth"
    data_dir.mkdir()
    cached_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    cached_checkpoint.write_bytes(b"cached")

    downloads: list[tuple[str, Path]] = []
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "maybe_notify_about_update", lambda: None)
    monkeypatch.setattr(
        cli,
        "_download_checkpoint",
        lambda link, destination: downloads.append((link, destination)),
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
            "--checkpoint-link",
            "https://example.com/model_last.pth",
            "--online",
        ],
    )

    assert result.exit_code == 0
    assert downloads == []
    assert captured["checkpoint_path"] == cached_checkpoint


def test_test_online_requires_checkpoint_link(monkeypatch, tmp_path: Path) -> None:
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
            "--online",
        ],
    )

    assert result.exit_code == 2
    assert "--checkpoint-link" in result.output
