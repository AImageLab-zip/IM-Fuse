from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import zipfile

import torch
from typer.testing import CliRunner

from mimose import cli
import mimose.cli_workflows as workflows
from mimose.checkpoints import save_weights_only_checkpoint
from mimose.model_export import build_export_package
from mimose.models.tiny_mimosa import TinyMimosa


def _build_test_model() -> TinyMimosa:
    model = TinyMimosa(
        num_cls=4,
        input_shape=(16, 16, 16),
        features_per_stage=(4, 8, 16, 32),
    ).eval()
    model._mimose_model_kwargs = {
        "num_cls": 4,
        "input_shape": (16, 16, 16),
        "features_per_stage": (4, 8, 16, 32),
    }
    model._mimose_model_name = "TinyMimosa"
    return model


def test_build_export_package_writes_manifest_config_weights_and_pyz(tmp_path: Path) -> None:
    model = _build_test_model()
    checkpoint_path = tmp_path / "final_weights_only.safetensors"
    save_weights_only_checkpoint(model, checkpoint_path)
    output_path = tmp_path / "export" / "tinymimosa.mimosepkg"

    result_path = build_export_package(
        model,
        checkpoint_path,
        output_path,
        run_suffix="fold1",
    )

    assert result_path == output_path
    assert output_path.is_file()

    with zipfile.ZipFile(output_path) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "config.json" in names
        assert "weights/final_weights_only.safetensors" in names
        assert "code/mimose.pyz" in names

        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["run_suffix"] == "fold1"
        assert manifest["model_class"] == "mimose.models.tiny_mimosa.TinyMimosa"
        assert "mimose_version" in manifest
        assert "checkpoint_sha256" in manifest

        config = json.loads(zf.read("config.json"))
        assert config["model_class"] == "TinyMimosa"
        assert config["model_kwargs"]["num_cls"] == 4

        pyz_bytes = zf.read("code/mimose.pyz")

    pyz_path = tmp_path / "mimose.pyz"
    pyz_path.write_bytes(pyz_bytes)
    with zipfile.ZipFile(pyz_path) as pyz:
        pyz_names = pyz.namelist()
        assert "mimose/models/tiny_mimosa.py" in pyz_names
        assert "mimose/models/abstract_model.py" in pyz_names
        assert "mimose/cli.py" in pyz_names
        assert not any("__pycache__" in name for name in pyz_names)


def test_exported_pyz_is_self_contained_and_predict_works(tmp_path: Path) -> None:
    model = _build_test_model()
    checkpoint_path = tmp_path / "final_weights_only.safetensors"
    save_weights_only_checkpoint(model, checkpoint_path)
    output_path = tmp_path / "tinymimosa.mimosepkg"
    build_export_package(model, checkpoint_path, output_path)

    with zipfile.ZipFile(output_path) as zf:
        pyz_bytes = zf.read("code/mimose.pyz")
        zf.extract("weights/final_weights_only.safetensors", tmp_path)

    pyz_path = tmp_path / "mimose.pyz"
    pyz_path.write_bytes(pyz_bytes)
    extracted_checkpoint = tmp_path / "weights" / "final_weights_only.safetensors"

    script = f"""
import sys

repo_src = {str(Path(__file__).resolve().parents[1] / "src")!r}
sys.path = [p for p in sys.path if p != repo_src]
sys.path.insert(0, {str(pyz_path)!r})

import mimose
assert mimose.__file__.startswith({str(pyz_path)!r}), mimose.__file__

from mimose.models.tiny_mimosa import TinyMimosa
from mimose.checkpoints import load_weights_only_checkpoint

model = TinyMimosa(
    num_cls=4,
    input_shape=(16, 16, 16),
    features_per_stage=(4, 8, 16, 32),
).eval()
load_weights_only_checkpoint(model, {str(extracted_checkpoint)!r}, device="cpu")

import torch
images = torch.zeros(1, 4, 16, 16, 16)
mask = torch.ones(1, 4, dtype=torch.bool)
with torch.no_grad():
    output = model.predict(images, mask)
print(tuple(output.shape))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "(1, 4, 16, 16, 16)"


def _write_export_config(path: Path, *, data_dir: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f"data_dir: {data_dir}",
                "art_dir: /tmp/artifacts",
                "trainer: imfuse",
                "model: imfuse",
                "optimizer: radam",
                "num_epochs: 1",
                "num_workers: 8",
                "wandb_run_name: imfuse23_training",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_export_uses_config_only_inputs(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "export.yaml"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_export_config(config_path, data_dir=data_dir)

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        workflows,
        "run_export_from_merged",
        lambda merged: captured.update(merged) or (tmp_path / "out.mimosepkg"),
    )

    result = runner.invoke(cli.app, ["export", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert captured["wandb_run_name"] == "imfuse23_training"
    assert captured["trainer"] == "imfuse"
    assert captured["model"] == "imfuse"
    assert captured.get("hf_repo") is None
    assert captured["push_to_hf"] is False


def test_export_applies_run_suffix_to_art_dir_and_run_name(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "export.yaml"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_export_config(config_path, data_dir=data_dir)

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        workflows,
        "run_export_from_merged",
        lambda merged: captured.update(merged) or (tmp_path / "out.mimosepkg"),
    )

    result = runner.invoke(
        cli.app,
        ["export", "--config", str(config_path), "--run-suffix", "fold1"],
    )

    assert result.exit_code == 0, result.output
    assert captured["art_dir"] == "/tmp/artifacts_fold1"
    assert captured["wandb_run_name"] == "imfuse23_training_fold1"


def test_export_default_output_path_uses_art_dir_and_run_name(tmp_path: Path) -> None:
    merged = {"art_dir": str(tmp_path / "artifacts"), "wandb_run_name": "myrun"}

    output_path = workflows._resolve_export_output_path(merged)

    assert output_path == tmp_path / "artifacts" / "export" / "myrun.mimosepkg"


def test_export_explicit_output_path_overrides_default(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "export.yaml"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_export_config(config_path, data_dir=data_dir)
    explicit_output = tmp_path / "custom" / "name.mimosepkg"

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        workflows,
        "run_export_from_merged",
        lambda merged: captured.update(merged) or explicit_output,
    )

    result = runner.invoke(
        cli.app,
        ["export", "--config", str(config_path), "--output-path", str(explicit_output)],
    )

    assert result.exit_code == 0, result.output
    assert captured["export_output_path"] == str(explicit_output)
    assert workflows._resolve_export_output_path(captured) == explicit_output
