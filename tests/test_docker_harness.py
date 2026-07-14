from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker"))

from harness.load import PackageIntegrityError, resolve_package  # noqa: E402
from harness.preprocess import (  # noqa: E402
    PreprocessingConfigError,
    preprocess_raw_cases,
    write_test_only_split,
)

from mimose.checkpoints import save_weights_only_checkpoint
from mimose.cli_workflows import _extract_preprocessing_config
from mimose.model_export import build_export_package
from mimose.models.tiny_mimosa import TinyMimosa

SAMPLE_PREPROCESSING_CONFIG = {
    "dataset_type": "brats18",
    "crop_mode": "non_empty",
    "crop_size": None,
    "crop_min_size": (32, 32, 32),
    "clamp_mode": "none",
    "clamp_percentile": None,
    "clamp_min": None,
    "clamp_max": None,
    "norm_mode": "subject_zscore",
    "norm_min_max_range": None,
    "norm_mean": None,
    "norm_std": None,
}


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


def _build_test_package(tmp_path: Path, *, preprocessing_config: dict | None = None) -> Path:
    model = _build_test_model()
    checkpoint_path = tmp_path / "final_weights_only.safetensors"
    save_weights_only_checkpoint(model, checkpoint_path)
    output_path = tmp_path / "tinymimosa.mimosepkg"
    return build_export_package(
        model, checkpoint_path, output_path, preprocessing_config=preprocessing_config
    )


def _write_raw_case(root: Path, case_id: str, shape: tuple[int, int, int] = (40, 40, 40)) -> None:
    case_dir = root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for modality in ("t1c", "t1n", "t2f", "t2w"):
        volume = rng.random(shape).astype(np.float32)
        sitk.WriteImage(sitk.GetImageFromArray(volume), str(case_dir / f"{case_id}-{modality}.nii.gz"))
    seg = np.zeros(shape, dtype=np.uint8)
    seg[10:20, 10:20, 10:20] = 1
    sitk.WriteImage(sitk.GetImageFromArray(seg), str(case_dir / f"{case_id}-seg.nii.gz"))


def _rewrite_member(pkg_path: Path, member: str, content: bytes) -> None:
    entries = {}
    with zipfile.ZipFile(pkg_path) as zf:
        for name in zf.namelist():
            entries[name] = zf.read(name)
    entries[member] = content
    with zipfile.ZipFile(pkg_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)


def test_resolve_package_rejects_checksum_mismatch(tmp_path: Path) -> None:
    pkg_path = _build_test_package(tmp_path)
    _rewrite_member(pkg_path, "weights/final_weights_only.safetensors", b"tampered-bytes")

    with pytest.raises(PackageIntegrityError, match="sha256 mismatch"):
        resolve_package(pkg_path, tmp_path / "extract")


def test_resolve_package_rejects_missing_members(tmp_path: Path) -> None:
    pkg_path = _build_test_package(tmp_path)
    with zipfile.ZipFile(pkg_path) as zf:
        entries = {name: zf.read(name) for name in zf.namelist() if name != "code/mimose.pyz"}
    with zipfile.ZipFile(pkg_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)

    with pytest.raises(PackageIntegrityError, match="missing required members"):
        resolve_package(pkg_path, tmp_path / "extract")


def test_load_mimosepkg_reconstructs_model_and_predicts(tmp_path: Path) -> None:
    """Runs in a fresh, isolated interpreter (docker/ on sys.path, src/ removed)
    to exercise the exact same one-package-per-process zipimport path a Docker
    container would use, mirroring
    tests/test_cli_export.py::test_exported_pyz_is_self_contained_and_predict_works.
    """
    pkg_path = _build_test_package(tmp_path)
    docker_dir = Path(__file__).resolve().parents[1] / "docker"

    script = f"""
import sys

repo_src = {str(Path(__file__).resolve().parents[1] / "src")!r}
sys.path = [p for p in sys.path if p != repo_src]
sys.path.insert(0, {str(docker_dir)!r})

from harness.load import load_mimosepkg

resolved = load_mimosepkg({str(pkg_path)!r}, {str(tmp_path / "extract")!r}, device="cpu")
assert resolved["manifest"]["model_class"] == "mimose.models.tiny_mimosa.TinyMimosa"

import mimose
assert mimose.__file__.startswith({str(tmp_path / "extract")!r}), mimose.__file__

import torch
model = resolved["model"]
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


def test_run_module_wires_resolved_package_into_run_testing(monkeypatch, tmp_path: Path) -> None:
    import harness.run as harness_run
    import mimose.testing as testing_mod

    fake_resolved = {
        "manifest": {"mimose_version": "0.0.2", "model_class": "pkg.Model", "run_suffix": "fold1"},
        "config": {"model_kwargs": {"num_cls": 4}},
        "model_class": TinyMimosa,
        "model_kwargs": {"num_cls": 4, "input_shape": (16, 16, 16), "features_per_stage": (4, 8, 16, 32)},
        "checkpoint_path": tmp_path / "weights.safetensors",
    }
    monkeypatch.setattr(harness_run, "resolve_package", lambda pkg, extract_dir: fake_resolved)

    captured: dict[str, object] = {}

    def fake_run_testing(**kwargs):
        captured.update(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr(testing_mod, "run_testing", fake_run_testing)

    output_dir = tmp_path / "output"
    harness_run.main(
        [
            "--package",
            str(tmp_path / "run.mimosepkg"),
            "--data-dir",
            str(tmp_path / "data"),
            "--output-dir",
            str(output_dir),
            "--dataset-type",
            "brats18",
            "--split-file",
            str(tmp_path / "split.json"),
            "--fold",
            "1",
        ]
    )

    assert captured["model_class"] is TinyMimosa
    assert captured["model_kwargs"] == fake_resolved["model_kwargs"]
    assert captured["checkpoint_path"] == fake_resolved["checkpoint_path"]
    assert captured["fold"] == 1
    assert captured["output_path"] == output_dir / "test_report.txt"


def test_build_export_package_embeds_preprocessing_config(tmp_path: Path) -> None:
    pkg_path = _build_test_package(tmp_path, preprocessing_config=SAMPLE_PREPROCESSING_CONFIG)

    with zipfile.ZipFile(pkg_path) as zf:
        import json

        manifest = json.loads(zf.read("manifest.json"))

    assert manifest["preprocessing"]["dataset_type"] == "brats18"
    assert manifest["preprocessing"]["crop_min_size"] == [32, 32, 32]


def test_build_export_package_omits_preprocessing_when_not_given(tmp_path: Path) -> None:
    pkg_path = _build_test_package(tmp_path)

    with zipfile.ZipFile(pkg_path) as zf:
        import json

        manifest = json.loads(zf.read("manifest.json"))

    assert "preprocessing" not in manifest


def test_extract_preprocessing_config_requires_all_fields() -> None:
    complete = {
        "dataset_type": "brats18",
        "crop_mode": "non_empty",
        "crop_min_size": (128, 128, 128),
        "clamp_mode": "none",
        "norm_mode": "subject_zscore",
    }
    assert _extract_preprocessing_config(complete) == {
        "dataset_type": "brats18",
        "crop_mode": "non_empty",
        "crop_size": None,
        "crop_min_size": (128, 128, 128),
        "clamp_mode": "none",
        "clamp_percentile": None,
        "clamp_min": None,
        "clamp_max": None,
        "norm_mode": "subject_zscore",
        "norm_min_max_range": None,
        "norm_mean": None,
        "norm_std": None,
    }

    incomplete = {"dataset_type": "brats18", "crop_mode": "non_empty"}
    assert _extract_preprocessing_config(incomplete) is None


def test_preprocess_raw_cases_requires_embedded_config(tmp_path: Path) -> None:
    with pytest.raises(PreprocessingConfigError):
        preprocess_raw_cases(tmp_path / "raw", tmp_path / "out", None)


def test_preprocess_raw_cases_writes_npz_matching_embedded_config(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    for case_id in ("001", "002"):
        _write_raw_case(raw_dir, case_id)

    output_dir = tmp_path / "preprocessed"
    case_ids = preprocess_raw_cases(raw_dir, output_dir, SAMPLE_PREPROCESSING_CONFIG, num_workers=2)

    assert case_ids == ["001", "002"]
    for case_id in case_ids:
        with np.load(output_dir / f"{case_id}.npz") as data:
            assert data["images"].shape[0] == 4
            assert all(dim >= 32 for dim in data["images"].shape[1:])
            assert data["seg"].shape[-3:] == data["images"].shape[-3:]

    split_path = write_test_only_split(case_ids, "brats18", tmp_path / "split.json")
    payload = split_path.read_text()
    assert '"001"' in payload and '"002"' in payload


def test_preprocess_run_entrypoint_never_imports_model_or_torch(tmp_path: Path) -> None:
    """Runs in a fresh interpreter, mirroring the CPU-only preprocessing container:
    proves harness.preprocess_run never needs torch/mamba_ssm importable."""
    pkg_path = _build_test_package(tmp_path, preprocessing_config=SAMPLE_PREPROCESSING_CONFIG)
    raw_dir = tmp_path / "raw"
    _write_raw_case(raw_dir, "001")
    docker_dir = Path(__file__).resolve().parents[1] / "docker"
    output_dir = tmp_path / "output"

    script = f"""
import sys

repo_src = {str(Path(__file__).resolve().parents[1] / "src")!r}
sys.path = [p for p in sys.path if p != repo_src]
sys.path.insert(0, {str(docker_dir)!r})

import harness.preprocess_run as preprocess_run
preprocess_run.main([
    "--package", {str(pkg_path)!r},
    "--raw-data-dir", {str(raw_dir)!r},
    "--output-dir", {str(output_dir)!r},
    "--extract-dir", {str(tmp_path / "extract")!r},
    "--num-workers", "1",
])
assert "torch" not in sys.modules
assert not any(m.startswith("mimose.models") for m in sys.modules)
print("OK")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("OK")
    assert (output_dir / "preprocessed" / "001.npz").is_file()
    assert (output_dir / "split.json").is_file()
