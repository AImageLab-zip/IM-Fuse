"""Load a `.mimosepkg` archive (see `mimose export`, `src/mimose/model_export.py`)
in a fresh interpreter with no dependency on the outer `mimose` package being
installed -- only `code/mimose.pyz` from inside the archive itself.

See `.claude/docker-inference-pipeline-guide.md` for the full design rationale.
Only ever run this in a process where `mimose` has not already been imported
from anywhere else -- see the note in `run()` below.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any


class PackageIntegrityError(ValueError):
    """Raised when a `.mimosepkg` archive fails a checksum or structure check."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_package(pkg_path: Path, extract_dir: Path) -> dict[str, Any]:
    """Extract and verify a `.mimosepkg` archive, and put its bundled `mimose.pyz`
    on `sys.path` (see the module docstring for the one-package-per-process rule).

    Returns a dict with keys: `manifest`, `config`, `checkpoint_path`, `pyz_path`.
    Deliberately does **not** import anything from `mimose.models` -- callers
    that only need `mimose.preprocessing`/`mimose.enums` (i.e. the CPU-only
    preprocessing container, which has no reason to require torch/mamba_ssm to
    be importable) should stop here. Use `resolve_package` below when the
    model class itself is needed (the GPU testing container).
    """
    pkg_path = Path(pkg_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(pkg_path) as zf:
        names = set(zf.namelist())
        required = {"manifest.json", "config.json", "weights/final_weights_only.safetensors", "code/mimose.pyz"}
        missing = required - names
        if missing:
            raise PackageIntegrityError(f"{pkg_path} is missing required members: {sorted(missing)}")

        manifest = json.loads(zf.read("manifest.json"))
        config = json.loads(zf.read("config.json"))
        zf.extract("code/mimose.pyz", extract_dir)
        zf.extract("weights/final_weights_only.safetensors", extract_dir)

    checkpoint_path = extract_dir / "weights" / "final_weights_only.safetensors"

    digest = _sha256_file(checkpoint_path)
    expected = manifest.get("checkpoint_sha256", "")
    if digest != expected:
        raise PackageIntegrityError(
            f"checkpoint sha256 mismatch for {pkg_path}: manifest says {expected!r}, got {digest!r}"
        )

    pyz_path = extract_dir / "code" / "mimose.pyz"

    if "mimose" in sys.modules:
        existing = getattr(sys.modules["mimose"], "__file__", "") or ""
        if str(pyz_path) not in existing:
            raise RuntimeError(
                "a different 'mimose' module is already imported in this process "
                f"(from {existing!r}) -- extract_package()/resolve_package() must "
                "run in a fresh interpreter per package, since zipimport can't "
                "override an already-cached top-level module. See "
                "docker-inference-pipeline-guide.md section 2."
            )
    else:
        sys.path.insert(0, str(pyz_path))

    return {
        "manifest": manifest,
        "config": config,
        "checkpoint_path": checkpoint_path,
        "pyz_path": pyz_path,
    }


def resolve_package(pkg_path: Path, extract_dir: Path) -> dict[str, Any]:
    """Like `extract_package`, but also imports the model class named in the
    manifest (`mimose.models.*`, which requires torch and, for some
    architectures, mamba_ssm/causal_conv1d to be importable) -- use this only
    in the GPU testing container. Adds `model_class`/`model_kwargs` to the
    returned dict.
    """
    resolved = extract_package(pkg_path, extract_dir)
    manifest, config = resolved["manifest"], resolved["config"]

    module_path, class_name = manifest["model_class"].rsplit(".", 1)
    model_class = getattr(importlib.import_module(module_path), class_name)
    resolved["model_class"] = model_class
    resolved["model_kwargs"] = config.get("model_kwargs", {})
    return resolved


def load_mimosepkg(pkg_path: Path, extract_dir: Path, *, device: str = "cpu") -> dict[str, Any]:
    """Like `resolve_package`, but also constructs the model and loads its
    weights, returning a ready-to-call `AbstractModel` under the `model` key.
    """
    resolved = resolve_package(pkg_path, extract_dir)
    model = resolved["model_class"](**resolved["model_kwargs"]).eval()

    from mimose.checkpoints import load_weights_only_checkpoint

    load_weights_only_checkpoint(model, resolved["checkpoint_path"], device=device, strict=True)
    resolved["model"] = model.to(device)
    return resolved
