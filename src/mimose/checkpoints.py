from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_model, save_model


def save_weights_only_checkpoint(
    model: torch.nn.Module,
    destination: str | Path,
) -> Path:
    checkpoint_path = Path(destination)
    # Some architectures (e.g. TinyMimosa's nnU-Net-style backbone) alias the
    # same parameter under multiple state_dict keys (shared modules). Plain
    # save_file() rejects that; save_model() detects and dedupes shared
    # tensors, and load_model() reconstructs the aliasing on load.
    save_model(model, str(checkpoint_path))
    return checkpoint_path


def load_weights_only_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> torch.nn.Module:
    path = Path(checkpoint_path)
    if path.suffix != ".safetensors":
        raise ValueError(
            f"Expected a .safetensors weights checkpoint, got: {path}"
        )
    load_model(model, str(path), strict=strict, device=str(device))
    return model
