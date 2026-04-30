from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def save_weights_only_checkpoint(
    state_dict: OrderedDict[str, torch.Tensor],
    destination: str | Path,
) -> Path:
    checkpoint_path = Path(destination)
    save_file(state_dict, str(checkpoint_path))
    return checkpoint_path


def load_weights_only_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> OrderedDict[str, torch.Tensor]:
    path = Path(checkpoint_path)
    if path.suffix != ".safetensors":
        raise ValueError(
            f"Expected a .safetensors weights checkpoint, got: {path}"
        )
    state_dict = load_file(str(path), device=str(device))
    return OrderedDict(state_dict)
