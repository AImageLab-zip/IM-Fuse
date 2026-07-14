#!/usr/bin/env python3
"""Locate where NaN/Inf first appears in an IMFuse forward pass.

Context: on BRATS2018 fold1 testing, a subset of subjects collapse to
~0.000 WT/TC/ET Dice specifically when T1c+T1+T2 (and worst of all, all
four modalities) are simultaneously active -- while the same subjects
score normally on every other modality mask. That signature (exact-zero,
subject- and mask-specific, not universal) points at NaN/Inf propagation
inside the model rather than a data or config bug.

This script loads a checkpoint, registers a forward hook on every leaf
module, and runs one subject through the sliding-window predict() path
for two masks: one known-good and one known-bad. It stops at the first
module (and, for windowed inputs, the first window) whose output contains
NaN/Inf, and reports the module name/type so the offending op can be
pinned down.

Requires a CUDA device with mamba_ssm's kernels available (i.e. run this
on a cluster node, not the login node).

Usage:
    python scripts/debug_imfuse_nan.py \
        --data-dir /work/phd_mimose/imfuse-preprocessed \
        --checkpoint /work/phd_mimose/runs/imfuse18/checkpoints/final_weights_only.safetensors \
        --subject BraTS-GLI-00318-000 \
        --masks t1c_t1n_t2f t1c_t1n_t2w t1c_t1n_t2f_t2w
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mimose.checkpoints import load_weights_only_checkpoint
from mimose.models.IMFuse import IMFuse, input_patch_size

MODALITY_NAMES = ("t1c", "t1n", "t2f", "t2w")


class NaNDetected(Exception):
    def __init__(self, module_name: str, module_type: str, window: tuple[int, int, int] | None):
        self.module_name = module_name
        self.module_type = module_type
        self.window = window
        super().__init__(f"NaN/Inf produced by {module_name} ({module_type}) window={window}")


def parse_mask(label: str) -> torch.Tensor:
    present = set(label.split("_"))
    unknown = present - set(MODALITY_NAMES)
    if unknown:
        raise ValueError(f"Unknown modality tokens in '{label}': {unknown}")
    return torch.tensor([name in present for name in MODALITY_NAMES], dtype=torch.bool)


def register_nan_hooks(model: torch.nn.Module, current_window: list[tuple[int, int, int] | None]) -> list:
    handles = []

    def make_hook(name: str):
        def hook(module, inputs, output):
            tensors = output if isinstance(output, (tuple, list)) else (output,)
            for t in tensors:
                if not torch.is_tensor(t):
                    continue
                if torch.isnan(t).any() or torch.isinf(t).any():
                    raise NaNDetected(name, type(module).__name__, current_window[0])
        return hook

    for name, module in model.named_modules():
        if name == "":
            continue
        # Only leaf-ish modules to keep the first-offender signal precise.
        if len(list(module.children())) == 0:
            handles.append(module.register_forward_hook(make_hook(name)))
    return handles


def windowed_forward(model: IMFuse, images: torch.Tensor, mask: torch.Tensor, current_window: list) -> None:
    """Mirrors IMFuse.predict's sliding window, but stops at the first NaN."""
    images = model._normalize_predict_input(images)
    _, _, height, width, depth_size = images.shape

    if (height, width, depth_size) == (input_patch_size, input_patch_size, input_patch_size):
        current_window[0] = (0, 0, 0)
        model(images, mask)
        return

    h_starts = model._window_starts(height)
    w_starts = model._window_starts(width)
    d_starts = model._window_starts(depth_size)

    for h in h_starts:
        for w in w_starts:
            for d in d_starts:
                current_window[0] = (h, w, d)
                patch = images[:, :, h:h + input_patch_size, w:w + input_patch_size, d:d + input_patch_size]
                model(patch, mask)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="Dir with <subject>.npz preprocessed files")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .safetensors weights")
    parser.add_argument("--subject", type=str, required=True, help="Subject id, e.g. BraTS-GLI-00318-000")
    parser.add_argument(
        "--masks",
        type=str,
        nargs="+",
        required=True,
        help="Underscore-joined modality masks in t1c/t1n/t2f/t2w naming, "
        "e.g. t1c_t1n_t2f t1c_t1n_t2w t1c_t1n_t2f_t2w",
    )
    parser.add_argument("--interleaved-tokenization", action="store_true", default=True)
    parser.add_argument("--no-interleaved-tokenization", dest="interleaved_tokenization", action="store_false")
    parser.add_argument("--mamba-skip", action="store_true", default=True)
    parser.add_argument("--no-mamba-skip", dest="mamba_skip", action="store_false")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA device (mamba_ssm kernels).")
    device = torch.device("cuda")

    npz_path = args.data_dir / f"{args.subject}.npz"
    with np.load(npz_path) as data:
        images = torch.from_numpy(data["images"]).clone().float().unsqueeze(0).to(device)

    model = IMFuse(
        num_cls=4,
        interleaved_tokenization=args.interleaved_tokenization,
        mamba_skip=args.mamba_skip,
    ).to(device)
    load_weights_only_checkpoint(model, args.checkpoint, device=device)
    model.eval()
    model.is_training = False

    for mask_label in args.masks:
        mask = parse_mask(mask_label).unsqueeze(0).to(device)
        current_window: list = [None]
        handles = register_nan_hooks(model, current_window)
        print(f"\n=== subject={args.subject} mask={mask_label} ===")
        try:
            with torch.no_grad():
                windowed_forward(model, images, mask, current_window)
            print("  no NaN/Inf detected in any leaf-module output.")
        except NaNDetected as exc:
            print(
                f"  FIRST NaN/Inf at module='{exc.module_name}' "
                f"type={exc.module_type} window_start={exc.window}"
            )
        finally:
            for h in handles:
                h.remove()


if __name__ == "__main__":
    main()
