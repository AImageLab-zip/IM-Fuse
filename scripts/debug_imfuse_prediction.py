#!/usr/bin/env python3
"""Inspect what IMFuse actually predicts for a subject/mask that collapses to ~0 Dice.

Follow-up to debug_imfuse_nan.py: that script found no NaN/Inf anywhere in the
forward pass for the masks that collapse to ~0.000 Dice on BRATS2018 fold1
(e.g. t1c_t1n_t2f_t2w on BraTS-GLI-00318-000), yet WT/TC/ET *and* every raw
label (NCR/NET, Edema, Enhancing) collapse together -- i.e. the whole
prediction goes blank, not one class. Since the softmax is finite, this is a
"confidently wrong" prediction rather than numerical garbage, so we need to
look at what class it actually predicts.

This script runs the real model.predict() path (the same one run_testing
uses) for one subject under a working mask and a collapsing mask, then
reports:
  - predicted-class voxel histogram (are voxels landing on background only?)
  - per-class max softmax confidence (low confidence would suggest genuine
    uncertainty; high confidence at the wrong class suggests a systematic
    bug rather than a hard case)
  - centroid of predicted foreground vs. centroid of ground-truth foreground
    (large offset would suggest a spatial/orientation bug, e.g. patches
    stitched at the wrong offset for that mask)

Usage:
    python scripts/debug_imfuse_prediction.py \
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
from mimose.models.IMFuse import IMFuse

MODALITY_NAMES = ("t1c", "t1n", "t2f", "t2w")
CLASS_NAMES = ("background", "NCR_NET", "Edema", "Enhancing")


def parse_mask(label: str) -> torch.Tensor:
    present = set(label.split("_"))
    unknown = present - set(MODALITY_NAMES)
    if unknown:
        raise ValueError(f"Unknown modality tokens in '{label}': {unknown}")
    return torch.tensor([name in present for name in MODALITY_NAMES], dtype=torch.bool)


def centroid(mask_np: np.ndarray) -> tuple[float, float, float] | None:
    coords = np.argwhere(mask_np)
    if coords.size == 0:
        return None
    c = coords.mean(axis=0)
    return float(c[0]), float(c[1]), float(c[2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--masks", type=str, nargs="+", required=True)
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
        seg = torch.from_numpy(data["seg"]).clone().long()  # [1, H, W, D], labels 0..3

    seg_np = seg.squeeze(0).numpy()
    gt_centroid = centroid(seg_np > 0)
    gt_voxel_count = int((seg_np > 0).sum())

    model = IMFuse(
        num_cls=4,
        interleaved_tokenization=args.interleaved_tokenization,
        mamba_skip=args.mamba_skip,
    ).to(device)
    load_weights_only_checkpoint(model, args.checkpoint, device=device)
    model.eval()
    model.is_training = False

    print(f"subject={args.subject}  gt_foreground_voxels={gt_voxel_count}  gt_centroid={gt_centroid}")

    for mask_label in args.masks:
        mask = parse_mask(mask_label).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model.predict(images, mask)  # [1, 4, H, W, D] softmax probs

        probs = output[0]  # [4, H, W, D]
        pred = probs.argmax(dim=0)  # [H, W, D]
        pred_np = pred.cpu().numpy()

        hist = {name: int((pred_np == i).sum()) for i, name in enumerate(CLASS_NAMES)}
        total_voxels = pred_np.size

        # Mean confidence (max softmax prob) restricted to voxels predicted as
        # each foreground class, so we can tell "low confidence everywhere"
        # (genuine uncertainty) apart from "high confidence at the wrong class".
        max_probs = probs.max(dim=0).values.cpu().numpy()
        conf_by_class = {}
        for i, name in enumerate(CLASS_NAMES):
            sel = pred_np == i
            conf_by_class[name] = float(max_probs[sel].mean()) if sel.any() else float("nan")

        pred_fg_centroid = centroid(pred_np > 0)
        pred_fg_voxels = int((pred_np > 0).sum())

        print(f"\n--- mask={mask_label} ---")
        print(f"  predicted voxel histogram: " + ", ".join(f"{k}={v}" for k, v in hist.items()))
        print(f"  fraction background: {hist['background'] / total_voxels:.6f}")
        print(f"  mean max-softmax confidence by predicted class: " +
              ", ".join(f"{k}={v:.4f}" for k, v in conf_by_class.items()))
        print(f"  predicted foreground voxels: {pred_fg_voxels}  centroid: {pred_fg_centroid}")
        if pred_fg_centroid is not None and gt_centroid is not None:
            offset = tuple(round(a - b, 1) for a, b in zip(pred_fg_centroid, gt_centroid))
            print(f"  centroid offset (pred - gt): {offset}")


if __name__ == "__main__":
    main()
