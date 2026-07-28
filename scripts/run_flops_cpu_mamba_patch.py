#!/usr/bin/env python
"""Runs `mimose` CLI commands with mamba_ssm patched to its pure-PyTorch
reference path, so Mamba-based models (e.g. IMFuse) can run on CPU.

By default, mamba_ssm.modules.mamba_simple.Mamba unconditionally reaches for
compiled CUDA kernels (causal_conv1d_fn -> fused mamba_inner_fn, and
selective_scan_fn -> selective_scan_cuda), which raise on CPU tensors. This
patches only the module-level references those kernels are looked up
through -- it does not touch any file in this repo or in mamba_ssm's
installed package -- forcing Mamba.forward to take its slow branch (a plain
nn.Conv1d) and use mamba_ssm's own selective_scan_ref implementation
instead of the CUDA extension.

Usage: same arguments as `mimose`, e.g.
    python scripts/run_flops_cpu_mamba_patch.py flops --config imfuse_23.yaml --device cpu
"""

from mamba_ssm.modules import mamba_simple
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

mamba_simple.causal_conv1d_fn = None
mamba_simple.causal_conv1d_update = None
mamba_simple.selective_scan_fn = selective_scan_ref

from mimose.cli import app  # noqa: E402  (import after patch is applied)

if __name__ == "__main__":
    app()
