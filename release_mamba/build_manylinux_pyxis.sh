#!/usr/bin/env bash
set -euo pipefail

# Builds causal-conv1d / mamba-ssm wheels for this cluster inside a
# manylinux_2_28 (glibc 2.28) CUDA 12.8 container, via Slurm's pyxis plugin.
# Compilation only needs CPU + nvcc, so this runs on the CPU-only serial
# partition and doesn't compete with GPU allocations.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-docker://pytorch/manylinux2_28-builder:cuda12.8-main}"

# Mount $HOME at the same absolute path (rather than remapping the project to
# /workspace) because .venv/bin/python is a symlink into uv's managed
# interpreter under ~/.local/share/uv, outside the project root. Keeping the
# path identical inside and outside the container lets that symlink resolve.
# We reuse this venv's already-installed torch==2.9.1+cu128 instead of
# pip-installing torch again inside the container, since compute nodes on
# this cluster can't reach download.pytorch.org.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # Already inside an allocation (interactive shell): launch the container
  # as a job step in it, no resource-request flags needed.
  srun \
    --container-image="${IMAGE}" \
    --container-mounts="${HOME}:${HOME}" \
    --container-workdir="${SCRIPT_DIR}" \
    bash "${SCRIPT_DIR}/build_inside_manylinux.sh"
else
  # No allocation yet: request one (CPU-only, doesn't need a GPU node).
  PARTITION="${PARTITION:-all_serial}"
  srun \
    --partition="${PARTITION}" \
    --cpus-per-task=8 \
    --mem=32G \
    --time=01:00:00 \
    --job-name=mamba-wheel-build \
    --container-image="${IMAGE}" \
    --container-mounts="${HOME}:${HOME}" \
    --container-workdir="${SCRIPT_DIR}" \
    bash "${SCRIPT_DIR}/build_inside_manylinux.sh"
fi
