#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

uv sync
uv cache clean
export CUDA_HOME=/homes/admin/spack/opt/spack/linux-ivybridge/cuda-12.6.3-cr3dswdcqnxbg772az4phx3g6qqmegwy
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MAX_JOBS=$(nproc)
uv pip install causal-conv1d>=1.4.0 --no-build-isolation
uv pip install mamba-ssm --no-build-isolation