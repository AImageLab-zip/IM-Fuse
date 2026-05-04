#!/bin/bash
echo REMEMBER TO REMOVE THE HARDCODED VARS
export CUDA_HOME=/homes/admin/spack/opt/spack/linux-ivybridge/cuda-12.6.3-cr3dswdcqnxbg772az4phx3g6qqmegwy
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MAX_JOBS=$(nproc)
export NVCC_THREADS=$(nproc)

uv sync
#uv cache clean

CUDA_BUILD_START=$(date +%s)

uv pip install "causal-conv1d>=1.4.0" --no-build-isolation -v 2>&1 | grep --line-buffered -E "^\[|^ninja|^nvcc|^c\+\+"
uv pip install mamba-ssm --no-build-isolation -v 2>&1 | grep --line-buffered -E "^\[|^ninja|^nvcc|^c\+\+"

CUDA_BUILD_END=$(date +%s)
ELAPSED=$(( CUDA_BUILD_END - CUDA_BUILD_START ))

if (( ELAPSED >= 60 )); then
  MINUTES=$(( ELAPSED / 60 ))
  SECONDS=$(( ELAPSED % 60 ))
  echo "CUDA build completed in ${MINUTES}m ${SECONDS}s"
else
  echo "CUDA build completed in ${ELAPSED}s"
fi
