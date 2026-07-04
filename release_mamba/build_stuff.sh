#!/usr/bin/env bash
set -euo pipefail

# Build rebuilt wheels for the Mamba dependencies with uv.
#
# Expected source archives:
#   sources/causal_conv1d-1.6.2.post1.tar.gz
#   sources/mamba_ssm-2.3.2.post1.tar.gz

#CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
WHEELHOUSE="${WHEELHOUSE:-wheelhouse}"
SOURCES_DIR="${SOURCES_DIR:-sources}"
PYTHON_BIN="${PYTHON_BIN:-3.12.13}"

#module load "${CUDA_MODULE}"

if command -v spack >/dev/null 2>&1; then
  CUDA_HOME="$(spack location -i cuda@12.6.3 target=x86_64_v2)"
else
  CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc not found after loading ${CUDA_MODULE}" >&2
    exit 1
fi

export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "${WHEELHOUSE}"
mkdir -p "${SOURCES_DIR}"

CAUSAL_SDIST="${SOURCES_DIR}/causal_conv1d-1.6.2.post1.tar.gz"
MAMBA_SDIST="${SOURCES_DIR}/mamba_ssm-2.3.2.post1.tar.gz"

if [[ ! -f "${CAUSAL_SDIST}" ]]; then
    echo "missing source archive: ${CAUSAL_SDIST}" >&2
    exit 1
fi

if [[ ! -f "${MAMBA_SDIST}" ]]; then
    echo "missing source archive: ${MAMBA_SDIST}" >&2
    exit 1
fi

uv build "${CAUSAL_SDIST}" \
    --wheel \
    --out-dir "${WHEELHOUSE}" \
    --python "${PYTHON_BIN}" \
    --no-build-isolation

uv build "${MAMBA_SDIST}" \
    --wheel \
    --out-dir "${WHEELHOUSE}" \
    --python "${PYTHON_BIN}" \
    --no-build-isolation

echo
echo "Built wheels:"
find "${WHEELHOUSE}" -maxdepth 1 -type f -name '*.whl' | sort
