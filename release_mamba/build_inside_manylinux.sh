#!/usr/bin/env bash
set -euo pipefail

# Runs *inside* the pytorch/manylinux-builder container (via pyxis/enroot).
# Builds causal-conv1d and mamba-ssm wheels against manylinux_2_28's glibc
# baseline so the result is portable across this cluster's nodes.

cd "$(dirname "${BASH_SOURCE[0]}")"

# Reuse the project's existing venv instead of pip-installing torch fresh:
# compute nodes on this cluster can't reach download.pytorch.org, and this
# venv already has the matching torch==2.9.1+cu128. Its bin/python is a
# symlink into uv's managed interpreter under $HOME/.local/share/uv, so $HOME
# must be mounted into the container at the same absolute path for this to resolve.
VENV_DIR="$(cd .. && pwd)/.venv"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "expected project venv not found at ${VENV_DIR} (is \$HOME mounted at the same path in the container?)" >&2
    exit 1
fi
source "${VENV_DIR}/bin/activate"

python -c "import torch; print('torch', torch.__version__)"
# packaging/setuptools/wheel already come with torch's env; ninja speeds up the
# build and is a pure PyPI package (unlike torch, PyPI itself is reachable here).
pip install --quiet ninja setuptools wheel

CUDA_HOME="$(ls -d /usr/local/cuda-12.8 2>/dev/null || ls -d /usr/local/cuda)"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "Using CUDA_HOME=${CUDA_HOME}"
nvcc --version

# Compile the __libc_single_threaded shim (see glibc_compat_shim.c) with the
# same compiler that will link the extensions, and feed it in via LDFLAGS -
# distutils/setuptools append $LDFLAGS to the final link command, so this
# gets pulled into both packages' .so files without patching their setup.py.
gcc -c -fPIC -o glibc_compat_shim.o glibc_compat_shim.c
export LDFLAGS="${PWD}/glibc_compat_shim.o ${LDFLAGS:-}"

WHEELHOUSE="wheelhouse"
SOURCES_DIR="sources"
rm -f "${WHEELHOUSE}"/causal_conv1d-*.whl "${WHEELHOUSE}"/mamba_ssm-*.whl
mkdir -p "${WHEELHOUSE}"

CAUSAL_SDIST="${SOURCES_DIR}/causal_conv1d-1.6.2.post1.tar.gz"
MAMBA_SDIST="${SOURCES_DIR}/mamba_ssm-2.3.2.post1.tar.gz"

for f in "${CAUSAL_SDIST}" "${MAMBA_SDIST}"; do
    if [[ ! -f "${f}" ]]; then
        echo "missing source archive: ${f}" >&2
        exit 1
    fi
done

pip wheel "${CAUSAL_SDIST}" --no-build-isolation --no-deps -w "${WHEELHOUSE}"
pip wheel "${MAMBA_SDIST}" --no-build-isolation --no-deps -w "${WHEELHOUSE}"

echo
echo "Built wheels:"
find "${WHEELHOUSE}" -maxdepth 1 -type f -name '*.whl' | sort

echo
echo "Checking glibc symbol versions referenced by the extensions (must be <= 2.28):"
too_new=0
for whl in "${WHEELHOUSE}"/causal_conv1d-*.whl "${WHEELHOUSE}"/mamba_ssm-*.whl; do
    tmpdir="$(mktemp -d)"
    python -m zipfile -e "${whl}" "${tmpdir}"
    find "${tmpdir}" -name '*.so' -print0 | while IFS= read -r -d '' so; do
        echo "-- ${so}"
        objdump -T "${so}" 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sort -Vu | tail -3
    done
    if objdump -T $(find "${tmpdir}" -name '*.so') 2>/dev/null | grep -oE 'GLIBC_2\.(29|3[0-9])' >/dev/null; then
        echo "!! ${whl} still references glibc > 2.28"
        too_new=1
    fi
    rm -rf "${tmpdir}"
done

if [[ "${too_new}" -eq 1 ]]; then
    echo "FAILED: one or more wheels still require glibc newer than 2.28" >&2
    exit 1
fi
echo "OK: all extensions are within the glibc 2.28 baseline."
