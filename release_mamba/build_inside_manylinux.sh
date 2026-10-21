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

# `uv venv` doesn't install pip into the venv (not even the module), so a bare
# `pip` on PATH would silently fall through to whatever pip the container
# image provides globally -- a different Python/torch/toolchain entirely
# (this is why earlier builds produced a cp311 wheel despite this venv being
# 3.12, and pulled in a mismatched glibc). Bootstrap pip here and call it only
# via `python -m pip` below so every build step unambiguously uses this venv.
python -m ensurepip
# packaging/setuptools/wheel already come with torch's env; ninja speeds up the
# build and is a pure PyPI package (unlike torch, PyPI itself is reachable here).
python -m pip install --quiet ninja setuptools wheel

CUDA_HOME="$(ls -d /usr/local/cuda-12.8 2>/dev/null || ls -d /usr/local/cuda)"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
# srun/pyxis forwards the submitting shell's environment into the container
# unsanitized. On this cluster that shell's LD_LIBRARY_PATH includes the host's
# own glibc (/usr/lib/x86_64-linux-gnu, newer than manylinux_2_28's baseline);
# ld consults LD_LIBRARY_PATH as a link-time search fallback too, so leaving it
# in place lets the extensions link against host glibc symbols (e.g. GLIBC_2.32)
# and fail the manylinux_2_28 check below. Start clean instead of appending.
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64"
unset LIBRARY_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH

echo "Using CUDA_HOME=${CUDA_HOME}"
nvcc --version
echo "gcc: $(command -v gcc) ($(gcc -dumpversion))"
echo "g++: $(command -v g++) ($(g++ -dumpversion))"
echo "ld:  $(command -v ld) ($(ld --version | { head -1 || true; }))"
ldd --version | { head -1 || true; }

# Compile the __libc_single_threaded shim (see glibc_compat_shim.c) with the
# same compiler that will link the extensions.
gcc -c -fPIC -o glibc_compat_shim.o glibc_compat_shim.c

# torch's cpp_extension builds these packages via ninja (installed above),
# which constructs its own link command and does NOT reliably forward the
# shell's $LDFLAGS the way plain distutils/setuptools does -- so the shim
# object above was silently dropped from the final link despite LDFLAGS.
# CC/CXX, on the other hand, are read by both the ninja and distutils build
# paths (they need it just to know what compiler to invoke at all), so wrap
# the real compilers to unconditionally append the shim object to every
# invocation instead. Passing an extra .o file to a compile-only (`-c`)
# invocation is harmless (gcc just warns it's unused), so this is safe to
# apply blindly to every compile *and* link call.
REAL_CC="$(command -v gcc)"
REAL_CXX="$(command -v g++)"
WRAPPER_DIR="${PWD}/cc_wrappers"
mkdir -p "${WRAPPER_DIR}"
cat > "${WRAPPER_DIR}/cc" <<EOF
#!/usr/bin/env bash
exec "${REAL_CC}" "\$@" "${PWD}/glibc_compat_shim.o"
EOF
cat > "${WRAPPER_DIR}/cxx" <<EOF
#!/usr/bin/env bash
exec "${REAL_CXX}" "\$@" "${PWD}/glibc_compat_shim.o"
EOF
chmod +x "${WRAPPER_DIR}/cc" "${WRAPPER_DIR}/cxx"
export CC="${WRAPPER_DIR}/cc"
export CXX="${WRAPPER_DIR}/cxx"
# Keep LDFLAGS too, in case any step does honor it -- harmless either way.
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

python -m pip wheel "${CAUSAL_SDIST}" --no-build-isolation --no-deps -w "${WHEELHOUSE}"
python -m pip wheel "${MAMBA_SDIST}" --no-build-isolation --no-deps -w "${WHEELHOUSE}"

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
        echo "   symbols pulling in glibc > 2.28:"
        objdump -T "${so}" 2>/dev/null | grep -E 'GLIBC_2\.(29|3[0-9])' | awk '{print "   " $NF}' | sort -u
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
