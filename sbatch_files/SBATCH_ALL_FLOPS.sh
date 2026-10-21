#!/usr/bin/env bash
# Submits every FLOPs profiling script (sbatch_files/flops/flops_*.sh).
# Run from anywhere; paths are resolved relative to this file.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shopt -s nullglob
scripts=("${script_dir}"/flops/flops_*.sh)

if [ "${#scripts[@]}" -eq 0 ]; then
    echo "No flops_*.sh scripts found under ${script_dir}/flops/" >&2
    exit 1
fi

for script in "${scripts[@]}"; do
    echo "Submitting ${script}"
    sbatch "${script}"
done
