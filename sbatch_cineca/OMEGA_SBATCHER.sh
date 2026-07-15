#!/usr/bin/env bash
# Submits the BraTS18 ("_18_") train scripts for every model directory under
# sbatch_files/. Run from anywhere; paths are resolved relative to this file.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shopt -s nullglob
scripts=("${script_dir}"/*/train/*_18_*.sh)

if [ "${#scripts[@]}" -eq 0 ]; then
    echo "No _18_ train scripts found under ${script_dir}/*/train/" >&2
    exit 1
fi

for script in "${scripts[@]}"; do
    echo "Submitting ${script}"
    sbatch "${script}"
done
