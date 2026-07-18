#!/usr/bin/env bash
# Submits every model's brats23-trained internal test script
# (sbatch_files/*/test/*_23_internal.sh). Run from anywhere; paths are
# resolved relative to this file.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shopt -s nullglob
scripts=("${script_dir}"/*/test/*_23_internal.sh)

if [ "${#scripts[@]}" -eq 0 ]; then
    echo "No *_23_internal.sh scripts found under ${script_dir}/*/test/" >&2
    exit 1
fi

for script in "${scripts[@]}"; do
    echo "Submitting ${script}"
    sbatch "${script}"
done
