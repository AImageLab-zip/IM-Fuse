#!/usr/bin/env bash
# Submits every model's preprocess_internal.sh under sbatch_files/*/. Run from
# anywhere; paths are resolved relative to this file.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shopt -s nullglob
scripts=("${script_dir}"/*/preprocess_internal.sh)

if [ "${#scripts[@]}" -eq 0 ]; then
    echo "No preprocess_internal.sh scripts found under ${script_dir}/*/" >&2
    exit 1
fi

for script in "${scripts[@]}"; do
    echo "Submitting ${script}"
    sbatch "${script}"
done
