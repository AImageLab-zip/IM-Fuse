#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
train_dir="${script_dir}/train"

shopt -s nullglob
scripts=("${train_dir}"/*_18_*.sh)
# Exclude seed 67 runs from this bulk submitter
filtered=()
for s in "${scripts[@]}"; do
    [[ "$s" == *_18_67.sh ]] || filtered+=("$s")
done
scripts=("${filtered[@]}")

if [ "${#scripts[@]}" -eq 0 ]; then
    echo "No _18_ train scripts found in ${train_dir}" >&2
    exit 1
fi

for script in "${scripts[@]}"; do
    echo "Submitting ${script}"
    sbatch "${script}"
done
