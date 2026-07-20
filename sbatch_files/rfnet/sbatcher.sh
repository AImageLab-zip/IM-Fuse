#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
train_dir="${script_dir}/train"

shopt -s nullglob
scripts=("${train_dir}"/*.sh)

if [ "${#scripts[@]}" -eq 0 ]; then
    echo "No train scripts found in ${train_dir}" >&2
    exit 1
fi

for script in "${scripts[@]}"; do
    echo "Submitting ${script}"
    sbatch "${script}"
done
