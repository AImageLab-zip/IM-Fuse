#!/usr/bin/env bash
# Submit every Phase 5 gradient-interference measurement.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
interference_dir="${script_dir}/interference"

shopt -s nullglob
scripts=("${interference_dir}"/*.sh)
if [ "${#scripts[@]}" -eq 0 ]; then
    echo "No interference scripts found in ${interference_dir}" >&2
    exit 1
fi

for script in "${scripts[@]}"; do
    echo "Submitting ${script}"
    sbatch "${script}"
done
