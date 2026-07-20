#!/usr/bin/env bash
# Submit the FOLD 1 combined (train+test) script for EVERY model's BraTS18 run under
# sbatch_files/*/all/ (i.e. *_18_1.sh only). Each script is submitted 4x with singleton
# dependency so the 4 same-named jobs run back-to-back (train resumes across the 24h wall
# limit; the run that finishes training also tests, then cancels the remaining queued
# duplicates). Run from anywhere; paths are resolved relative to this file.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shopt -s nullglob
scripts=("${script_dir}"/*/all/*_18_1.sh)
if [ "${#scripts[@]}" -eq 0 ]; then
    echo "No fold-1 BraTS18 (*_18_1.sh) all scripts found under ${script_dir}/*/all/" >&2
    exit 1
fi

for script in "${scripts[@]}"; do
    for i in 1 2 3 4; do
        echo "Submitting ${script} (${i}/4)"
        sbatch --dependency=singleton "${script}"
    done
done
