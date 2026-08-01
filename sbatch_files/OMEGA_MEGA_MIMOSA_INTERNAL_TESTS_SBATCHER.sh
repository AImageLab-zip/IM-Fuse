#!/usr/bin/env bash
# Submits every mimosa_[size] internal test job, split one-fold-per-job
# (sbatch_files/mimosa_*/test/*_{18,23}_internal_fold{1,3,5}.sh).
# That's 8 sizes x 2 datasets (BraTS18/23) x 3 folds = 48 jobs.
# Run from anywhere; paths are resolved relative to this file.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shopt -s nullglob
scripts=("${script_dir}"/mimosa_*/test/*_internal_fold*.sh)

if [ "${#scripts[@]}" -eq 0 ]; then
    echo "No mimosa_*_internal_fold*.sh scripts found under ${script_dir}/mimosa_*/test/" >&2
    exit 1
fi

for script in "${scripts[@]}"; do
    echo "Submitting ${script}"
    sbatch "${script}"
done
