#!/usr/bin/env bash
# Submit the FOLD 4 combined (train+test) scripts for the whole mimosa_size
# sweep on both BraTS18 and BraTS23 -- one script per size/dataset, each submitted
# N times under --dependency=singleton so the same-named jobs run back-to-back
# (train resumes across the wall limit; the run that finishes training also tests,
# then cancels the queued duplicates).
#
# Usage: bash MIMOSA_SIZES_FOLD4_SBATCHER.sh N
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <N>" >&2
    exit 1
fi

fold=4
sizes=(mimosa_micro mimosa_tiny mimosa_small mimosa_medium mimosa_base mimosa_large mimosa_huge mimosa_gargantuan mimosa_colossal mimosa_titanic mimosa_crazy mimosa_insane mimosa_ultimate)
datasets=(23)
repeats="$1"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for size in "${sizes[@]}"; do
    for ds in "${datasets[@]}"; do
        script="${script_dir}/${size}/all/${size}_${ds}_${fold}.sh"
        if [ ! -f "${script}" ]; then
            echo "Missing ${script}" >&2
            exit 1
        fi
        for ((i = 1; i <= repeats; i++)); do
            echo "Submitting ${script} (${i}/${repeats})"
            sbatch --dependency=singleton "${script}"
        done
    done
done
