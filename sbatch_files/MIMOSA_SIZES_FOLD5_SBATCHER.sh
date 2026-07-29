#!/usr/bin/env bash
# Submit the FOLD 5 combined (train+test) scripts for the whole size sweep on both
# BraTS18 and BraTS23 -- 8 scripts, each 4x under --dependency=singleton so the
# same-named jobs run back-to-back (train resumes across the wall limit; the run that
# finishes training also tests, then cancels the queued duplicates).
set -euo pipefail

fold=5
sizes=(mimosa_small mimosa_medium mimosa_large mimosa_huge)
datasets=(18 23)
repeats=4

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
