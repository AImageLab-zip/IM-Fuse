#!/usr/bin/env bash
# Submit ONLY the fold-2 and fold-4 combined (train+test) scripts for
# BraTS18 ("_18_") in all/ -- the folds added after the original
# 1/3/5 sweep. Companion to allsbatcher18.sh, which submits every fold;
# use this one to top up an existing 1/3/5 run without resubmitting it.
# Each is submitted 4x under --dependency=singleton so the same-named
# jobs run back-to-back (train resumes across the wall limit; the run that
# finishes training also tests).
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
all_dir="${script_dir}/all"

shopt -s nullglob
scripts=("${all_dir}"/*_18_2.sh "${all_dir}"/*_18_4.sh)
if [ "${#scripts[@]}" -eq 0 ]; then
    echo "No fold-2/4 _18_ all scripts found in ${all_dir}" >&2
    exit 1
fi

for script in "${scripts[@]}"; do
    for i in 1 2 3 4; do
        echo "Submitting ${script} (${i}/4)"
        sbatch --dependency=singleton "${script}"
    done
done
