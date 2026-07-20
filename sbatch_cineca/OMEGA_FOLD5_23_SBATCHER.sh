#!/usr/bin/env bash
# Submit the FOLD 5 combined (train+test) script for EVERY model's BraTS23 run under
# sbatch_files/*/all/ (i.e. *_23_5.sh only). Each script is submitted 4x with a combined
# singleton + after dependency: singleton so the 4 same-named jobs run back-to-back (train
# resumes across the 24h wall limit; the run that finishes training also tests, then cancels
# the remaining queued duplicates), and after the BraTS18 jobs listed in brats18_jobs.txt so
# BraTS23 doesn't start competing for GPUs until BraTS18 training has finished. Run from
# anywhere; paths are resolved relative to this file.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
jobs_file="${script_dir}/brats18_jobs.txt"

if [ ! -f "${jobs_file}" ]; then
    echo "Missing ${jobs_file}" >&2
    exit 1
fi

after_ids="$(tr -d '[:space:]' < "${jobs_file}")"
if [ -z "${after_ids}" ]; then
    echo "${jobs_file} is empty" >&2
    exit 1
fi
dependency="singleton,after:${after_ids}"

shopt -s nullglob
scripts=("${script_dir}"/*/all/*_23_5.sh)
if [ "${#scripts[@]}" -eq 0 ]; then
    echo "No fold-5 BraTS23 (*_23_5.sh) all scripts found under ${script_dir}/*/all/" >&2
    exit 1
fi

for script in "${scripts[@]}"; do
    for i in 1 2 3 4; do
        echo "Submitting ${script} (${i}/4) with dependency=${dependency}"
        sbatch --dependency="${dependency}" "${script}"
    done
done
