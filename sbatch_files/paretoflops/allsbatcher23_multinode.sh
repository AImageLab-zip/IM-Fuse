#!/usr/bin/env bash
# Submit every multi-node (train+test) fold script for BraTS23 ("_23_") in
# all_multinode/, each 4x with singleton dependency so the 4 same-named jobs run
# back-to-back (train resumes across the 24h wall limit; the run that finishes
# training also tests).
#
# These jobs ask for 2 nodes at once, so they queue until 2 matching nodes are
# free simultaneously -- expect a longer pending time than the single-node
# scripts in all/.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
all_dir="${script_dir}/all_multinode"

shopt -s nullglob
scripts=("${all_dir}"/*_23_*.sh)
if [ "${#scripts[@]}" -eq 0 ]; then
    echo "No _23_ multi-node scripts found in ${all_dir}" >&2
    exit 1
fi

for script in "${scripts[@]}"; do
    for i in 1 2 3 4; do
        echo "Submitting ${script} (${i}/4)"
        sbatch --dependency=singleton "${script}"
    done
done
