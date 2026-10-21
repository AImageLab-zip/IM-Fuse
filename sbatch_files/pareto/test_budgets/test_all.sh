#!/usr/bin/env bash
# Submit every per-budget test job in this directory (one sbatch each -- these
# are single-shot evaluations, so no singleton chaining like allsbatcher*.sh).
# Scripts are submitted in increasing budget order (1.4M ... 91.4M).
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shopt -s nullglob
scripts=("${script_dir}"/pareto_*_test_budget_*.sh)
if [ "${#scripts[@]}" -eq 0 ]; then
    echo "No test budget scripts found in ${script_dir}" >&2
    exit 1
fi

# sort by the numeric budget embedded in the filename (…_budget_<N>M_…)
mapfile -t scripts < <(
    for s in "${scripts[@]}"; do
        budget="$(basename "$s" | sed -E 's/.*_budget_([0-9.]+)M_.*/\1/')"
        printf '%s\t%s\n' "${budget}" "${s}"
    done | sort -g -k1,1 | cut -f2-
)

for script in "${scripts[@]}"; do
    echo "Submitting $(basename "${script}")"
    sbatch "${script}"
done
