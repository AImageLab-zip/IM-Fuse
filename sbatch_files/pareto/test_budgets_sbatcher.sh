#!/usr/bin/env bash
# Submit every dense-Pareto-sweep test job (Phase 9): pareto_23, fold 1, one
# job per budget in test_budgets/.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
test_budgets_dir="${script_dir}/test_budgets"

shopt -s nullglob
scripts=("${test_budgets_dir}"/*.sh)
if [ "${#scripts[@]}" -eq 0 ]; then
    echo "No test-budget scripts found in ${test_budgets_dir}" >&2
    exit 1
fi

for script in "${scripts[@]}"; do
    echo "Submitting ${script}"
    sbatch "${script}"
done
