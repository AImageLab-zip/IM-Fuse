#!/usr/bin/env bash
# Submit every dense-Pareto-sweep FLOPs job (Phase 9): pareto_23, fold 1, one
# job per budget in flops_budgets/. Companion to test_budgets_sbatcher.sh,
# which submits the matching Dice evaluations.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
flops_budgets_dir="${script_dir}/flops_budgets"

shopt -s nullglob
scripts=("${flops_budgets_dir}"/*.sh)
if [ "${#scripts[@]}" -eq 0 ]; then
    echo "No flops-budget scripts found in ${flops_budgets_dir}" >&2
    exit 1
fi

for script in "${scripts[@]}"; do
    echo "Submitting ${script}"
    sbatch "${script}"
done
