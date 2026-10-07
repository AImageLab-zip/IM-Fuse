#!/usr/bin/env bash
# Submit one FLOPs job per mimosa_[size] parameter budget: pareto_23, fold 1,
# one job per script in flops_sizes/.
#
# Sibling of flops_budgets_sbatcher.sh. That one sweeps round log-spaced
# budgets; this one pins each budget to a mimosa_[size] baseline's exact
# parameter count, so the Pareto curve can be read off against those models
# point by point.
#
# mimosa_micro (0.69M params, 4 stages) has no script here: it is below the
# supernet's narrowest subnetwork (~1.43M, 5 stages), so no budget selects it.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
flops_sizes_dir="${script_dir}/flops_sizes"

shopt -s nullglob
scripts=("${flops_sizes_dir}"/*.sh)
if [ "${#scripts[@]}" -eq 0 ]; then
    echo "No flops-size scripts found in ${flops_sizes_dir}" >&2
    exit 1
fi

for script in "${scripts[@]}"; do
    echo "Submitting ${script}"
    sbatch "${script}"
done
