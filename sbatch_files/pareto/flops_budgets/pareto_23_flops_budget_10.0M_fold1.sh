#!/bin/bash
#SBATCH --job-name=PARETO23_FLOPS_B10.0M_FOLD1
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/pareto_23_flops_budget_10.0M_fold1.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/pareto_23_flops_budget_10.0M_fold1.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose

set -euo pipefail

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate

# Resource half of one dense-Pareto point (Phase 9), paired with
# test_budgets/pareto_23_test_budget_10.0M_fold1.sh, which measures Dice at
# the same 10000000 parameter budget. Both write into
# pareto_b10.0M_23, so the flops/summary scripts -- which scan
# "<model>_<dsnum>" dirs for results*.xlsx alongside flops*.csv -- join accuracy
# and cost for this architecture automatically.
#
# --budget replaces the supernet with the standalone network it selects before
# profiling. Without it every budget would report the full-width supernet's
# params/FLOPs/memory/latency, since setting a budget only masks channels
# rather than shrinking the computation.
#
# --checkpoint matters because gate thresholds move during training: the
# architecture a budget selects after training is not the one the analytic
# calibration picks at init.
#
# GPU, because latency and peak VRAM are Pareto axes and are meaningless on CPU.

CKPT="/work/phd_mimose/runs/pareto23_fold1/checkpoints/final_weights_only.safetensors"

if [ ! -f "${CKPT}" ]; then
    echo "Supernet checkpoint not found: ${CKPT}" >&2
    echo "Train it first: sbatch_files/pareto/all/pareto_23_1.sh" >&2
    exit 1
fi

mimose flops \
    --config pareto_23.yaml \
    --device cuda \
    --warmup 3 \
    --checkpoint "${CKPT}" \
    --budget 10000000 \
    --output-path /work/phd_mimose/results/pareto_b10.0M_23/flops.txt
