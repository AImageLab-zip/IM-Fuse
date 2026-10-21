#!/bin/bash
#SBATCH --job-name=FLOPS_PARETO_SWEEP_GPU
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/flops_pareto_sweep_gpu.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/flops_pareto_sweep_gpu.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose

set -euo pipefail

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate

# Dense Pareto sweep: profile the standalone network the supernet selects at
# each of N log-spaced budgets.
#
# Two things make this different from a plain `mimose flops` run, and both are
# required for the numbers to mean anything:
#
#   --budget-sweep   Setting a budget on the supernet only *masks* channels --
#                    the forward still runs at full width, so params, FLOPs,
#                    peak memory and latency would all come out as the widest
#                    model's numbers at every budget. This flag replaces the
#                    supernet with the extracted standalone SimpleUnet before
#                    profiling, which is the architecture the Pareto plots are
#                    actually about.
#
#   --checkpoint     Gate thresholds move during training, so the architecture a
#                    budget selects before and after training are different
#                    networks. Without this, the sweep profiles the analytic
#                    calibration rather than what was learned.
#
# Runs on GPU because latency and peak VRAM are two of the Pareto axes and are
# meaningless on CPU. Each budget writes its own report, suffixed with the
# realized parameter count, into the config's results_dir.

DATASET="${DATASET:-23}"
FOLD="${FOLD:-1}"
N_BUDGETS="${N_BUDGETS:-24}"

CONFIG="pareto_${DATASET}.yaml"
CKPT="/work/phd_mimose/runs/pareto${DATASET}_fold${FOLD}/checkpoints/final_weights_only.safetensors"

if [ ! -f "${CKPT}" ]; then
    echo "Supernet checkpoint not found: ${CKPT}" >&2
    echo "Train it first (sbatch_files/pareto/all/pareto_${DATASET}_${FOLD}.sh)," >&2
    echo "or set DATASET/FOLD to a run that exists." >&2
    exit 1
fi

echo "=== pareto_${DATASET} fold${FOLD}: ${N_BUDGETS}-point budget sweep ==="
echo "checkpoint: ${CKPT}"

mimose flops \
    --config "${CONFIG}" \
    --device cuda \
    --warmup 3 \
    --checkpoint "${CKPT}" \
    --budget-sweep "${N_BUDGETS}"
