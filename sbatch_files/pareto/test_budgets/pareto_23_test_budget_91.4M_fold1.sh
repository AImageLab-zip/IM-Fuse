#!/bin/bash
#SBATCH --job-name=PARETO23_TEST_B91.4M_FOLD1
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/pareto_23_test_budget_91.4M_fold1.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/pareto_23_test_budget_91.4M_fold1.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

# One point of the dense Pareto sweep referenced by test/pareto_23_test_fold1.sh
# ("Phase 9"): evaluates the pareto_23 supernet at a fixed 91400000 parameter
# budget instead of the default (widest) one. Writes into its own
# "pareto_b91.4M_23" results directory -- matching the "<model>_<dsnum>"
# layout build_reproduction_comparison.py's discover() scans for -- so the
# recap/summary scripts pick up each budget as its own row.
mimose test --config pareto_23.yaml --fold 1 --run-suffix fold1 \
    --budget 91400000 \
    --output-path /work/phd_mimose/results/pareto_b91.4M_23/results.txt \
    --precision bf16
