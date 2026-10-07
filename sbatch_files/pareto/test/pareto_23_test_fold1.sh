#!/bin/bash
#SBATCH --job-name=PARETO23_TEST_FOLD1
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/pareto_23_test_fold1.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/pareto_23_test_fold1.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

# Evaluates the supernet at its default (widest) budget -- see Phase 9 for the
# dense sweep across the whole 1.43M..91.4M range.
mimose test --config pareto_23.yaml --fold 1 --run-suffix fold1 --precision bf16
