#!/bin/bash
#SBATCH --job-name=PARETO23_INTERFERENCE_INIT
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/interference_23_init.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/interference_23_init.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate

# Phase 5, measured at initialization -- no checkpoint. This is the control the
# trained measurement is read against: any change in interference over training
# is only meaningful relative to where it started.
#
# batch_size 1 keeps peak memory at one full-width forward while the 7 budgets
# are measured back to back; --n-batches averages away minibatch noise and also
# supplies the per-budget noise floor.
mimose interference \
    --config pareto_23.yaml \
    --fold 1 \
    --n-budgets 7 \
    --n-batches 16 \
    --batch-size 1 \
    --num-workers 8 \
    --precision bf16 \
    --output-dir /work/phd_mimose/results/pareto_23/interference_init
