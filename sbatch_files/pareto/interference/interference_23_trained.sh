#!/bin/bash
#SBATCH --job-name=PARETO23_INTERFERENCE_TRAINED
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/interference_23_trained.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/interference_23_trained.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate

# Phase 5 on a trained supernet. Interference at initialization reflects only
# the architecture; the question that matters for multi-budget training is
# whether the shared weights come into conflict once they have specialized.
#
# Uses model_last.pth (full training state) rather than
# final_weights_only.safetensors, so this works on a run still in progress.
mimose interference \
    --config pareto_23.yaml \
    --fold 1 \
    --checkpoint /work/phd_mimose/runs/pareto23_fold1/checkpoints/model_last.pth \
    --n-budgets 7 \
    --n-batches 16 \
    --batch-size 1 \
    --num-workers 8 \
    --precision bf16 \
    --output-dir /work/phd_mimose/results/pareto_23/interference_trained
