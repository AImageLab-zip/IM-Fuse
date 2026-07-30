#!/bin/bash
#SBATCH --job-name=MIMOSA_LARGE23_FOLD_5
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/mimosa_large_23_5.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/mimosa_large_23_5.out
#SBATCH --gres=gpu:2
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

mimose train --config mimosa_large_23.yaml --fold 5 --run-suffix fold5 --seed 0 --try-resume --distributed
mimose test --config mimosa_large_23.yaml --fold 5 --run-suffix fold5 --fp16

# Training completed (we reached the test), so cancel the remaining queued
# resume submissions of this same job from allsbatcher23.sh's singleton chain.
scancel --state=PENDING --name="$SLURM_JOB_NAME"
