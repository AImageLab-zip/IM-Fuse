#!/bin/bash
#SBATCH --job-name=MIMOSA_CRAZY18_FOLD_2
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/mimosa_crazy_18_2.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/mimosa_crazy_18_2.out
#SBATCH --gres=gpu:2
#SBATCH --account=phd_mimose
#SBATCH --constraint=gpu_RTXPro6000B_96G

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

mimose train --config mimosa_crazy_18.yaml --fold 2 --run-suffix fold2 --seed 0 --try-resume --distributed --lr 0.0004 --batch-size 2
mimose test --config mimosa_crazy_18.yaml --fold 2 --run-suffix fold2 --precision bf16

# Training completed (we reached the test), so cancel the remaining queued
# resume submissions of this same job from allsbatcher18.sh's singleton chain.
scancel --state=PENDING --name="$SLURM_JOB_NAME"
