#!/bin/bash
#SBATCH --job-name=MIMOSA_ULTIMATE23_FOLD_4
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/mimosa_ultimate_23_4.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/mimosa_ultimate_23_4.out
#SBATCH --gres=gpu:2
#SBATCH --account=phd_mimose
#SBATCH --constraint=gpu_RTXPro6000B_96G

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

mimose train --config mimosa_ultimate_23.yaml --fold 4 --run-suffix fold4 --seed 0 --try-resume --lr 0.0004 --batch-size 2 --distributed
mimose test --config mimosa_ultimate_23.yaml --fold 4 --run-suffix fold4 --precision bf16

# Training completed (we reached the test), so cancel the remaining queued
# resume submissions of this same job from allsbatcher23.sh's singleton chain.
scancel --state=PENDING --name="$SLURM_JOB_NAME"
