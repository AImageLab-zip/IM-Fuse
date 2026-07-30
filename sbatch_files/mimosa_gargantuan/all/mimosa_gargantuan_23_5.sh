#!/bin/bash
#SBATCH --job-name=MIMOSA_GARGANTUAN23_FOLD_5
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/mimosa_gargantuan_23_5.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/mimosa_gargantuan_23_5.out
#SBATCH --gres=gpu:2
#SBATCH --account=phd_mimose
#SBATCH --constraint=gpu_RTXPro6000B_96G

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

mimose train --config mimosa_gargantuan_23.yaml --fold 5 --run-suffix fold5 --seed 0 --try-resume --num-workers 60 --distributed
mimose test --config mimosa_gargantuan_23.yaml --fold 5 --run-suffix fold5 --fp16 --num-workers 60

# Training completed (we reached the test), so cancel the remaining queued
# resume submissions of this same job from allsbatcher23.sh's singleton chain.
scancel --state=PENDING --name="$SLURM_JOB_NAME"
