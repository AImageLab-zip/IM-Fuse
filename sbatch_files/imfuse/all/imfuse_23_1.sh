#!/bin/bash
#SBATCH --job-name=IM_FUSO23_FOLD_1
#SBATCH --partition=boost_usr_prod 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/imfuse_23_1.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/imfuse_23_1.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose
#_SBATCH --constraint=gpu_L40S_48G|gpu_A40_48G|gpu_RTXA5000_24G|gpu_RTX6000_24G

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate

mimose train --config imfuse_23.yaml --fold 1 --run-suffix fold1 --seed 0 --try-resume
mimose test --config imfuse_23.yaml --fold 1 --run-suffix fold1 

# Training completed (we reached the test), so cancel the remaining queued
# resume submissions of this same job from allsbatcher23.sh's singleton chain.
scancel --state=PENDING --name="$SLURM_JOB_NAME"