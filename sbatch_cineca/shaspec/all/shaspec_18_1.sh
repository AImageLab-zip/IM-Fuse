#!/bin/bash
#SBATCH --job-name=SHASPC18_FOLD_1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=4-00:00:00
#SBATCH -e /leonardo/home/userexternal/ocarpent/slurm/outerr/shaspec_18_1.err
#SBATCH -o /leonardo/home/userexternal/ocarpent/slurm/outerr/shaspec_18_1.out
#SBATCH --gres=gpu:1
#SBATCH --account=IscrB_MEDSYN

cd /leonardo/home/userexternal/ocarpent/MiMoSe
source /leonardo/home/userexternal/ocarpent/MiMoSe/.venv/bin/activate

mimose train --config shaspec_18.yaml --fold 1 --run-suffix fold1 --seed 0 --try-resume
mimose test --config shaspec_18.yaml --fold 1 --run-suffix fold1

# Training completed (we reached the test), so cancel the remaining queued
# resume submissions of this same job from allsbatcher18.sh's singleton chain.
scancel --state=PENDING --name="$SLURM_JOB_NAME"
