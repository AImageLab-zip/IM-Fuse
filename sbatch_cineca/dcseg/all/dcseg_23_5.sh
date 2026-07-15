#!/bin/bash
#SBATCH --job-name=DCSEG23_FOLD_5
#SBATCH --partition=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=4-00:00:00
#SBATCH -e /leonardo/home/userexternal/ocarpent/slurm/outerr/dcseg_23_5.err
#SBATCH -o /leonardo/home/userexternal/ocarpent/slurm/outerr/dcseg_23_5.out
#SBATCH --gres=gpu:1
#SBATCH --account=IscrB_MEDSYN

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate

mimose train --config dcseg_23.yaml --fold 5 --run-suffix fold5 --seed 0 --try-resume
mimose test --config dcseg_23.yaml --fold 5 --run-suffix fold5 --fp16

# Training completed (we reached the test), so cancel the remaining queued
# resume submissions of this same job from allsbatcher23.sh's singleton chain.
scancel --state=PENDING --name="$SLURM_JOB_NAME"
