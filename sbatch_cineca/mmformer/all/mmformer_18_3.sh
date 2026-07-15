#!/bin/bash
#SBATCH --job-name=MMFMR18_FOLD_3
#SBATCH --partition=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=4-00:00:00
#SBATCH -e /leonardo/home/userexternal/ocarpent/slurm/outerr/mmformer_18_3.err
#SBATCH -o /leonardo/home/userexternal/ocarpent/slurm/outerr/mmformer_18_3.out
#SBATCH --gres=gpu:1
#SBATCH --account=IscrB_MEDSYN

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate

mimose train --config mmformer_18.yaml --fold 3 --run-suffix fold3 --seed 0 --try-resume
mimose test --config mmformer_18.yaml --fold 3 --run-suffix fold3 --fp16

# Training completed (we reached the test), so cancel the remaining queued
# resume submissions of this same job from allsbatcher18.sh's singleton chain.
scancel --state=PENDING --name="$SLURM_JOB_NAME"
