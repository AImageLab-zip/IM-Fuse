#!/bin/bash
#SBATCH --job-name=MIMOSA_LARGE23_FOLD_4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=4-00:00:00
#SBATCH -e /leonardo/home/userexternal/ocarpent/slurm/outerr/mimosa_large_23_4.err
#SBATCH -o /leonardo/home/userexternal/ocarpent/slurm/outerr/mimosa_large_23_4.out
#SBATCH --gres=gpu:2
#SBATCH --account=IscrB_MEDSYN

cd /leonardo/home/userexternal/ocarpent/MiMoSe
source /leonardo/home/userexternal/ocarpent/MiMoSe/.venv/bin/activate

mimose train --config mimosa_large_23.yaml --fold 4 --run-suffix fold4 --seed 0 --try-resume --distributed
mimose test --config mimosa_large_23.yaml --fold 4 --run-suffix fold4 --fp16

# Training completed (we reached the test), so cancel the remaining queued
# resume submissions of this same job from allsbatcher23.sh's singleton chain.
scancel --state=PENDING --name="$SLURM_JOB_NAME"
