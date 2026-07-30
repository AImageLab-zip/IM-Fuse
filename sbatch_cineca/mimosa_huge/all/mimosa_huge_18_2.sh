#!/bin/bash
#SBATCH --job-name=MIMOSA_HUGE18_FOLD_2
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=400G
#SBATCH --time=4-00:00:00
#SBATCH -e /leonardo/home/userexternal/ocarpent/slurm/outerr/mimosa_huge_18_2.err
#SBATCH -o /leonardo/home/userexternal/ocarpent/slurm/outerr/mimosa_huge_18_2.out
#SBATCH --gres=gpu:2
#SBATCH --account=IscrB_MEDSYN
#SBATCH --constraint=gpu_RTXPro6000B_96G

cd /leonardo/home/userexternal/ocarpent/MiMoSe
source /leonardo/home/userexternal/ocarpent/MiMoSe/.venv/bin/activate

mimose train --config mimosa_huge_18.yaml --fold 2 --run-suffix fold2 --seed 0 --try-resume --num-workers 60 --distributed
mimose test --config mimosa_huge_18.yaml --fold 2 --run-suffix fold2 --fp16 --num-workers 60

# Training completed (we reached the test), so cancel the remaining queued
# resume submissions of this same job from allsbatcher18.sh's singleton chain.
scancel --state=PENDING --name="$SLURM_JOB_NAME"
