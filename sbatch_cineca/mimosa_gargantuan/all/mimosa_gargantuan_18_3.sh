#!/bin/bash
#SBATCH --job-name=MIMOSA_GARGANTUAN18_FOLD_3
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=400G
#SBATCH --time=4-00:00:00
#SBATCH -e /leonardo/home/userexternal/ocarpent/slurm/outerr/mimosa_gargantuan_18_3.err
#SBATCH -o /leonardo/home/userexternal/ocarpent/slurm/outerr/mimosa_gargantuan_18_3.out
#SBATCH --gres=gpu:2
#SBATCH --account=IscrB_MEDSYN
#SBATCH --constraint=gpu_RTXPro6000B_96G

cd /leonardo/home/userexternal/ocarpent/MiMoSe
source /leonardo/home/userexternal/ocarpent/MiMoSe/.venv/bin/activate

mimose train --config mimosa_gargantuan_18.yaml --fold 3 --run-suffix fold3 --seed 0 --try-resume --num-workers 60 --distributed
mimose test --config mimosa_gargantuan_18.yaml --fold 3 --run-suffix fold3 --fp16 --num-workers 60

# Training completed (we reached the test), so cancel the remaining queued
# resume submissions of this same job from allsbatcher18.sh's singleton chain.
scancel --state=PENDING --name="$SLURM_JOB_NAME"
