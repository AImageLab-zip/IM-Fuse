#!/bin/bash
#SBATCH --job-name=MIMOSA_HUGE18_SEED_67
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH -e /leonardo/home/userexternal/ocarpent/slurm/outerr/mimosa_huge_18_67.err
#SBATCH -o /leonardo/home/userexternal/ocarpent/slurm/outerr/mimosa_huge_18_67.out
#SBATCH --gres=gpu:2
#SBATCH --account=phd_mimose
#SBATCH --constraint=gpu_RTXPro6000B_96G

cd /leonardo/home/userexternal/ocarpent/MiMoSe
source /leonardo/home/userexternal/ocarpent/MiMoSe/.venv/bin/activate

mimose train --config mimosa_huge_18.yaml --run-suffix seed67 --seed 67 --try-resume --num-workers 60 --distributed
