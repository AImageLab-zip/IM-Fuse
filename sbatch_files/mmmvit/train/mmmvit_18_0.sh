#!/bin/bash
#SBATCH --job-name=MMMVIT18_SEED_0
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/mmmvit_18_0.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/mmmvit_18_0.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose
#SBATCH --constraint=gpu_RTXPro6000B_96G

mimose train --config mmmvit_18.yaml --run-suffix seed0 --seed 0 --try-resume
