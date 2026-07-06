#!/bin/bash
#SBATCH --job-name=IM_FUSO18_SEED_42
#SBATCH --partition=boost_usr_prod 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/imfuse_18_42.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/imfuse_18_42.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose
#_SBATCH --constraint=gpu_L40S_48G|gpu_A40_48G|gpu_RTXA5000_24G|gpu_RTX6000_24G

mimose train --config imfuse_18.yaml --run-suffix seed42 --seed 42 --try-resume