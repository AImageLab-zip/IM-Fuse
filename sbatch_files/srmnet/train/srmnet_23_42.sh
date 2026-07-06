#!/bin/bash
#SBATCH --job-name=SRMNET23_SEED_42
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/srmnet_23_42.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/srmnet_23_42.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose
#SBATCH --constraint=gpu_RTXPro6000B_96G

mimose train --config srmnet_23.yaml --run-suffix seed42 --seed 42 --try-resume
