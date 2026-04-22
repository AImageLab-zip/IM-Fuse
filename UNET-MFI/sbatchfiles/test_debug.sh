#!/bin/bash
#SBATCH --job-name=UNET_MFI_testing_yes
#SBATCH --partition=all_usr_prod 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=3:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/err.txt
#SBATCH -o /homes/ocarpentiero/slurm/outerr/out.txt
#SBATCH --gres=gpu:1
#SBATCH --account=grana_neuro
#_SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G|gpu_RTX6000_24G|gpu_RTX_A5000_24G
sleep infinity