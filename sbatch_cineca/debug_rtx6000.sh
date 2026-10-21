#!/bin/bash
#SBATCH --job-name=Missing_Modalities_Segmentation
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH -e /leonardo/home/userexternal/ocarpent/slurm/outerr/debug_rtx6000.err
#SBATCH -o /leonardo/home/userexternal/ocarpent/slurm/outerr/debug_rtx6000.out
#SBATCH --gres=gpu:1
#SBATCH --account=grana_neuro
#SBATCH --constraint=gpu_RTXPro6000B_96G

sleep infinity
