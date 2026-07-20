#!/bin/bash
#SBATCH --job-name=Missing_Modalities_Segmentation=fiorellini
#SBATCH --partition=boost_usr_prod 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=6:00:00
#SBATCH -e /leonardo/home/userexternal/ocarpent/slurm/outerr/debug_heavy_train.err
#SBATCH -o /leonardo/home/userexternal/ocarpent/slurm/outerr/debug_heavy_train.out
#SBATCH --gres=gpu:2
#SBATCH --account=grana_neuro
#_SBATCH --constraint=gpu_L40S_48G|gpu_A40_48G|gpu_RTXA5000_24G|gpu_RTX6000_24G

sleep infinity