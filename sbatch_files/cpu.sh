#!/bin/bash
#SBATCH --job-name=Missing_Modalities_Segmentation=fiorellini
#SBATCH --partition=boost_usr_prod 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=6:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/cpu.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/cpu.out
#SBATCH --gres=gpu:0
#SBATCH --account=grana_neuro
#_SBATCH --constraint=gpu_L40S_48G|gpu_A40_48G|gpu_RTXA5000_24G|gpu_RTX6000_24G

sleep infinity