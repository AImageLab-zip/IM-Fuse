#!/bin/bash
#SBATCH --job-name=medgemma_time_lets_go_report_generation_more_like_report_godi-ation
#SBATCH --partition=boost_usr_prod 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=6:00:00
#SBATCH -e /leonardo/home/userexternal/ocarpent/slurm/outerr/imfuse_18.err
#SBATCH -o /leonardo/home/userexternal/ocarpent/slurm/outerr/imfuse_18.out
#SBATCH --gres=gpu:1
#SBATCH --account=grana_neuro
#_SBATCH --constraint=gpu_L40S_48G|gpu_A40_48G|gpu_RTXA5000_24G|gpu_RTX6000_24G

sleep infinity