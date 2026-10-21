#!/bin/bash
#SBATCH --job-name=model_releasing_tests
#SBATCH --partition=all_usr_prod 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/small_debugging.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/small_debugging.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose
#_SBATCH --constraint=gpu_L40S_48G|gpu_A40_48G|gpu_RTXA5000_24G|gpu_RTX6000_24G

sleep infinity