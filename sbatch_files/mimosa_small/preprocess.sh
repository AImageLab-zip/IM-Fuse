#!/bin/bash
#SBATCH --job-name=SONO_PREPROCESSATO
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=0:20:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/preprocess.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/preprocess.out
#SBATCH --gres=gpu:0
#SBATCH --account=phd_mimose
#_SBATCH --constraint=gpu_L40S_48G|gpu_A40_48G|gpu_RTXA5000_24G|gpu_RTX6000_24G

mimose preprocess --config mimosa_small_18.yaml --yes --dataset-type brats23
