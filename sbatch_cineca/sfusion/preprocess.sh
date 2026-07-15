#!/bin/bash
#SBATCH --job-name=SONO_PREPROCESSATO
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=1:00:00
#SBATCH -e /leonardo/home/userexternal/ocarpent/slurm/outerr/preprocess.err
#SBATCH -o /leonardo/home/userexternal/ocarpent/slurm/outerr/preprocess.out
#SBATCH --account=IscrB_MEDSYN

mimose preprocess --config sfusion_18.yaml --dataset-type brats23 --yes
