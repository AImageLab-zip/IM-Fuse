#!/bin/bash
#SBATCH --job-name=PARETO_PREPROCESS
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=0:20:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/pareto_preprocess.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/pareto_preprocess.out
#SBATCH --gres=gpu:0
#SBATCH --account=phd_mimose

# The pareto configs share mimosa_preprocessed with the whole mimosa_[size]
# sweep, so this only needs running if that folder does not already exist.
mimose preprocess --config pareto_18.yaml --yes --dataset-type brats18
