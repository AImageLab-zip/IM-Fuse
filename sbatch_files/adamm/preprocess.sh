#!/bin/bash
#SBATCH --job-name=ADAMM_PREPROCESS
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=2:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/adamm_preprocess.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/adamm_preprocess.out
#SBATCH --gres=gpu:0
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate

# Same preprocessing as MST-KDNet (non_empty crop, min 128^3, subject z-score) into adamm-preprocessed.
# One BraTS23 pass covers both datasets: BraTS18 runs select their subjects through the split files.
mimose preprocess --config adamm_23.yaml --dataset-type brats23 --yes
