#!/bin/bash
#SBATCH --job-name=MEGATEST_UNETMFI23_FOLD_3
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/unetmfi_23_1.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/unetmfi_23_1.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

mimose test-fast --config unetmfi_23.yaml --fold 3 --run-suffix fold3 --fp16 --batch-size 1

