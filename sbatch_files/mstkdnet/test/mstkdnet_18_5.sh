#!/bin/bash
#SBATCH --job-name=MSTKDNET18_TEST_FOLD5
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/mstkdnet_18_fold5_test.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/mstkdnet_18_fold5_test.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

mimose test --config mstkdnet_18.yaml --fold 5 --run-suffix fold5
