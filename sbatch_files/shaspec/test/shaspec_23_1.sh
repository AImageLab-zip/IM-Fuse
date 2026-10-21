#!/bin/bash
#SBATCH --job-name=SHASPEC23_TEST_FOLD1
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/shaspec_23_fold1_test.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/shaspec_23_fold1_test.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

mimose test --config shaspec_23.yaml --fold 1 --run-suffix fold1
