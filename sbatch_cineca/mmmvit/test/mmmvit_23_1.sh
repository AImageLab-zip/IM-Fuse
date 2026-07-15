#!/bin/bash
#SBATCH --job-name=MMMVIT23_TEST_FOLD1
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -e /leonardo/home/userexternal/ocarpent/slurm/outerr/mmmvit_23_fold1_test.err
#SBATCH -o /leonardo/home/userexternal/ocarpent/slurm/outerr/mmmvit_23_fold1_test.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose

cd /leonardo/home/userexternal/ocarpent/MiMoSe
source /leonardo/home/userexternal/ocarpent/MiMoSe/.venv/bin/activate

mimose test --config mmmvit_23.yaml --fold 1 --run-suffix fold1
