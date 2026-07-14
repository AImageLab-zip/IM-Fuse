#!/bin/bash
#SBATCH --job-name=SHASPEC23_TEST_FOLD5
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/shaspec_23_fold5_test.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/shaspec_23_fold5_test.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate

mimose test --config shaspec_23.yaml --fold 5 --run-suffix fold5
