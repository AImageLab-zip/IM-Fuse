#!/bin/bash
#SBATCH --job-name=LCKD18_TEST_FOLD1
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/lckd_18_fold1_test.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/lckd_18_fold1_test.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate

mimose test --config lckd_18.yaml --fold 1 --run-suffix fold1
