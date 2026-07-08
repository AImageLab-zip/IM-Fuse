#!/bin/bash
#SBATCH --job-name=UHVED23_TEST
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=150G
#SBATCH --time=6:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/uhved_23_test.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/uhved_23_test.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose
#SBATCH --constraint=gpu_RTXPro6000B_96G

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate

mimose test --config uhved_23.yaml --run-suffix seed0 --fp16 > /homes/ocarpentiero/slurm/outerr/uhved_23_test_seed0.log 2>&1 &
mimose test --config uhved_23.yaml --run-suffix seed42 --fp16 > /homes/ocarpentiero/slurm/outerr/uhved_23_test_seed42.log 2>&1 &
mimose test --config uhved_23.yaml --run-suffix seed69 --fp16 > /homes/ocarpentiero/slurm/outerr/uhved_23_test_seed69.log 2>&1 &

wait
