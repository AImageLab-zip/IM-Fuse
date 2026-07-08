#!/bin/bash
#SBATCH --job-name=IMS2TRANS23_SEED_69
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/ims2trans_23_69.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/ims2trans_23_69.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose
#SBATCH --constraint=gpu_RTXPro6000B_96G

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate

mimose train --config ims2trans_23.yaml --run-suffix seed69 --seed 69 --try-resume
