#!/bin/bash
#SBATCH --job-name=INOUTFUSION23_SEED_67
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/inoutfusion_23_67.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/inoutfusion_23_67.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

mimose train --config inoutfusion_23.yaml --run-suffix seed67 --seed 67 --try-resume
