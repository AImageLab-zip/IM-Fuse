#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --job-name="eval_MIFPN"
#SBATCH --time=12:00:00
#SBATCH --output=/work/grana_neuro/missing_modalities/MIFPN/slurm_out/test_MIFPN_%j.out
#SBATCH --account=grana_neuro
#SBATCH --cpus-per-task=8

source /work/grana_neuro/missing_modalities/mmformer_venv/bin/activate
cd /work/grana_neuro/missing_modalities/MIFPN

python /work/grana_neuro/missing_modalities/MIFPN/test.py 

