#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --job-name="eval_DC-Seg"
#SBATCH --time=12:00:00
#SBATCH --output=/work/grana_neuro/missing_modalities/DC-Seg/slurm_out/test_DC-Seg_%j.out
#SBATCH --account=grana_neuro
#SBATCH --cpus-per-task=8

source /work/grana_neuro/missing_modalities/mmformer_venv/bin/activate
cd /work/grana_neuro/missing_modalities/DC-Seg

python /work/grana_neuro/missing_modalities/DC-Seg/test.py 

