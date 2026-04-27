#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --job-name="eval_M2FTrans"
#SBATCH --array=1
#SBATCH --time=6:00:00
#SBATCH --output=/work/grana_neuro/missing_modalities/M2FTrans/M2FTrans_v1/slurm_out/test_M2FTrans_%j.out
#SBATCH --account=grana_neuro
#SBATCH --cpus-per-task=8

source /work/grana_neuro/missing_modalities/mmformer_venv/bin/activate
cd /work/grana_neuro/missing_modalities/M2FTrans/M2FTrans_v1

python /work/grana_neuro/missing_modalities/M2FTrans/M2FTrans_v1/test.py

