#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --job-name="eval_SRMNet"
#SBATCH --time=12:00:00
#SBATCH --output=/work/grana_neuro/missing_modalities/SRMNet/slurm_out/test_SRMNet_%j.out
#SBATCH --account=grana_neuro
#SBATCH --cpus-per-task=8

source /work/grana_neuro/missing_modalities/mmformer_venv/bin/activate
cd /work/grana_neuro/missing_modalities/SRMNet

python /work/grana_neuro/missing_modalities/SRMNet/test.py \
    --dataname BRATS2023 \
    --datapath /work/grana_neuro/missing_modalities/BRATS2023_Training_npy \
    --resume /work/grana_neuro/missing_modalities/SRMNet/output23/output_8/best.pth