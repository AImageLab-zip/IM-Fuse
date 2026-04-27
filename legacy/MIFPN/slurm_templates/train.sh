#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --job-name="train_MIFPN"
#SBATCH --time=24:00:00
#SBATCH --output=/work/grana_neuro/missing_modalities/MIFPN/slurm_out/train_MIFPN_%j.out
#SBATCH --account=grana_neuro
#SBATCH --cpus-per-task=8
#SBATCH --dependency=afterany:2735502

export WANDB_ENTITY="NeuroTumor" 
export WANDB_PROJECT="SegmentationMM"

source /work/grana_neuro/missing_modalities/mmformer_venv/bin/activate
cd /work/grana_neuro/missing_modalities/MIFPN

python /work/grana_neuro/missing_modalities/MIFPN/train.py \
    --datapath /work/grana_neuro/missing_modalities/BRATS2023_Training_npy \
    --num_epochs 1000 \
    --dataname BRATS2023 \
    --savepath /work/grana_neuro/missing_modalities/MIFPN/output5 \
    --batch_size 2 \
    --lr 2e-4 \
    --resume /work/grana_neuro/missing_modalities/MIFPN/output4/model_last.pth 
