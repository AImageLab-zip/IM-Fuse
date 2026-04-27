#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --job-name="train_SRMNet"
#SBATCH --time=24:00:00
#SBATCH --output=/work/grana_neuro/missing_modalities/SRMNet/slurm_out/train_SRMNet_%j.out
#SBATCH --account=grana_neuro
#SBATCH --cpus-per-task=8
#SBATCH --dependency=afterany:11745

export WANDB_ENTITY="NeuroTumor" 
export WANDB_PROJECT="SegmentationMM"

source /work/grana_neuro/missing_modalities/mmformer_venv/bin/activate
cd /work/grana_neuro/missing_modalities/SRMNet

python /work/grana_neuro/missing_modalities/SRMNet/train.py \
    --batch_size 1 \
    --dataname BRATS2023 \
    --iter_per_epoch -1 \
    --num_epochs 1000 \
    --savepath /work/grana_neuro/missing_modalities/SRMNet/output_8 \
    --load True \
    --datapath /work/grana_neuro/missing_modalities/BRATS2023_Training_npy \
    --resume /work/grana_neuro/missing_modalities/SRMNet/output_7/model_last.pth 