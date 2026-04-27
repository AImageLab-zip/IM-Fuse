#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --job-name="SRMNet18"
#SBATCH --time=24:00:00
#SBATCH --output=/work/grana_neuro/missing_modalities/SRMNet/slurm_out18/train18_SRMNet_%j.out
#SBATCH --account=grana_neuro
#SBATCH --cpus-per-task=8


export WANDB_ENTITY="NeuroTumor" 
export WANDB_PROJECT="SegmentationMM"

source /work/grana_neuro/missing_modalities/mmformer_venv/bin/activate
cd /work/grana_neuro/missing_modalities/SRMNet

python /work/grana_neuro/missing_modalities/SRMNet/train.py \
    --batch_size 1 \
    --dataname BRATS2018 \
    --iter_per_epoch -1 \
    --num_epochs 1000 \
    --savepath /work/grana_neuro/missing_modalities/SRMNet/output18/output_2 \
    --load True \
    --datapath /work/grana_neuro/missing_modalities/BRATS2018_Training_none_npy \
    --resume /work/grana_neuro/missing_modalities/SRMNet/output18/output/model_last.pth 