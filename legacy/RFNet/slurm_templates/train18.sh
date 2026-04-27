#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --job-name="RFNet18"
#SBATCH --time=24:00:00
#SBATCH --output=/work/grana_neuro/missing_modalities/RFNet/slurm_out/train18_RFNet_%j.out
#SBATCH --account=grana_neuro
#SBATCH --cpus-per-task=8
#SBATCH --dependency=afterany:2670824

export WANDB_ENTITY="NeuroTumor" 
export WANDB_PROJECT="SegmentationMM"

source /work/grana_neuro/missing_modalities/mmformer_venv/bin/activate
cd /work/grana_neuro/missing_modalities/RFNet

python /work/grana_neuro/missing_modalities/RFNet/train.py \
    --datapath /work/grana_neuro/missing_modalities/BRATS2018_Training_none_npy \
    --num_epochs 300 \
    --dataname BRATS2018 \
    --savepath /work/grana_neuro/missing_modalities/RFNet/output18 \
    --batch_size 2 \
    --region_fusion_start_epoch 20 \
    --lr 2e-4 \
    #--resume /work/grana_neuro/missing_modalities/RFNet/output/model_last.pth
