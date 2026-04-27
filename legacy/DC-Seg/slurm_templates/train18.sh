#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --job-name="train18_DC-Seg"
#SBATCH --time=24:00:00
#SBATCH --output=/work/grana_neuro/missing_modalities/DC-Seg/slurm_out/train18_DC-Seg_%j.out
#SBATCH --account=grana_neuro
#SBATCH --cpus-per-task=8
#SBATCH --dependency=afterany:14055

export WANDB_ENTITY="NeuroTumor" 
export WANDB_PROJECT="SegmentationMM"

source /work/grana_neuro/missing_modalities/mmformer_venv/bin/activate
cd /work/grana_neuro/missing_modalities/DC-Seg

python /work/grana_neuro/missing_modalities/DC-Seg/train.py \
    --batch_size 1 \
    --dataname BRATS2018 \
    --iter_per_epoch -1 \
    --num_epochs 500 \
    --fusion_type RFM \
    --use_reg_loss \
    --use_recon_loss \
    --use_ana_contrastive \
    --use_mod_contrastive \
    --crop_size 112 \
    --savepath /work/grana_neuro/missing_modalities/DC-Seg/output18 \
    --datapath /work/grana_neuro/missing_modalities/BRATS2018_Training_none_npy \
    --resume /work/grana_neuro/missing_modalities/DC-Seg/output18/model_last.pth