#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --job-name="M2FTrans18"
#SBATCH --time=24:00:00
#SBATCH --output=/work/grana_neuro/missing_modalities/M2FTrans/M2FTrans_v1/slurm_out/train18_M2FTrans_%j.out
#SBATCH --account=grana_neuro
#SBATCH --cpus-per-task=8

export WANDB_ENTITY="NeuroTumor" 
export WANDB_PROJECT="SegmentationMM"

source /work/grana_neuro/missing_modalities/mmformer_venv/bin/activate
cd /work/grana_neuro/missing_modalities/M2FTrans/M2FTrans_v1

python /work/grana_neuro/missing_modalities/M2FTrans/M2FTrans_v1/train.py \
    --datapath /work/grana_neuro/missing_modalities/BRATS2018_Training_none_npy \
    --num_epochs 1000 \
    --dataname BRATS2018 \
    --savepath /work/grana_neuro/missing_modalities/M2FTrans/M2FTrans_v1/output18/output \
    --batch_size 4 \
    --region_fusion_start_epoch 0 \
    --lr 2e-4 \
    #--resume /work/grana_neuro/missing_modalities/M2FTrans/M2FTrans_v1/output3/model_last.pth