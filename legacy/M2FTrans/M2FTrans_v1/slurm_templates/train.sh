#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --job-name="train_M2FTrans"
#SBATCH --time=24:00:00
#SBATCH --output=/work/grana_neuro/missing_modalities/M2FTrans/M2FTrans_v1/slurm_out/train_M2FTrans_%j.out
#SBATCH --account=grana_neuro
#SBATCH --cpus-per-task=8
#SBATCH --dependency=afterany:2675099

export WANDB_ENTITY="NeuroTumor" 
export WANDB_PROJECT="SegmentationMM"

source /work/grana_neuro/missing_modalities/mmformer_venv/bin/activate
cd /work/grana_neuro/missing_modalities/M2FTrans/M2FTrans_v1

python /work/grana_neuro/missing_modalities/M2FTrans/M2FTrans_v1/train.py \
    --datapath /work/grana_neuro/missing_modalities/BRATS2023_Training_npy \
    --num_epochs 1000 \
    --dataname BRATS2023 \
    --savepath /work/grana_neuro/missing_modalities/M2FTrans/M2FTrans_v1/output4 \
    --batch_size 4 \
    --region_fusion_start_epoch 0 \
    --lr 2e-4 \
    --resume /work/grana_neuro/missing_modalities/M2FTrans/M2FTrans_v1/output3/model_last.pth