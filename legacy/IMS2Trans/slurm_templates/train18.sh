#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --job-name="train_IMS2Trans"
#SBATCH --time=24:00:00
#SBATCH --output=/work/grana_neuro/missing_modalities/IMS2Trans/slurm_out/train_IMS2Trans_%j.out
#SBATCH --account=grana_neuro
#SBATCH --cpus-per-task=8
#SBATCH --dependency=afterany:17265

export WANDB_ENTITY="NeuroTumor" 
export WANDB_PROJECT="SegmentationMM"

source /work/grana_neuro/missing_modalities/mmformer_venv/bin/activate
cd /work/grana_neuro/missing_modalities/IMS2Trans

python /work/grana_neuro/missing_modalities/IMS2Trans/train.py \
    --datapath /work/grana_neuro/missing_modalities/BRATS2018_Training_none_npy \
    --num_epochs 1000 \
    --dataname BRATS2018 \
    --savepath /work/grana_neuro/missing_modalities/IMS2Trans/output18/output_8 \
    --batch_size 1 \
    --use_checkpoint \
    --resume /work/grana_neuro/missing_modalities/IMS2Trans/output18/output_7/model_last.pth