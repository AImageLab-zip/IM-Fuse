#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --job-name="train18_MIFPN"
#SBATCH --time=24:00:00
#SBATCH --output=/work/grana_neuro/missing_modalities/MIFPN/slurm_out/train18_MIFPN_%j.out
#SBATCH --account=grana_neuro
#SBATCH --cpus-per-task=8
#SBATCH --dependency=afterany:13976

export WANDB_ENTITY="NeuroTumor" 
export WANDB_PROJECT="SegmentationMM"

source /work/grana_neuro/missing_modalities/mmformer_venv/bin/activate
cd /work/grana_neuro/missing_modalities/MIFPN

python /work/grana_neuro/missing_modalities/MIFPN/train.py \
    --datapath /work/grana_neuro/missing_modalities/BRATS2018_Training_none_npy \
    --num_epochs 1000 \
    --dataname BRATS2018 \
    --savepath /work/grana_neuro/missing_modalities/MIFPN/output_18/output2 \
    --batch_size 2 \
    --lr 2e-4 \
    --resume /work/grana_neuro/missing_modalities/MIFPN/output_18/output/model_last.pth
