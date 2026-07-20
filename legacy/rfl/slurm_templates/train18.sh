#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --job-name="rfl18"
#SBATCH --time=24:00:00
#SBATCH --output=/work/grana_neuro/missing_modalities/rfl/slurm_out/train_rfl_%j.out
#SBATCH --account=grana_neuro
#SBATCH --cpus-per-task=8
#SBATCH --dependency=afterany:493

export WANDB_ENTITY="NeuroTumor" 
export WANDB_PROJECT="SegmentationMM"

source /work/grana_neuro/missing_modalities/mmformer_venv/bin/activate
cd /work/grana_neuro/missing_modalities/rfl

python /work/grana_neuro/missing_modalities/rfl/train.py \
    --datapath /work/grana_neuro/missing_modalities/BRATS2018_Training_none_npy \
    --num_epochs 1000 \
    --dataname BRATS2018 \
    --savepath /work/grana_neuro/missing_modalities/rfl/output18/output_2 \
    --batch_size 1 \
    --lr 2e-4 \
    --resume /work/grana_neuro/missing_modalities/rfl/output18/output/model_last.pth
    
