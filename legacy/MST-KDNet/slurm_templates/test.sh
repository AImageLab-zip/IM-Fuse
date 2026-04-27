#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --job-name="eval_MST-KDNet"
#SBATCH --time=24:00:00
#SBATCH --output=/work/grana_neuro/missing_modalities/MST-KDNet/slurm_out/eval_MST-KDNet_%j.out
#SBATCH --account=grana_neuro
#SBATCH --cpus-per-task=8

export WANDB_ENTITY="NeuroTumor" 
export WANDB_PROJECT="SegmentationMM"

source /work/grana_neuro/missing_modalities/mmformer_venv/bin/activate
cd /work/grana_neuro/missing_modalities/MST-KDNet

python /work/grana_neuro/missing_modalities/MST-KDNet/eval.py