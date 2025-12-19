#!/bin/bash
#SBATCH --job-name=protokd_preprocessing
#SBATCH --partition=all_usr_prod 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=2:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/err.txt
#SBATCH -o /homes/ocarpentiero/slurm/outerr/out.txt
#SBATCH --gres=gpu:0
#SBATCH --account=grana_neuro
#_SBATCH --constraint=gpu_L40S_48G|gpu_A40_48G|gpu_RTXA5000_24G|gpu_RTX6000_24G

source /homes/ocarpentiero/IM-Fuse/ProtoKD/protokd_venv/bin/activate
python /homes/ocarpentiero/IM-Fuse/ProtoKD/code/preprocess.py \
--input-path '/work/grana_neuro/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData' \
--output-path '/work/grana_neuro/missing_modalities/BRATS2023_Training_protokd_npy' \
--num-workers 8 \
--compressed