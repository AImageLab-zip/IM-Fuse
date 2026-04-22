#!/bin/bash
#SBATCH --job-name=LETS_PREPROCESS_UNET_MFI_LETS_GO
#SBATCH --partition=all_usr_prod 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=3:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/err.txt
#SBATCH -o /homes/ocarpentiero/slurm/outerr/out.txt
#SBATCH --gres=gpu:0
#SBATCH --account=grana_neuro
#_SBATCH --constraint=gpu_L40S_48G|gpu_A40_48G|gpu_RTXA5000_24G|gpu_RTX6000_24G

source /homes/ocarpentiero/IM-Fuse/UNET-MFI/unetmfi_venv/bin/activate
python /homes/ocarpentiero/IM-Fuse/UNET-MFI/preprocess.py \
 --input-path /work/grana_neuro/MICCAI_BraTS_2018_Data_Training_v2/ --output-path /work/grana_neuro/missing_modalities/UNET-MFI/dataset_18 \
 --num-workers 16 
