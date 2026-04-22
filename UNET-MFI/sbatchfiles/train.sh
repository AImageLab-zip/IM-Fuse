#!/bin/bash
#SBATCH --job-name=UNET_MFI_TRAINING_LETS_GO
#SBATCH --partition=boost_usr_prod 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/err.txt
#SBATCH -o /homes/ocarpentiero/slurm/outerr/out.txt
#SBATCH --gres=gpu:1
#SBATCH --account=grana_neuro
#_SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G|gpu_RTX6000_24G|gpu_RTX_A5000_24G

source /homes/ocarpentiero/IM-Fuse/UNET-MFI/unetmfi_venv/bin/activate

python /homes/ocarpentiero/IM-Fuse/UNET-MFI/train.py \
  --datapath /work/grana_neuro/missing_modalities/UNET-MFI/dataset \
  --num-epochs 1200 \
  --checkpoint-path /work/grana_neuro/missing_modalities/UNET-MFI/checkpoints \
  --wandb-project-name UNET-MFI \
  --num-workers 16 \
  --batch-size 2