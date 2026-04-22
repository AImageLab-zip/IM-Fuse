#!/bin/bash
#SBATCH --job-name=UNET_MFI_Test_LETS_GO
#SBATCH --partition=all_usr_prod 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/errunetmfitestslow.txt
#SBATCH -o /homes/ocarpentiero/slurm/outerr/out.txt
#SBATCH --gres=gpu:1
#SBATCH --account=grana_neuro
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G|gpu_RTX6000_24G|gpu_RTX_A5000_24G

source /homes/ocarpentiero/IM-Fuse/UNET-MFI/unetmfi_venv/bin/activate

python /homes/ocarpentiero/IM-Fuse/legacy/UNET-MFI/test.py \
  --datapath /work/grana_neuro/missing_modalities/UNET-MFI/dataset_18 \
  --resume /work/grana_neuro/missing_modalities/UNET-MFI/checkpoints_18/chk_1199.pth \
  --savepath /homes/ocarpentiero/results/test_unet_mfi_18_slow.txt \
  --version brats18
