#!/bin/bash
#SBATCH --job-name=protokd_pretraining
#SBATCH --partition=all_usr_prod 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/err.txt
#SBATCH -o /homes/ocarpentiero/slurm/outerr/out.txt
#SBATCH --gres=gpu:1
#SBATCH --account=grana_neuro
#_SBATCH --constraint=gpu_L40S_48G|gpu_A40_48G|gpu_RTXA5000_24G|gpu_RTX6000_24G

source /homes/ocarpentiero/IM-Fuse/ProtoKD/protokd_venv/bin/activate
mask_values=(0 1)
for m1 in ${mask_values[@]}; do
    for m2 in ${mask_values[@]}; do
        for m3 in ${mask3[@]}; do
            for m4 in ${mask4[@]}; do
                if [ "$m1$m2$m3$m4" == "0000" ]; then
                    continue
                fi
                mask="$m1,$m2,$m3,$m4"
                python /homes/ocarpentiero/IM-Fuse/ProtoKD/code/train.py \
                  --data-path /work/grana_neuro/missing_modalities/BRATS2023_Training_protokd_npy \
                  --max-epoch 229 \
                  --checkpoint-path /work/grana_neuro/missing_modalities/protokd/train \
                  --teacher-checkpoint /work/grana_neuro/missing_modalities/protokd/pretrain/ch_229.pth.tar
                  --wandb-project-name ProtoKD \
                  --num-workers 8 \
                  --batch-size 16 \
                  --cache-dataset \# Optional: use it only if you want faster training
                  --modality $mask

