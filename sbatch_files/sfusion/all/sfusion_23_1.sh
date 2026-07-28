#!/bin/bash
#SBATCH --job-name=SFUS23_FOLD_1
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/sfusion_23_1.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/sfusion_23_1.out
#SBATCH --gres=gpu:2
#SBATCH --account=phd_mimose
#SBATCH --constraint=gpu_RTXPro6000B_96G

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

mimose train --config sfusion_23.yaml --fold 1 --run-suffix fold1 --seed 0 --try-resume --distributed
mimose test --config sfusion_23.yaml --fold 1 --run-suffix fold1

SPLIT_FILE=/homes/ocarpentiero/MiMoSe/src/mimose/data/splits/internal.json
DATA_DIR=/work/phd_mimose/internal_raw/sfusion_internal

for trained_fold in 1; do
    mimose test \
        --config sfusion_23.yaml \
        --run-suffix "fold${trained_fold}" \
        --dataset-type internal \
        --split-file "$SPLIT_FILE" \
        --fold 1 \
        --data-dir "$DATA_DIR" \
        --output-path "/work/phd_mimose/results/sfusion_23/results_internal.txt"
done

# Training completed (we reached the test), so cancel the remaining queued
# resume submissions of this same job from allsbatcher23.sh's singleton chain.
scancel --state=PENDING --name="$SLURM_JOB_NAME"
