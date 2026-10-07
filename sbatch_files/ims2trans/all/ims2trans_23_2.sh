#!/bin/bash
#SBATCH --job-name=IMS2TRANS23_FOLD_2
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/ims2trans_23_2.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/ims2trans_23_2.out
#SBATCH --gres=gpu:4
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

mimose train --config ims2trans_23.yaml --fold 2 --run-suffix fold2 --seed 0 --try-resume --distributed
mimose test --config ims2trans_23.yaml --fold 2 --run-suffix fold2 --precision fp16

# Training completed (we reached the test), so cancel the remaining queued
# resume submissions of this same job from allsbatcher23.sh's singleton chain.
scancel --state=PENDING --name="$SLURM_JOB_NAME"
