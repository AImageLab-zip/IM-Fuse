#!/bin/bash
#SBATCH --job-name=TINYMIMOSAWEIGHTED18_FOLD_1
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/tinymimosaweighted_18_1.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/tinymimosaweighted_18_1.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate

mimose train --config tinymimosaweighted_18.yaml --fold 1 --run-suffix fold1 --seed 0 --try-resume
mimose test --config tinymimosaweighted_18.yaml --fold 1 --run-suffix fold1 --fp16

# Training completed (we reached the test), so cancel the remaining queued
# resume submissions of this same job from allsbatcher18.sh's singleton chain.
scancel --state=PENDING --name="$SLURM_JOB_NAME"
