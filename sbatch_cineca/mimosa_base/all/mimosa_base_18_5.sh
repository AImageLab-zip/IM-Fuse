#!/bin/bash
#SBATCH --job-name=MIMOSA_BASE18_FOLD_5
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=4-00:00:00
#SBATCH -e /leonardo/home/userexternal/ocarpent/slurm/outerr/mimosa_base_18_5.err
#SBATCH -o /leonardo/home/userexternal/ocarpent/slurm/outerr/mimosa_base_18_5.out
#SBATCH --gres=gpu:2
#SBATCH --account=IscrB_MEDSYN

cd /leonardo/home/userexternal/ocarpent/MiMoSe
source /leonardo/home/userexternal/ocarpent/MiMoSe/.venv/bin/activate

mimose train --config mimosa_base_18.yaml --fold 5 --run-suffix fold5 --seed 0 --try-resume --distributed
mimose test --config mimosa_base_18.yaml --fold 5 --run-suffix fold5 --fp16

# Training completed (we reached the test), so cancel the remaining queued
# resume submissions of this same job from allsbatcher18.sh's singleton chain.
scancel --state=PENDING --name="$SLURM_JOB_NAME"
