#!/bin/bash
#SBATCH --job-name=FLOPS_MIMOSA_SIZE_GPU
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/flops_mimosa_size_gpu.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/flops_mimosa_size_gpu.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate

# Profiles FLOPs/MACs/peak memory on GPU for the mimosa_[size] family
# (mimosa_base, mimosa_gargantuan, mimosa_huge, mimosa_large, mimosa_medium,
# mimosa_micro, mimosa_small, mimosa_tiny). FLOPs only depend on model
# architecture, not on which dataset a checkpoint was trained on, so a single
# _23.yaml config per model is enough; each run writes its report to that
# config's own <results_dir>/flops.txt (see `mimose flops --output-path`).
MODELS=(
    mimosa_base
    mimosa_gargantuan
    mimosa_huge
    mimosa_large
    mimosa_medium
    mimosa_micro
    mimosa_small
    mimosa_tiny
)

for model in "${MODELS[@]}"; do
    echo "=== ${model} ==="
    mimose flops --config "${model}_23.yaml" --device cuda 
done
