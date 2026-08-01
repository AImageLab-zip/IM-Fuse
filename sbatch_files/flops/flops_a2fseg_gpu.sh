#!/bin/bash
#SBATCH --job-name=FLOPS_A2FSEG_GPU
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/flops_a2fseg_gpu.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/flops_a2fseg_gpu.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate

# GPU counterpart of flops_a2fseg.sh. Profiles FLOPs/MACs/peak memory, plus
# GPU-only kernel-launch count and peak reserved memory (see
# measure_kernel_launches_and_memory in mimose/flops.py), for a2fseg. FLOPs
# only depend on model architecture, not on which dataset a checkpoint was
# trained on, so a single _23.yaml config is enough; the report is written
# to this config's own <results_dir>/flops_gpu.txt (see
# `mimose flops --output-path`).
mimose flops --config "a2fseg_23.yaml" --device cuda
