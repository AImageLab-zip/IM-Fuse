#!/bin/bash
#SBATCH --job-name=FLOPS_MIMOSA_TINY_CPU
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/flops_mimosa_tiny_cpu.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/flops_mimosa_tiny_cpu.out
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate

# Profiles FLOPs/MACs/peak memory for mimosa_tiny on CPU. FLOPs only depend on
# model architecture, not on which dataset a checkpoint was trained on, so
# a single _23.yaml config is enough; the report is written to this
# config's own <results_dir>/flops_cpu.txt (see `mimose flops --output-path`).
mimose flops --config "mimosa_tiny_23.yaml" --device cpu
