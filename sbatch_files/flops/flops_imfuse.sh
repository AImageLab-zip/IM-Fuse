#!/bin/bash
#SBATCH --job-name=FLOPS_IMFUSE
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/flops_imfuse.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/flops_imfuse.out
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate

# Profiles FLOPs/MACs/peak memory for imfuse on CPU. FLOPs only depend on
# model architecture, not on which dataset a checkpoint was trained on, so
# a single _23.yaml config is enough; the report is written to this
# config's own <results_dir>/flops_cpu.txt (see `mimose flops --output-path`).
#
# IMFuse's Mamba layers reach for CUDA-only kernels (causal_conv1d,
# selective_scan_cuda) by default, which crash on CPU tensors. The wrapper
# below patches mamba_ssm (at runtime, not in the repo) to use its
# pure-PyTorch reference path instead -- see scripts/run_flops_cpu_mamba_patch.py.
python scripts/run_flops_cpu_mamba_patch.py flops --config "imfuse_23.yaml" --device cpu
