#!/bin/bash
#SBATCH --job-name=FLOPS_MIFPN
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/flops_mifpn.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/flops_mifpn.out
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate

# Profiles FLOPs/MACs/peak memory for mifpn on CPU. FLOPs only depend on
# model architecture, not on which dataset a checkpoint was trained on, so
# a single _23.yaml config is enough; the report is written to this
# config's own <results_dir>/flops_cpu.txt (see `mimose flops --output-path`).
mimose flops --config "mifpn_23.yaml" --device cpu
