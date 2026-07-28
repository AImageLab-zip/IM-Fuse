#!/bin/bash
#SBATCH --job-name=FLOPS_18MODELS
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/flops_18models.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/flops_18models.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate

# Profiles FLOPs/MACs/peak memory for the 18 models in the paper's Table 1
# (tab:models), plus tinymimosa and manymimosas (the lightweight
# MissingBench-arch models, not yet in the paper's model table). FLOPs only
# depend on model architecture, not on which dataset a checkpoint was trained
# on, so a single _23.yaml config per model is enough; each run writes its
# report to that config's own <results_dir>/flops.txt (see
# `mimose flops --output-path`).
MODELS=(
    a2fseg
    dcseg
    imfuse
    ims2trans
    inoutfusion
    lckd
    m2ftrans
    m3fecon
    mifpn
    mmformer
    mmmvit
    rfl
    rfnet
    robustseg
    sfusion
    srmnet
    uhved
    unetmfi
    tinymimosa
    manymimosas
)

for model in "${MODELS[@]}"; do
    echo "=== ${model} ==="
    mimose flops --config "${model}_23.yaml" --device cpu
done
