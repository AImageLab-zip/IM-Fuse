#!/usr/bin/env bash
# Submits the 20 per-model GPU flops jobs (flops_<model>_gpu.sh) for the
# paper's Table 1 (tab:models) 18 models plus TinyMimosa and ManyMimosas,
# filling in the GPU-only kernel-launch-count and peak-reserved-memory data
# (see measure_kernel_launches_and_memory in mimose/flops.py) that the
# existing CPU-only flops_<model>.sh jobs can't provide.
set -euo pipefail

models=(
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
)

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for model in "${models[@]}"; do
    script="${script_dir}/flops_${model}_gpu.sh"
    if [ ! -f "${script}" ]; then
        echo "Missing ${script}" >&2
        exit 1
    fi
    sbatch "${script}"
done
