#!/usr/bin/env bash
# Submits the 8 per-model CPU flops jobs (flops_mimosa_<size>_cpu.sh) for the
# mimosa_[size] family, filling in the CPU latency data that's missing
# alongside their existing GPU-only flops_gpu.csv (see flops_mimosa_size_gpu.sh).
set -euo pipefail

sizes=(base gargantuan huge large medium micro small tiny)

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for size in "${sizes[@]}"; do
    script="${script_dir}/flops_mimosa_${size}_cpu.sh"
    if [ ! -f "${script}" ]; then
        echo "Missing ${script}" >&2
        exit 1
    fi
    sbatch "${script}"
done
