#!/bin/bash
#SBATCH --job-name=MIMOSA_HUGE18_TEST_INTERNAL_FOLD3
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/mimosa_huge_18_test_internal_fold3.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/mimosa_huge_18_test_internal_fold3.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose
#SBATCH --constraint=gpu_RTXPro6000B_96G

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

# Tests the brats18-trained mimosa_huge checkpoint (fold 3) against
# the internal dataset, reusing each checkpoint's own art_dir/run_suffix
# resolution but pointing the data at the internal-preprocessed set and the
# internal split file.
SPLIT_FILE=/homes/ocarpentiero/MiMoSe/src/mimose/data/splits/internal.json
DATA_DIR=/work/phd_mimose/internal_raw/simpleunet_internal

mimose test \
    --config mimosa_huge_18.yaml \
    --run-suffix "fold3" \
    --dataset-type internal \
    --split-file "$SPLIT_FILE" \
    --fold 1 \
    --data-dir "$DATA_DIR" \
    --output-path "/work/phd_mimose/results/mimosa_huge_18/results_internal.txt" \
    --num-workers 8
