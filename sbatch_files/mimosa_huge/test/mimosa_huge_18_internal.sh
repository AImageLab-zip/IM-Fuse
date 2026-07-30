#!/bin/bash
#SBATCH --job-name=MIMOSA_HUGE18_TEST_INTERNAL
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/mimosa_huge_18_test_internal.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/mimosa_huge_18_test_internal.out
#SBATCH --gres=gpu:2
#SBATCH --account=phd_mimose
#SBATCH --constraint=gpu_RTXPro6000B_96G

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

# Tests the brats18-trained mimosa_huge checkpoints (folds 1/3/5) against
# the internal dataset, reusing each checkpoint's own art_dir/run_suffix
# resolution but pointing the data at the internal-preprocessed set and the
# internal split file.
SPLIT_FILE=/homes/ocarpentiero/MiMoSe/src/mimose/data/splits/internal.json
DATA_DIR=/work/phd_mimose/internal_raw/simpleunet_internal

for trained_fold in 1 3 5; do
    mimose test \
        --config mimosa_huge_18.yaml \
        --run-suffix "fold${trained_fold}" \
        --dataset-type internal \
        --split-file "$SPLIT_FILE" \
        --fold 1 \
        --data-dir "$DATA_DIR" \
        --output-path "/work/phd_mimose/results/mimosa_huge_18/results_internal.txt" \
        --num-workers 60
done
