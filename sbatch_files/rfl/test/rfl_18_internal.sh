#!/bin/bash
#SBATCH --job-name=RFL18_TEST_INTERNAL
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/rfl_18_test_internal.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/rfl_18_test_internal.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate

# Tests the brats18-trained rfl checkpoints (folds 1/3/5) against
# the internal dataset, reusing each checkpoint's own art_dir/run_suffix
# resolution but pointing the data at the internal-preprocessed set and the
# internal split file.
SPLIT_FILE=/homes/ocarpentiero/MiMoSe/src/mimose/data/splits/internal.json
DATA_DIR=/work/phd_mimose/internal_raw/rfl_internal

for trained_fold in 1 3 5; do
    mimose test \
        --config rfl_18.yaml \
        --run-suffix "fold${trained_fold}" \
        --dataset-type internal \
        --split-file "$SPLIT_FILE" \
        --fold 1 \
        --data-dir "$DATA_DIR" \
        --output-path "/work/phd_mimose/results/rfl_18/results_internal.txt"
done
