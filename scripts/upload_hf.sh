#!/bin/bash
set -e

BASE=/work/phd_mimose/runs/
REPO=AImageLab-Zip/mimose_runs

for model in unetmfi; do
    for fold in 1 3 5; do
        folder="${model}23_fold${fold}"

        hf upload "$REPO" \
            "$BASE/$folder" \
            "$folder" \
            --repo-type dataset
    done
done