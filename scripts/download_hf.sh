#!/bin/bash
set -e

BASE=/work/phd_mimose/runs
REPO=AImageLab-Zip/mimose_runs

JOBS=${JOBS:-20}
RETRIES=${RETRIES:-3}

mkdir -p "$BASE"

download_one() {
    local folder="$1"
    local attempt=1
    while (( attempt <= RETRIES )); do
        if hf download "$REPO" \
            --repo-type dataset \
            --include "$folder/**" \
            --local-dir "$BASE"; then
            return 0
        fi
        echo "download failed for $folder (attempt $attempt/$RETRIES), retrying..." >&2
        ((attempt++))
        sleep $((attempt * 5))
    done
    echo "download permanently failed for $folder" >&2
    return 1
}
export -f download_one
export REPO BASE RETRIES

folders=()

for fold in 1 3 5; do
    folders+=("manymimosas18_fold${fold}")
done

for fold in 1 3 5; do
    folders+=("manymimosas23_fold${fold}")
done

for fold in 1 3 5; do
    folders+=("tinymimosa23_fold${fold}")
done

for model in lckd rfnet robustseg uhved inoutfusion ims2trans mmmvit rfl unetmfi m2ftrans srmnet; do
    for fold in 1 3 5; do
        folders+=("${model}23_fold${fold}")
    done
done

printf '%s\n' "${folders[@]}" | xargs -I{} -P "$JOBS" bash -c 'download_one "$@"' _ {}