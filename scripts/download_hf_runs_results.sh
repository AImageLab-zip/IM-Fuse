#!/bin/bash
# Download baseline run folders (checkpoints + wandb) and results from HF.
#   runs    -> $RUNS_DIR/{model}{brats}_fold{fold}/
#   results -> $RESULTS_ROOT/results/{model}_{brats}/results_fold{fold}*
# Override any of the variables below via env, e.g. MODELS=mmformer FOLDS=2.
set -euo pipefail

REPO=${REPO:-AImageLab-Zip/mimose_runs}
RUNS_DIR=${RUNS_DIR:-/work/phd_mimose/runs}
RESULTS_ROOT=${RESULTS_ROOT:-/work/phd_mimose}
HF=${HF:-/homes/ocarpentiero/MiMoSe/.venv/bin/hf}
PY=${PY:-/homes/ocarpentiero/MiMoSe/.venv/bin/python}

JOBS=${JOBS:-1}
RETRIES=${RETRIES:-3}

MODELS=${MODELS:-"a2fseg dcseg imfuse ims2trans inoutfusion lckd m2ftrans m3fecon mifpn mmformer mmmvit rfl rfnet robustseg sfusion srmnet uhved unetmfi"}
BRATS=${BRATS:-"18 23"}
FOLDS=${FOLDS:-"2 4"}

# Results that were never produced; their checkpoints are on HF.
KNOWN_MISSING_RESULTS="imfuse_18 imfuse_23 unetmfi_23"

if "$PY" -I -c "import hf_transfer" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
fi

mkdir -p "$RUNS_DIR" "$RESULTS_ROOT"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Preflight: list what is actually on HF, as "run|<folder>" and "res|<model>_<brats>|<fold>".
"$PY" -I -c '
import re, sys
from huggingface_hub import list_repo_files
present = set()
for f in list_repo_files(sys.argv[1], repo_type="dataset"):
    parts = f.split("/")
    if len(parts) > 1 and parts[0] != "results":
        present.add("run|" + parts[0])
    m = re.fullmatch(r"results/([^/]+)/results_fold(\d+)(?:\.txt|\.xlsx|_per_subject\.csv)", f)
    if m:
        present.add("res|%s|%s" % m.groups())
print("\n".join(sorted(present)))
' "$REPO" > "$TMP/present"

download_one() {
    local item="$1" kind name fold include local_dir
    IFS='|' read -r kind name fold <<< "$item"
    if [[ $kind == run ]]; then
        include="$name/**"
        local_dir="$RUNS_DIR"
    else
        include="results/$name/results_fold${fold}*"
        local_dir="$RESULTS_ROOT"
    fi
    local attempt=1
    while (( attempt <= RETRIES )); do
        if "$HF" download "$REPO" \
            --repo-type dataset \
            --include "$include" \
            --local-dir "$local_dir"; then
            return 0
        fi
        echo "download failed for $item (attempt $attempt/$RETRIES), retrying..." >&2
        ((attempt++))
        sleep $((attempt * 5))
    done
    echo "download permanently failed for $item" >&2
    echo "$item" >> "$TMP/failed"
    return 1
}
export -f download_one
export REPO RUNS_DIR RESULTS_ROOT HF RETRIES TMP

items=()
n_known=0
n_missing=0

for brats in $BRATS; do
    for model in $MODELS; do
        for fold in $FOLDS; do
            for item in "run|${model}${brats}_fold${fold}" "res|${model}_${brats}|${fold}"; do
                # HF results come from v1 runs; never let them overwrite a local v2 run's results.
                if [[ $item == res* && -d $RUNS_DIR/${model}${brats}_v2_fold${fold} ]]; then
                    echo "note: local v2 run ${model}${brats}_v2_fold${fold} exists, not downloading its results" >&2
                    ((++n_known))
                elif grep -qFx "$item" "$TMP/present"; then
                    items+=("$item")
                elif [[ $item == res* && " $KNOWN_MISSING_RESULTS " == *" ${model}_${brats} "* ]]; then
                    echo "note: no results for ${model}_${brats} fold $fold (known gap), skipping" >&2
                    ((++n_known))
                else
                    echo "WARNING: $item not on HF yet, skipping" >&2
                    ((++n_missing))
                fi
            done
        done
    done
done

if (( ${#items[@]} > 0 )); then
    printf '%s\n' "${items[@]}" | xargs -I{} -P "$JOBS" bash -c 'download_one "$@"' _ {} || true
fi

n_failed=0
[[ -f $TMP/failed ]] && n_failed=$(wc -l < "$TMP/failed")

echo
echo "summary: requested=${#items[@]} failed=$n_failed skipped_known=$n_known skipped_missing=$n_missing"
if (( n_failed > 0 )); then
    sed 's/^/  failed: /' "$TMP/failed" >&2
    exit 1
fi
