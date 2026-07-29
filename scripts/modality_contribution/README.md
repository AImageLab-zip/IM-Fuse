# Modality contribution

Matched marginal Dice contribution of each MRI modality (T1c, T1n, T2f, T2w)
for missing-modality brain-tumor segmentation, per model and tumor region
(WT/TC/ET). See the module docstring in `modality_contribution.py` for the
full method description.

## Install

```
pip install -r requirements.txt
```

## Run directly against this repo's real results (no pre-built CSV needed)

```
python modality_contribution.py \
    --results-dir /work/phd_mimose/results \
    --dataset MB96-BraTS25prechp \
    --output-dir outputs \
    --bootstrap-repetitions 2000
```

`--dataset` selects which (checkpoint, cohort) group to load:
`BraTS18`, `BraTS25-pre` (official test split), `MB96-BraTS18chp`,
`MB96-BraTS25prechp` (MissingBench-96 internal cohort; the default, and
the one used for the main paper figure).

## Run against a pre-built long-format CSV

```
python modality_contribution.py \
    --input case_level_results.csv \
    --dataset MB96_BraTS25 \
    --output-dir outputs \
    --bootstrap-repetitions 2000
```

Expected columns: `dataset, model, region, configuration, dice`, plus
optional `checkpoint, case_id, hd95` (patient-level bootstrap CIs require
`case_id` and `checkpoint`; without them only config-level point estimates
are reported).

## Outputs

Written to `--output-dir`:
- `modality_marginal_contribution_main.{pdf,svg,png}` -- the 3-panel (WT/TC/ET) heatmap figure
- `modality_marginal_summary.csv`
- `modality_marginal_matched_pairs.csv`
- `modality_marginal_bootstrap.csv` (only when case-level data is available)
- `analysis_report.txt` -- validation summary, per-model mean Dice, and the suggested figure caption
