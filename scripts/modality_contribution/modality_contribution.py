#!/usr/bin/env python3
"""Matched marginal Dice contribution of each MRI modality.

For missing-modality brain-tumor segmentation, a model is evaluated on all 15
non-empty subsets of {T1c, T1n, T2f, T2w}. This script asks, per model and
tumor region (WT/TC/ET): how much Dice does adding modality q typically buy,
holding everything else fixed? It answers that with MATCHED pairs only --
Dice(S union {q}) - Dice(S) for every non-empty subset S of the other three
modalities (7 such S per modality: C(3,1)+C(3,2)+C(3,3)) -- never by comparing
the unmatched groups "all configs with q" vs "all configs without q".

Two data sources are supported:

1. --results-dir (default): reads real per-case results directly off disk --
   <results-dir>/<model>_<dsnum>/results_[internal_]fold*_per_subject.csv,
   wide format (one row per case x configuration). Reshaped to the long
   schema below in-process; no separate adapter script or pre-built CSV
   needed. --dataset selects which of the four (source, dsnum) groups to
   read (see DATASET_GROUPS), matching the convention used elsewhere in this
   repo's scripts/ (generate_summary_tables.py's GROUPS/DATASET_LABELS):
   dsnum "18"/"23" = the BraTS2018-trained / BraTS25-pre-trained checkpoint;
   "internal" = the MissingBench-96 (MB-96) cohort. The default,
   MB96-BraTS25prechp, is "models trained on BraTS25-pre, evaluated on
   MissingBench-96".

2. --input: a pre-built long-format CSV (dataset, model, region,
   configuration, dice, and optionally checkpoint/case_id/hd95), for
   portability outside this repo or once such a CSV exists elsewhere.
   Columns are validated; if they don't match, a mapping recommendation is
   printed instead of guessing.

Case-level data (case_id + checkpoint, available from --results-dir, where
"checkpoint" = fold1/fold3/fold5, i.e. the three independently-trained model
replicates) drives a paired patient-level bootstrap 95% CI per (model,
region, modality). Without it (a bare --input CSV lacking those columns),
only the config-level point estimate is reported, and that limitation is
stated in analysis_report.txt.

Outputs (into --output-dir): modality_marginal_contribution_main.{pdf,svg,png},
modality_marginal_summary.csv, modality_marginal_matched_pairs.csv,
modality_marginal_bootstrap.csv (case-level only), analysis_report.txt.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from tqdm import tqdm  # noqa: E402

MODALITIES: tuple[str, ...] = ("T1c", "T1n", "T2f", "T2w")
MODALITY_LOOKUP: dict[str, str] = {m.lower(): m for m in MODALITIES}
REGIONS: tuple[str, ...] = ("WT", "TC", "ET")

# (source, dsnum) per dataset label, mirroring generate_summary_tables.py's
# GROUPS convention but redefined locally -- this script has no import
# dependency on the rest of scripts/, by design (see module docstring).
DATASET_GROUPS: dict[str, dict[str, str]] = {
    "BraTS18": {"source": "official", "dsnum": "18"},
    "BraTS25-pre": {"source": "official", "dsnum": "23"},
    "MB96-BraTS18chp": {"source": "internal", "dsnum": "18"},
    "MB96-BraTS25prechp": {"source": "internal", "dsnum": "23"},
}
DEFAULT_DATASET = "MB96-BraTS25prechp"

# Dropped by default -- deprecated/duplicate model variants, not part of the
# reported comparison. Still overridable via --exclude-models.
DEFAULT_EXCLUDED_MODELS = "manymimosas,tinymimosa"

# Display-name casing, matched to generate_summary_tables.py's
# display_names/FALLBACK_DISPLAY_NAMES output (see outputs/summary_table_mean
# for the reference table these mirror), keyed by normalize_model_name(...).
# Any model not listed here falls back to its directory name, upper-cased.
DISPLAY_NAMES: dict[str, str] = {
    "a2fseg": "A2FSeg",
    "dcseg": "DC-Seg",
    "imfuse": "IM-Fuse",
    "ims2trans": "IMS2Trans",
    "inoutfusion": "InOutFusion",
    "lckd": "LCKD",
    "m2ftrans": "M2FTrans",
    "m3fecon": "M3FeCon",
    "mifpn": "MIFPN",
    "mmformer": "mmFormer",
    "mmmvit": "MMMViT",
    "rfl": "RFL",
    "rfnet": "RFNet",
    "robustseg": "RobustSeg",
    "sfusion": "SFusion",
    "srmnet": "SRMNet",
    "uhved": "U-HVED",
    "unetmfi": "UNET-MFI",
    "manymimosas": "ManyMimosas",
    "tinymimosa": "TinyMimosa",
}

LONG_REQUIRED_COLUMNS = ["dataset", "model", "region", "configuration", "dice"]
LONG_OPTIONAL_COLUMNS = ["checkpoint", "case_id", "hd95"]

CAPTION = (
    "Matched marginal Dice contribution of each MRI modality. Each cell "
    "reports the mean change in Dice obtained by adding the indicated "
    "modality to an otherwise identical set of available modalities. "
    "Contributions are averaged over all seven valid matched configuration "
    "pairs and over the trained checkpoints. Results are shown separately "
    "for (a) whole tumor, (b) tumor core, and (c) enhancing tumor. Positive "
    "values indicate improved segmentation after adding the modality, while "
    "negative values indicate reduced performance. A dot marks contributions "
    "whose paired patient-bootstrap 95% confidence interval excludes zero. "
    "All panels use the same model ordering and color scale."
)


# --------------------------------------------------------------------------- #
# Validation bookkeeping.
# --------------------------------------------------------------------------- #
@dataclass
class ValidationReport:
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"WARNING: {msg}", file=sys.stderr)

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        print(f"ERROR: {msg}", file=sys.stderr)

    def note(self, msg: str) -> None:
        self.info.append(msg)
        print(msg)

    def fail_if_errors(self) -> None:
        if self.errors:
            print(f"\n{len(self.errors)} fatal error(s); aborting.", file=sys.stderr)
            sys.exit(1)


# --------------------------------------------------------------------------- #
# Loading: --results-dir (real per-case data, no adapter step) or --input.
# --------------------------------------------------------------------------- #
def normalize_model_name(raw: str) -> str:
    """Lowercase, alnum-only key used to group directories into one model."""
    return re.sub(r"[^a-z0-9]", "", raw.lower())


def display_model_name(raw: str) -> str:
    """Casing used throughout this repo's other summary tables (DISPLAY_NAMES),
    falling back to an upper-cased copy of `raw` for anything not listed."""
    return DISPLAY_NAMES.get(normalize_model_name(raw), raw.upper())


def discover_model_dirs(
    results_dir: Path, dsnum: str, exclude_models: set[str]
) -> list[tuple[str, Path]]:
    """[(display_model_name, dir_path), ...] for <model>_<dsnum> dirs directly
    under results_dir, skipping any whose normalized name is in exclude_models."""
    found = []
    for entry in sorted(results_dir.iterdir()):
        if not entry.is_dir():
            continue
        m = re.match(r"^(.+)_(\d{2})$", entry.name)
        if not m or m.group(2) != dsnum:
            continue
        model_raw = m.group(1)
        if normalize_model_name(model_raw) in exclude_models:
            continue
        found.append((model_raw, entry))
    return found


def fold_files(model_dir: Path, source: str) -> list[tuple[str, Path]]:
    """[(checkpoint_label, path), ...] sorted by fold number -- "checkpoint"
    here means one of the three independently fold-trained model replicates
    (fold1/fold3/fold5), each evaluated once on every case/configuration."""
    pattern = "results_internal_fold*_per_subject.csv" if source == "internal" else "results_fold*_per_subject.csv"

    def fold_number(p: Path) -> int:
        return int(re.search(r"fold(\d+)", p.name).group(1))

    paths = sorted(model_dir.glob(pattern), key=fold_number)
    return [(f"fold{fold_number(p)}", p) for p in paths]


def read_wide_per_subject_csv(
    path: Path, model: str, dataset_key: str, checkpoint: str
) -> pd.DataFrame:
    """Reshape one wide per-subject CSV (columns: subject, modalities,
    WT_dice, TC_dice, ET_dice, WT_hd95, TC_hd95, ET_hd95, ...) into the long
    schema this script operates on from here on: one row per
    (case_id, region, configuration)."""
    wide = pd.read_csv(path)
    frames = []
    for region in REGIONS:
        dice_col, hd95_col = f"{region}_dice", f"{region}_hd95"
        frames.append(
            pd.DataFrame(
                {
                    "dataset": dataset_key,
                    "model": model,
                    "checkpoint": checkpoint,
                    "case_id": wide["subject"].astype(str),
                    "region": region,
                    "configuration": wide["modalities"],
                    "dice": wide[dice_col],
                    "hd95": wide[hd95_col] if hd95_col in wide.columns else np.nan,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def load_from_results_dir(
    results_dir: Path, dataset_key: str, exclude_models: set[str], report: ValidationReport
) -> pd.DataFrame:
    if dataset_key not in DATASET_GROUPS:
        report.error(
            f"Unknown --dataset '{dataset_key}'; choose one of {sorted(DATASET_GROUPS)}"
        )
        report.fail_if_errors()
    group = DATASET_GROUPS[dataset_key]
    model_dirs = discover_model_dirs(results_dir, group["dsnum"], exclude_models)
    if not model_dirs:
        report.error(
            f"No <model>_{group['dsnum']} directories found under {results_dir}"
        )
        report.fail_if_errors()

    frames = []
    for model_raw, model_dir in model_dirs:
        folds = fold_files(model_dir, group["source"])
        if not folds:
            report.warn(f"{model_dir.name}: no per-subject CSVs found, skipping model")
            continue
        for checkpoint, path in folds:
            frames.append(read_wide_per_subject_csv(path, display_model_name(model_raw), dataset_key, checkpoint))
    if not frames:
        report.error("No usable per-subject data found; nothing to analyze")
        report.fail_if_errors()
    df = pd.concat(frames, ignore_index=True)
    report.note(
        f"Loaded {len(df)} long-format rows from {results_dir} "
        f"({dataset_key}: {group['source']} split, dsnum={group['dsnum']}), "
        f"{df['model'].nunique()} models, {df['case_id'].nunique()} cases, "
        f"{df['checkpoint'].nunique()} checkpoints."
    )
    return df


def validate_and_map_columns(df: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    """If df already has the required long-format columns (case-insensitive),
    normalize their names and return. Otherwise print a best-effort mapping
    recommendation (closest-name guesses) and abort rather than guess."""
    lower_map = {c.lower(): c for c in df.columns}
    missing = [c for c in LONG_REQUIRED_COLUMNS if c not in lower_map]
    if not missing:
        rename = {lower_map[c]: c for c in LONG_REQUIRED_COLUMNS}
        for c in LONG_OPTIONAL_COLUMNS:
            if c in lower_map:
                rename[lower_map[c]] = c
        return df.rename(columns=rename)

    report.error(
        f"--input columns {list(df.columns)} are missing required "
        f"{missing}. Expected long-format columns: "
        f"{LONG_REQUIRED_COLUMNS} (+ optional {LONG_OPTIONAL_COLUMNS})."
    )
    print("Column-mapping suggestions (closest name matches):", file=sys.stderr)
    for want in missing:
        candidates = sorted(
            df.columns,
            key=lambda c: _levenshtein(c.lower(), want),
        )[:3]
        print(f"  {want}  <-  one of {candidates}?", file=sys.stderr)
    report.fail_if_errors()
    return df  # unreachable; fail_if_errors exits


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def load_input_csv(path: Path, dataset_key: str, report: ValidationReport) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = validate_and_map_columns(df, report)
    if "dataset" not in df.columns or df["dataset"].isna().all():
        df["dataset"] = dataset_key
    report.note(f"Loaded {len(df)} rows from --input {path}.")
    return df


# --------------------------------------------------------------------------- #
# Configuration parsing + Dice-scale detection.
# --------------------------------------------------------------------------- #
def parse_configuration(raw: str, report: ValidationReport) -> tuple[frozenset[str], str] | None:
    """Parse a configuration string into (modality set, canonical string), or
    None (with a warning) if it contains unknown tokens or duplicates."""
    tokens = [t for t in re.split(r"[_+,;\s]+", str(raw).strip()) if t]
    modalities = set()
    for tok in tokens:
        canon = MODALITY_LOOKUP.get(tok.lower())
        if canon is None:
            report.warn(f"Unknown modality token '{tok}' in configuration '{raw}'; row dropped")
            return None
        if canon in modalities:
            report.warn(f"Duplicate modality '{canon}' in configuration '{raw}'; row dropped")
            return None
        modalities.add(canon)
    if not modalities:
        report.warn(f"Empty configuration '{raw}'; row dropped")
        return None
    canonical = "_".join(m for m in MODALITIES if m in modalities)
    return frozenset(modalities), canonical


def detect_dice_scale(series: pd.Series) -> str:
    """'0-100' if any value exceeds 1.5 (a 0-1 fraction never would), else '0-1'."""
    return "0-100" if series.max(skipna=True) > 1.5 else "0-1"


def prepare_dataframe(df: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    """Parse configurations to canonical form, drop unparseable rows,
    normalize Dice to a 0-1 fraction (auto-detecting the input scale),
    uniform the model column's casing (DISPLAY_NAMES; unrecognized names are
    left as given from --input), and fill in optional columns absent from
    --input so downstream code can assume they always exist."""
    for col in LONG_OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["model"] = df["model"].apply(
        lambda v: DISPLAY_NAMES.get(normalize_model_name(str(v)), v)
    )

    parsed = df["configuration"].apply(lambda v: parse_configuration(v, report))
    df = df.assign(
        modality_set=parsed.apply(lambda p: p[0] if p else None),
        configuration_canonical=parsed.apply(lambda p: p[1] if p else None),
    )
    n_dropped = df["configuration_canonical"].isna().sum()
    if n_dropped:
        report.warn(f"Dropped {n_dropped} row(s) with unparseable configurations")
    df = df.dropna(subset=["configuration_canonical"]).copy()

    scale = detect_dice_scale(df["dice"])
    report.note(f"Detected Dice scale: {scale}.")
    if scale == "0-100":
        df["dice"] = df["dice"] / 100.0
    out_of_range = df[(df["dice"] < -1e-6) | (df["dice"] > 1 + 1e-6)]
    if len(out_of_range):
        report.warn(f"{len(out_of_range)} Dice value(s) fall outside [0,1] after scale normalization")

    dup_cols = ["model", "region", "checkpoint", "case_id", "configuration_canonical"]
    dup_mask = df.duplicated(subset=dup_cols, keep=False) & df["case_id"].notna()
    if dup_mask.any():
        report.warn(f"{dup_mask.sum()} duplicate row(s) found on {dup_cols}; keeping the first")
        df = df[~df.duplicated(subset=dup_cols, keep="first") | ~dup_mask]

    missing_regions = set(REGIONS) - set(df["region"].unique())
    if missing_regions:
        report.warn(f"Missing region(s) entirely: {sorted(missing_regions)}")

    return df


def check_completeness(df: pd.DataFrame, report: ValidationReport) -> None:
    all_configs = {
        "_".join(m for m in MODALITIES if m in combo)
        for size in range(1, 5)
        for combo in itertools.combinations(MODALITIES, size)
    }
    for (model, region), g in df.groupby(["model", "region"]):
        present = set(g["configuration_canonical"].unique())
        missing = all_configs - present
        if missing:
            report.warn(
                f"{model}/{region}: {len(present)}/15 configurations present; "
                f"missing {sorted(missing)}"
            )


# --------------------------------------------------------------------------- #
# Matched pairs: the 7 non-empty subsets of "the other three modalities".
# --------------------------------------------------------------------------- #
def matched_subsets(modality: str) -> list[frozenset[str]]:
    others = [m for m in MODALITIES if m != modality]
    subsets = []
    for size in (1, 2, 3):
        subsets.extend(frozenset(c) for c in itertools.combinations(others, size))
    assert len(subsets) == 7, "each modality must have exactly 7 matched background subsets"
    for s in subsets:
        assert modality not in s, "background subset must not already contain the target modality"
    return subsets


def canon_string(modalities: frozenset[str]) -> str:
    return "_".join(m for m in MODALITIES if m in modalities)


def build_matched_pairs(df: pd.DataFrame, dataset_key: str, report: ValidationReport) -> pd.DataFrame:
    """One row per (model, region, modality, background subset S): the
    config-level (mean over all case_id/checkpoint rows) matched marginal
    Dice gain of adding `modality` to S. This is written straight out as
    modality_marginal_matched_pairs.csv."""
    config_means = df.groupby(["model", "region", "configuration_canonical"])["dice"].mean()

    rows = []
    for (model, region), _ in df.groupby(["model", "region"]):
        for modality in MODALITIES:
            n_valid = 0
            for S in matched_subsets(modality):
                without_str, with_str = canon_string(S), canon_string(S | {modality})
                try:
                    dice_without = config_means[(model, region, without_str)]
                    dice_with = config_means[(model, region, with_str)]
                except KeyError:
                    continue
                n_valid += 1
                rows.append(
                    {
                        "dataset": dataset_key,
                        "model": model,
                        "region": region,
                        "modality": modality,
                        "configuration_without": without_str,
                        "configuration_with": with_str,
                        "background_cardinality": len(S),
                        "dice_without": dice_without,
                        "dice_with": dice_with,
                        "marginal_gain": dice_with - dice_without,
                    }
                )
            if n_valid < 7:
                report.warn(
                    f"{model}/{region}/{modality}: only {n_valid}/7 matched pairs available"
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Case-level aggregation + paired patient bootstrap.
# --------------------------------------------------------------------------- #
def case_level_available(df: pd.DataFrame) -> bool:
    return df["case_id"].notna().any() and df["checkpoint"].notna().any()


def build_group_pivot(sub: pd.DataFrame) -> pd.DataFrame | None:
    """(case_id, checkpoint) x configuration_canonical Dice pivot for one
    (model, region) group -- built once per group and shared across all 4
    modalities' matrices, rather than re-pivoted per modality."""
    if sub.empty:
        return None
    return sub.pivot_table(
        index=["case_id", "checkpoint"], columns="configuration_canonical", values="dice"
    )


def per_patient_pair_matrix(
    pivot: pd.DataFrame, model: str, region: str, modality: str
) -> tuple[np.ndarray | None, list[str]]:
    """(n_patients, 7) matrix of the matched-pair Dice diff for `modality` in
    `model`/`region`, each entry already averaged over checkpoints for that
    patient. Row i, column j = patient i's checkpoint-averaged
    Dice(S_j union {modality}) - Dice(S_j). Returns (None, []) if no pair has
    any complete patient. Returns warnings as strings instead of writing to a
    ValidationReport directly, so this can run inside a worker process."""
    columns = []
    for S in matched_subsets(modality):
        without_str, with_str = canon_string(S), canon_string(S | {modality})
        if without_str not in pivot.columns or with_str not in pivot.columns:
            columns.append(None)
            continue
        diff = (pivot[with_str] - pivot[without_str]).reset_index()
        diff.columns = ["case_id", "checkpoint", "diff"]
        per_case = diff.groupby("case_id")["diff"].mean()  # average over checkpoints
        columns.append(per_case)

    valid_cols = [c for c in columns if c is not None]
    if not valid_cols:
        return None, []
    all_cases = sorted(set.union(*(set(c.index) for c in valid_cols)))
    matrix = np.full((len(all_cases), 7), np.nan)
    for j, c in enumerate(columns):
        if c is None:
            continue
        for i, case in enumerate(all_cases):
            if case in c.index:
                matrix[i, j] = c.loc[case]

    warnings = []
    n_missing_cells = int(np.isnan(matrix).sum())
    if n_missing_cells:
        warnings.append(
            f"{model}/{region}/{modality}: {n_missing_cells} missing patient x pair "
            f"cell(s) out of {matrix.size} (patient lacked one of the two configs)"
        )
    return matrix, warnings


def bootstrap_ci(
    matrix: np.ndarray, n_reps: int, rng: np.random.Generator
) -> tuple[float, float, float, int]:
    """(point_estimate, ci_lower, ci_upper, n_patients), all in Dice pp.
    Point estimate = nanmean over patients and pairs. Each bootstrap
    replicate resamples patient rows (with replacement, keeping every
    pair/checkpoint value belonging to a sampled patient together), then
    takes the same nanmean."""
    n_patients = matrix.shape[0]
    point = np.nanmean(matrix) * 100.0
    if n_patients == 0:
        return point, float("nan"), float("nan"), 0
    idx = rng.integers(0, n_patients, size=(n_reps, n_patients))
    resampled = matrix[idx]  # (n_reps, n_patients, 7)
    with np.errstate(invalid="ignore"):
        reps = np.nanmean(resampled.reshape(n_reps, -1), axis=1) * 100.0
    reps = reps[~np.isnan(reps)]
    if len(reps) == 0:
        return point, float("nan"), float("nan"), n_patients
    ci_lower, ci_upper = np.percentile(reps, [2.5, 97.5])
    return point, ci_lower, ci_upper, n_patients


def auto_worker_count() -> int:
    """CPU count honoring a cgroup/SLURM affinity mask if available (e.g. on
    a shared Cineca node with fewer cores allocated than the machine has),
    falling back to os.cpu_count()."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # sched_getaffinity is Linux-only
        return max(1, os.cpu_count() or 1)


def _process_unit(
    model: str, region: str, modality: str, sub: pd.DataFrame, dataset_key: str, n_reps: int, seed: int
) -> dict:
    """Worker unit: everything needed to bootstrap one (model, region,
    modality) unit, run in a separate process by compute_case_level_summary.
    Self-contained (only module-level functions + plain arguments) so it's
    picklable for ProcessPoolExecutor. `sub` is already filtered to this
    unit's (model, region); the pivot is rebuilt per unit (cheap relative to
    the bootstrap itself) so each unit is a fully independent task and
    progress can be reported per unit rather than only once per group."""
    rng = np.random.default_rng(seed)
    pivot = build_group_pivot(sub)
    if pivot is None:
        return {"summary_rows": [], "bootstrap_rows": [], "warnings": []}

    matrix, warnings = per_patient_pair_matrix(pivot, model, region, modality)
    if matrix is None:
        return {"summary_rows": [], "bootstrap_rows": [], "warnings": warnings}

    n_pairs_valid = int((~np.isnan(matrix).all(axis=0)).sum())
    point, lo, hi, n_patients = bootstrap_ci(matrix, n_reps, rng)
    summary_rows = [
        {
            "dataset": dataset_key,
            "model": model,
            "region": region,
            "modality": modality,
            "mean_gain": point,
            "ci_lower": lo,
            "ci_upper": hi,
            "number_of_pairs": n_pairs_valid,
            "number_of_cases": n_patients,
        }
    ]
    n_patients_eff = matrix.shape[0]
    idx = rng.integers(0, n_patients_eff, size=(n_reps, n_patients_eff))
    with np.errstate(invalid="ignore"):
        reps = np.nanmean(matrix[idx].reshape(n_reps, -1), axis=1) * 100.0
    bootstrap_rows = [
        {
            "dataset": dataset_key,
            "model": model,
            "region": region,
            "modality": modality,
            "replicate": rep_i,
            "gain": val,
        }
        for rep_i, val in enumerate(reps)
    ]
    return {"summary_rows": summary_rows, "bootstrap_rows": bootstrap_rows, "warnings": warnings}


def compute_case_level_summary(
    df: pd.DataFrame,
    dataset_key: str,
    n_reps: int,
    seed: int,
    report: ValidationReport,
    workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (summary_df, bootstrap_replicates_df). Each (model, region,
    modality) unit is independent, so units are farmed out across `workers`
    processes (auto-detected from CPU affinity if not given) -- one tick of
    the progress bar per unit finished, not per (model, region) group, so
    progress is reported 4x more often (one tick per modality rather than
    once per group's 4 modalities). Per-unit RNG seeds are spawned
    deterministically from `seed` via SeedSequence, keyed by each unit's
    position in a fixed (model, region, modality) sort order -- so results
    are identical regardless of worker count or completion order."""
    workers = workers or auto_worker_count()
    groups = sorted(df.groupby(["model", "region"]).groups.keys())
    group_subs = {
        (model, region): df[(df["model"] == model) & (df["region"] == region)]
        for model, region in groups
    }
    units = sorted((model, region, modality) for model, region in groups for modality in MODALITIES)
    seeds = np.random.SeedSequence(seed).spawn(len(units))

    tasks = [
        (model, region, modality, group_subs[(model, region)], dataset_key, n_reps, s)
        for (model, region, modality), s in zip(units, seeds)
    ]

    summary_rows, bootstrap_rows = [], []
    if workers == 1 or len(tasks) <= 1:
        report.note(f"Bootstrapping {len(tasks)} (model, region, modality) unit(s) sequentially.")
        results = [
            _process_unit(*task)
            for task in tqdm(tasks, desc="bootstrap units", unit="unit")
        ]
    else:
        n_workers = min(workers, len(tasks))
        report.note(
            f"Bootstrapping {len(tasks)} (model, region, modality) unit(s) across {n_workers} worker processes."
        )
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_process_unit, *task) for task in tasks]
            results = [
                f.result()
                for f in tqdm(
                    as_completed(futures), total=len(futures), desc="bootstrap units", unit="unit"
                )
            ]

    for result in results:
        summary_rows.extend(result["summary_rows"])
        bootstrap_rows.extend(result["bootstrap_rows"])
        for w in result["warnings"]:
            report.warn(w)

    return pd.DataFrame(summary_rows), pd.DataFrame(bootstrap_rows)


def compute_config_level_summary(matched_pairs: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """Fallback summary (no case_id/checkpoint): mean_gain from config-level
    matched_pairs, no confidence interval."""
    grouped = matched_pairs.groupby(["model", "region", "modality"]).agg(
        mean_gain=("marginal_gain", lambda s: s.mean() * 100.0),
        number_of_pairs=("marginal_gain", "size"),
    )
    grouped["dataset"] = dataset_key
    grouped["ci_lower"] = np.nan
    grouped["ci_upper"] = np.nan
    grouped["number_of_cases"] = 0
    return grouped.reset_index()[
        [
            "dataset",
            "model",
            "region",
            "modality",
            "mean_gain",
            "ci_lower",
            "ci_upper",
            "number_of_pairs",
            "number_of_cases",
        ]
    ]


# --------------------------------------------------------------------------- #
# Model ordering.
# --------------------------------------------------------------------------- #
def order_models(df: pd.DataFrame) -> list[str]:
    """Models in case-insensitive alphabetical order -- the single row order
    reused by every figure panel and both CSVs. Case-insensitive so a
    lowercase-leading name (e.g. "mmFormer") sorts next to its peers ("M2FTrans",
    "MIFPN", ...) instead of after every upper-case name."""
    return sorted(df["model"].unique(), key=str.lower)


# --------------------------------------------------------------------------- #
# Figure.
# --------------------------------------------------------------------------- #
def diverging_colormap() -> LinearSegmentedColormap:
    """Blue (negative) <-> neutral gray (zero) <-> red (positive), the
    diverging pair from this repo's dataviz-skill reference palette
    (references/palette.md: 'blue <-> red ... neutral midpoint is gray')."""
    stops = [
        (0.00, "#0d366b"),
        (0.25, "#2a78d6"),
        (0.50, "#f0efec"),
        (0.75, "#e34948"),
        (1.00, "#7a1f1e"),
    ]
    return LinearSegmentedColormap.from_list("modality_gain_diverging", stops, N=256)


def build_figure(
    summary: pd.DataFrame,
    ordered_models: list[str],
    output_dir: Path,
) -> None:
    panels = REGIONS
    panel_letters = {"WT": "a", "TC": "b", "ET": "c"}

    gain_matrices, sig_matrices = {}, {}
    for region in panels:
        g = summary[summary["region"] == region].set_index(["model", "modality"])
        mat = np.full((len(ordered_models), len(MODALITIES)), np.nan)
        sig = np.zeros_like(mat, dtype=bool)
        for i, model in enumerate(ordered_models):
            for j, modality in enumerate(MODALITIES):
                if (model, modality) not in g.index:
                    continue
                row = g.loc[(model, modality)]
                mat[i, j] = row["mean_gain"]
                lo, hi = row["ci_lower"], row["ci_upper"]
                sig[i, j] = bool(pd.notna(lo) and pd.notna(hi) and (lo > 0 or hi < 0))
        gain_matrices[region] = mat
        sig_matrices[region] = sig

    max_abs = max(np.nanmax(np.abs(m)) for m in gain_matrices.values() if not np.all(np.isnan(m)))
    vmax = math.ceil(max_abs) if max_abs > 0 else 1.0
    vmin = -vmax

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.linewidth": 0.6,
        }
    )
    n_models = len(ordered_models)
    fig_height = max(2.0, 0.125 * n_models + 0.85)
    fig, axes = plt.subplots(1, 3, figsize=(7.05, fig_height), constrained_layout=True)

    cmap = diverging_colormap()
    im = None
    for ax, region in zip(axes, panels):
        mat = gain_matrices[region]
        sig = sig_matrices[region]
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(MODALITIES)))
        ax.set_xticklabels(MODALITIES, rotation=0)
        ax.set_yticks(range(n_models))
        if region == panels[0]:
            ax.set_yticklabels(ordered_models, fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.set_title(f"({panel_letters[region]}) {region}")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if np.isnan(val):
                    ax.text(j, i, "--", ha="center", va="center", fontsize=8, color="0.5")
                    continue
                lightness = abs(val) / vmax if vmax else 0
                text_color = "white" if lightness > 0.55 else "black"
                ax.text(
                    j, i, f"{val:+.1f}", ha="center", va="center",
                    fontsize=8, color=text_color,
                )
                if sig[i, j]:
                    ax.plot(
                        j + 0.45, i - 0.32, marker="o", markersize=2.2,
                        color=text_color, markeredgewidth=0,
                    )

    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015, label="Matched Marginal Dice Gain (pp)")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "modality_marginal_contribution_main"
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.svg")
    fig.savefig(f"{stem}.png", dpi=600)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report.
# --------------------------------------------------------------------------- #
def write_report(
    path: Path,
    dataset_key: str,
    df: pd.DataFrame,
    report: ValidationReport,
    case_level: bool,
    n_reps: int,
) -> None:
    mean_dice_by_model = (
        df.groupby("model")["dice"].mean().sort_values(ascending=False) * 100.0
    )
    lines = [
        f"Matched marginal modality-contribution analysis -- dataset '{dataset_key}'",
        "=" * 72,
        "",
        "Overall mean Dice (pp) by model, this dataset, all 15 configurations x "
        "WT/TC/ET (compare against your existing aggregate benchmark table):",
    ]
    for model, val in mean_dice_by_model.items():
        lines.append(f"  {model:<20s} {val:6.2f}")
    lines.append("")

    lines.append("Case-level availability:")
    if case_level:
        lines.append(
            f"  case_id and checkpoint columns available -- patient-level "
            f"bootstrap 95% CIs computed ({n_reps} repetitions per model/region/modality)."
        )
    else:
        lines.append(
            "  LIMITATION: case_id/checkpoint not available in the input; "
            "mean_gain is computed from configuration-level means only. "
            "No confidence intervals are reported (ci_lower/ci_upper are blank)."
        )
    lines.append("")

    lines.append(f"Validation: {len(report.errors)} error(s), {len(report.warnings)} warning(s).")
    for w in report.warnings:
        lines.append(f"  WARNING: {w}")
    lines.append("")

    lines.append("Suggested figure caption:")
    lines.append(CAPTION)
    lines.append("")

    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# main.
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=None, help="Pre-built long-format CSV (see module docstring).")
    p.add_argument(
        "--results-dir", type=Path, default=Path("/work/phd_mimose/results"),
        help="Directory of <model>_<dsnum> result dirs, used when --input is omitted.",
    )
    p.add_argument(
        "--dataset", default=DEFAULT_DATASET, choices=sorted(DATASET_GROUPS),
        help="Which (source, checkpoint) group to load / label to use (default: %(default)s).",
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory to write outputs into.")
    p.add_argument("--bootstrap-repetitions", type=int, default=20000, help="Patient bootstrap repetitions.")
    p.add_argument(
        "--exclude-models", default=DEFAULT_EXCLUDED_MODELS,
        help="Comma-separated model names to drop (default: %(default)s).",
    )
    p.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed, for reproducibility.")
    p.add_argument(
        "--workers", type=int, default=None,
        help="Worker processes for the patient bootstrap (default: auto-detected CPU count).",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    report = ValidationReport()
    exclude_models = {normalize_model_name(m) for m in args.exclude_models.split(",") if m.strip()}

    if args.input is not None:
        df = load_input_csv(args.input, args.dataset, report)
    else:
        df = load_from_results_dir(args.results_dir, args.dataset, exclude_models, report)

    df = prepare_dataframe(df, report)
    check_completeness(df, report)
    report.fail_if_errors()

    ordered_models = order_models(df)
    matched_pairs = build_matched_pairs(df, args.dataset, report)

    case_level = case_level_available(df)
    if case_level:
        summary, bootstrap = compute_case_level_summary(
            df, args.dataset, args.bootstrap_repetitions, args.seed, report, workers=args.workers
        )
    else:
        summary = compute_config_level_summary(matched_pairs, args.dataset)
        bootstrap = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "modality_marginal_summary.csv"
    pairs_path = args.output_dir / "modality_marginal_matched_pairs.csv"
    summary.to_csv(summary_path, index=False)
    matched_pairs.to_csv(pairs_path, index=False)
    print(f"Wrote {summary_path}")
    print(f"Wrote {pairs_path}")

    if bootstrap is not None:
        bootstrap_path = args.output_dir / "modality_marginal_bootstrap.csv"
        bootstrap.to_csv(bootstrap_path, index=False)
        print(f"Wrote {bootstrap_path}")

    build_figure(summary, ordered_models, args.output_dir)
    for ext in ("pdf", "svg", "png"):
        print(f"Wrote {args.output_dir / f'modality_marginal_contribution_main.{ext}'}")

    report_path = args.output_dir / "analysis_report.txt"
    write_report(report_path, args.dataset, df, report, case_level, args.bootstrap_repetitions)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
