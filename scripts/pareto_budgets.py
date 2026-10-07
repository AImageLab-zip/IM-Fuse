#!/usr/bin/env python3
"""The pareto_23 budget sweep as a model family, for the Pareto scripts.

One budget = one model. Each entry of BUDGETS is a distinct architecture the
supernet's gates select at that parameter budget, extracted to a standalone
SimpleUnet and then measured on its own: Dice/HD95 from
`sbatch_files/pareto/test_budgets/` and params/FLOPs/memory/latency from
`sbatch_files/pareto/flops_budgets/`. Both halves of a budget write into the
same `pareto_b<label>_23` results directory, which normalizes to a single
model key -- so accuracy and cost join per budget, and the budgets compete
for the frontier individually rather than collapsing into one "pareto" point.

Fold coverage: only `runs/pareto23_fold1` exists, so this family is fold1-only
and passes EXPECTED_FOLDS = ["fold1"] wherever the rest of the project assumes
fold1/fold3/fold5. That is why the numbers here carry no across-fold std --
they are single-fold measurements, not 3-fold means, and should be read as
such next to the 3-fold baselines.

FLOPs live in flops_gpu.csv (a single-budget `mimose flops` run keeps the
plain name, see src/mimose/cli.py), not the flops.csv the literature
baselines use; GPU_FILENAME carries that. There is no CPU run for this
family, so lat_cpu is n/a throughout -- the same situation as mimosa_[size],
and the reason generate_pareto_table.py leaves CPU latency out of the
dominance test.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_reproduction_comparison as brc  # noqa: E402
import generate_flops_table as gft  # noqa: E402
import generate_model_comparison_table as gmc  # noqa: E402

# Family key used by generate_pareto_table.py / generate_pareto_plot.py.
FAMILY = "ParetoBudget"

# Results directories are written as flops_gpu.csv, not flops.csv.
GPU_FILENAME = "flops_gpu.csv"

# Only fold1 of the supernet was trained, so only fold1 was tested.
EXPECTED_FOLDS = ["fold1"]

DATASET = "23"

# (directory label, parameter budget passed to --budget). The label is
# whatever `pareto_b<label>_<dsnum>` uses on disk; it must match the sbatch
# scripts in sbatch_files/pareto/{test,flops}_budgets/ exactly.
BUDGETS = [
    ("1.4M", 1_430_000),
    ("5.0M", 5_000_000),
    ("10.0M", 10_000_000),
    ("20.0M", 20_000_000),
    ("45.0M", 45_000_000),
    ("91.4M", 91_400_000),
]


def results_dirname(label: str, dsnum: str = DATASET) -> str:
    return f"pareto_b{label}_{dsnum}"


def model_norm(label: str) -> str:
    """Normalized model key for a budget, matching what both
    gft.discover_flops_csvs and gst.discover derive from the directory name
    (e.g. "pareto_b10.0M_23" -> "paretob100m" -- normalize() strips the dot,
    so these keys are not human-readable; DISPLAY_NAMES is what tables show)."""
    return brc.normalize(f"pareto_b{label}")


def display_name(label: str) -> str:
    return f"MiMoSe-P {label}"


MODEL_SET = {model_norm(label) for label, _params in BUDGETS}
DISPLAY_NAMES = {model_norm(label): display_name(label) for label, _params in BUDGETS}
# {display_name: budget}, for callers that want to order or annotate points by
# budget rather than by name (display order is alphabetical, which puts 10.0M
# before 5.0M).
BUDGET_BY_DISPLAY = {display_name(label): params for label, params in BUDGETS}


def short_name(display: str) -> str:
    """"MiMoSe-P 10.0M" -> "mp_10.0M", the compact form used on the scatter
    plot, mirroring generate_pareto_plot.py's mb_[size] naming for the
    mimosa_[size] family."""
    _prefix, _, label = display.partition(" ")
    return f"mp_{label}" if label else display


def register_display_names() -> None:
    """Merge this family into gft.DISPLAY_NAMES, which is the single lookup
    both gpt.collect_performance_rows and gmc.build_rows use to turn a model
    key into a row label. Without it every budget would render as the
    normalized key upper-cased ("PARETOB100M")."""
    gft.DISPLAY_NAMES = {**gft.DISPLAY_NAMES, **DISPLAY_NAMES}


# The only evaluation this family has: BraTS25-pre (dsnum 23, official). It
# was never tested on BraTS18 or on the internal MB-96 cohort, so this is the
# one gst.GROUPS entry other families must be restricted to for a fair
# side-by-side Dice comparison.
GROUPS = [("BraTS25-pre", "official", DATASET)]


def build_rows(
    *, groups: list[tuple[str, str, str]] | None = None
) -> list[tuple[str, dict[str, float | None]]]:
    """Same shape as gmc.build_rows: [(display, metrics), ...], one row per
    budget that has data on disk. `groups` is accepted (and forwarded) purely
    so callers can restrict every family uniformly; any group other than
    BraTS25-pre simply has no data here and contributes nothing."""
    register_display_names()
    return gmc.build_rows(
        MODEL_SET,
        gpu_filename=GPU_FILENAME,
        expected_folds=EXPECTED_FOLDS,
        groups=groups,
    )


if __name__ == "__main__":
    for display, metrics in build_rows():
        print(display, metrics)
