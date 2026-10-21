#!/usr/bin/env python3
"""Combined rank-correlation table (like rank_correlation_combined.py) but
with each model's Dice (or HD95) averaged across WT/TC/ET before
ranking/correlating, instead of one region at a time.

One row per comparison -- BraTS2018-vs-BraTS2023 official test split, and
MB-96 internal-cohort BraTS18-chp-vs-BraTS25-pre-chp -- each computed with
rank_correlation.py's mean-across-regions helper
(compute_legacy_correlations_mean), which already collapses WT/TC/ET into a
single per-model number before ranking. Model sets/counts differ between the
two comparisons (see the printed model-set report), so each row carries its
own n.

Dice and HD95 are computed independently and rendered as separate column
groups within a single table -- never averaged or blended into one number,
since Dice (higher-is-better) and HD95 (lower-is-better, different units)
aren't comparable, but both fit side by side as one row per comparison.

Reuses rank_correlation.py's stats-building and mean-across-regions
correlation helpers (official_stats_by_model, build_common_set,
compute_legacy_correlations_mean) for both rows -- pointed at `models` for
the BraTS row and `internal_models` for the internal row -- rather than
re-deriving any of that pipeline. Mirrors rank_correlation_combined.py's
two-comparison structure and rank_correlation_internal_mean.py's
mean-across-regions table shape.

Writes a LaTeX table plus a standalone main.tex into OUTPUT_DIR, compiling
it to PDF with tectonic (if available), and prints the same data as a rich
console table.
"""

from __future__ import annotations

import os
import sys

from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_reproduction_comparison as brc  # noqa: E402
import generate_summary_tables as gst  # noqa: E402  (reuse discover/is_complete/aggregate)
import rank_correlation as rc  # noqa: E402  (reuse stats-building/correlation helpers)
from rank_correlation_internal import print_model_set_report  # noqa: E402

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "rank_correlation_combined_mean"
)

BRATS_LABEL = "BraTS2018 vs BraTS2023 (official test split)"
INTERNAL_LABEL = "MB-96 internal cohort: BraTS18 chp vs BraTS25-pre chp"


def print_combined_table_mean(
    results_by_comparison: dict[str, dict[str, dict]],
) -> None:
    """Like rc.print_legacy_correlation_table_mean, but titled/captioned for
    this script's two reliability comparisons (BraTS official split,
    MB-96 internal cohort), and with one row per (comparison, metric) pair
    -- a Dice row and an HD95 row per comparison -- rather than the
    Legacy-vs-reproduced comparison that function's wording assumes."""
    table = Table(
        title="BraTS official-split vs MB-96 internal-cohort rank correlation "
        "(Dice / HD95 averaged over WT/TC/ET)"
    )
    table.add_column("Comparison", style="bold cyan")
    table.add_column("Metric")
    table.add_column("Spearman ρ", justify="right")
    table.add_column("Kendall τ", justify="right")
    table.add_column("Median |Δrank|", justify="right")
    for comparison, by_metric in results_by_comparison.items():
        for metric in gst.METRICS:
            r = by_metric[metric]
            table.add_row(
                comparison,
                metric,
                f"{r['spearman']:.3f}",
                f"{r['kendall']:.3f}",
                f"{r['median_abs_rank_change']:.1f}",
            )
    Console().print(table)


def build_combined_latex_table_mean(
    results_by_comparison: dict[str, dict[str, dict]],
    *,
    label: str = "tab:rank_corr_combined_mean",
) -> str:
    body = []
    for comparison, by_metric in results_by_comparison.items():
        for metric in gst.METRICS:
            r = by_metric[metric]
            body.append(
                f"{comparison} & {metric} & {r['spearman']:.3f} & "
                f"{r['kendall']:.3f} & {r['median_abs_rank_change']:.1f} \\\\"
            )
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Spearman and Kendall rank correlation and median "
        r"absolute rank change, for two reliability comparisons: the "
        r"official-test-split BraTS2018-vs-BraTS2023 model ranking, and the "
        r"MB-96 internal-cohort ranking under the BraTS18-trained "
        r"checkpoint vs. the BraTS25-pre-trained checkpoint. Each model's "
        r"Dice and HD95 are separately averaged over WT/TC/ET before "
        r"ranking/correlating (rather than compared per region), and are "
        r"never blended into one number -- each comparison has its own Dice "
        r"row and HD95 row, computed independently. The two comparisons are "
        r"independent model sets.}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Comparison} & \textbf{Metric} & \textbf{Spearman $\rho$} & "
        r"\textbf{Kendall $\tau$} & \textbf{Median $|\Delta\text{rank}|$} \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    display_names = {}
    if os.path.exists(gst.LEGACY_FILE):
        _, display_names = brc.parse_legacy(gst.LEGACY_FILE)

    models, internal_models, dsnums = gst.discover(gst.RESULTS_DIR)
    if "18" not in dsnums or "23" not in dsnums:
        print(
            f"Need both dsnum='18' and dsnum='23' under {gst.RESULTS_DIR}; found {dsnums}",
            file=sys.stderr,
        )
        return

    brats_stats_18 = rc.official_stats_by_model(models, "18", display_names)
    brats_stats_23 = rc.official_stats_by_model(models, "23", display_names)
    brats_common, brats_only_18, brats_only_23 = rc.build_common_set(brats_stats_18, brats_stats_23)
    if not brats_common:
        print("No models complete on the official split for both datasets", file=sys.stderr)
        return
    print_model_set_report(brats_common, brats_only_18, brats_only_23)

    internal_stats_18 = rc.official_stats_by_model(internal_models, "18", display_names)
    internal_stats_23 = rc.official_stats_by_model(internal_models, "23", display_names)
    internal_common, internal_only_18, internal_only_23 = rc.build_common_set(
        internal_stats_18, internal_stats_23
    )
    if not internal_common:
        print("No models complete on the internal cohort under both checkpoints", file=sys.stderr)
        return
    print_model_set_report(internal_common, internal_only_18, internal_only_23)

    results_by_comparison = {
        BRATS_LABEL: {
            metric: rc.compute_legacy_correlations_mean(
                brats_stats_18, brats_stats_23, brats_common, metric
            )
            for metric in gst.METRICS
        },
        INTERNAL_LABEL: {
            metric: rc.compute_legacy_correlations_mean(
                internal_stats_18, internal_stats_23, internal_common, metric
            )
            for metric in gst.METRICS
        },
    }

    print_combined_table_mean(results_by_comparison)

    tables_path = os.path.join(OUTPUT_DIR, "tables.tex")
    with open(tables_path, "w") as f:
        f.write(build_combined_latex_table_mean(results_by_comparison))

    main_tex = (
        "\\documentclass[12pt]{article}\n"
        "\\usepackage[margin=0.8in]{geometry}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{lmodern}\n"
        "\\pagestyle{plain}\n"
        "\\begin{document}\n"
        "\\input{tables.tex}\n"
        "\\end{document}\n"
    )
    main_path = os.path.join(OUTPUT_DIR, "main.tex")
    with open(main_path, "w") as f:
        f.write(main_tex)

    print(f"Wrote {tables_path}")
    print(f"Wrote {main_path}")
    gst.compile_pdf(main_path)


if __name__ == "__main__":
    main()
