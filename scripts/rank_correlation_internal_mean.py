#!/usr/bin/env python3
"""Rank correlation between BraTS2018-checkpoint and BraTS2025-pre-checkpoint
model rankings on the internal (MB-96) cohort, ranking by Dice AVERAGED
ACROSS REGIONS (WT/TC/ET) rather than one region at a time.

Same data and model-set as rank_correlation_internal.py, but collapsed the
way rank_correlation.py's Legacy-vs-reproduced "mean" table collapses
regions: each model's Dice, under each checkpoint, is first averaged over
WT/TC/ET, then that single number drives the rank/correlation -- "does the
internal-cohort ranking still hold when checkpoint scores are pooled across
regions", as opposed to per-region agreement.

Reuses rank_correlation.py's stats-building (official_stats_by_model,
pointed at internal_models instead of models) and mean-across-regions
correlation/table helpers directly -- no new pipeline, just a different
stats source than that script's own Legacy-vs-BraTS comparisons.

Writes a LaTeX table plus a standalone main.tex into OUTPUT_DIR, compiling
it to PDF with tectonic (if available), and prints the same data as a rich
console table.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rich.console import Console
from rich.table import Table

import build_reproduction_comparison as brc  # noqa: E402
import generate_summary_tables as gst  # noqa: E402  (reuse discover/is_complete/aggregate)
import rank_correlation as rc  # noqa: E402  (reuse stats-building/correlation helpers)
from rank_correlation_internal import print_model_set_report  # noqa: E402

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "rank_correlation_internal_mean"
)

COMPARISON_LABEL = "MB-96: BraTS18 chp vs BraTS25-pre chp"


def print_correlation_table_mean(results_by_comparison: dict[str, dict]) -> None:
    """Like rc.print_legacy_correlation_table_mean, but labeled for the
    internal cohort's two checkpoints rather than the Legacy-vs-reproduced
    official-test-split comparison that function's title assumes."""
    table = Table(
        title="MB-96 internal cohort: BraTS18 chp vs BraTS25-pre chp rank "
        "correlation (Dice averaged over WT/TC/ET)"
    )
    table.add_column("Comparison", style="bold cyan")
    table.add_column("n", justify="right")
    table.add_column("Spearman ρ", justify="right")
    table.add_column("Kendall τ", justify="right")
    table.add_column("Median |Δrank|", justify="right")
    table.add_column("Mean±Std Diff (23chp-18chp)", justify="right")
    for comparison, r in results_by_comparison.items():
        table.add_row(
            comparison,
            str(r["n"]),
            f"{r['spearman']:.3f}",
            f"{r['kendall']:.3f}",
            f"{r['median_abs_rank_change']:.1f}",
            f"{r['mean_diff']:+.2f}±{r['std_diff']:.2f}",
        )
    Console().print(table)


def build_correlation_table_mean(
    results_by_comparison: dict[str, dict], *, label: str = "tab:rank_corr_internal_mean"
) -> str:
    """Like rc.build_legacy_correlation_table_mean, but captioned for the
    internal cohort's two checkpoints instead of Legacy-vs-reproduced."""
    body = []
    for comparison, r in results_by_comparison.items():
        body.append(
            f"{comparison} & {r['spearman']:.3f} & "
            f"{r['kendall']:.3f} & {r['median_abs_rank_change']:.1f} & "
            f"{r['mean_diff']:+.2f}$\\pm${r['std_diff']:.2f} \\\\"
        )
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Spearman and Kendall rank correlation, median absolute "
        r"rank change, and mean$\pm$std signed Dice difference "
        r"(BraTS25-pre-chp minus BraTS18-chp, positive = the "
        r"BraTS25-pre-trained checkpoint reads higher; std is the "
        r"population std of that per-model difference across common "
        r"models), between per-model MB-96 internal-cohort scores under "
        r"the BraTS18-trained checkpoint and under the BraTS25-pre-trained "
        r"checkpoint, with each model's Dice averaged over WT/TC/ET rather "
        r"than compared per region. Confirms whether the internal-cohort "
        r"ranking is preserved across checkpoints, and whether one "
        r"checkpoint reads systematically higher or lower.}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Comparison} & \textbf{Spearman $\rho$} & "
        r"\textbf{Kendall $\tau$} & \textbf{Median $|\Delta\text{rank}|$} & "
        r"\textbf{Mean$\pm$Std Diff} \\",
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

    _models, internal_models, dsnums = gst.discover(gst.RESULTS_DIR)
    if "18" not in dsnums or "23" not in dsnums:
        print(
            f"Need both dsnum='18' and dsnum='23' under {gst.RESULTS_DIR}; found {dsnums}",
            file=sys.stderr,
        )
        return

    stats_18 = rc.official_stats_by_model(internal_models, "18", display_names)
    stats_23 = rc.official_stats_by_model(internal_models, "23", display_names)
    common, only_18, only_23 = rc.build_common_set(stats_18, stats_23)
    if not common:
        print(
            "No models complete on the internal cohort under both checkpoints",
            file=sys.stderr,
        )
        return

    print_model_set_report(common, only_18, only_23)
    results_by_comparison = {
        COMPARISON_LABEL: rc.compute_legacy_correlations_mean(stats_18, stats_23, common)
    }

    print_correlation_table_mean(results_by_comparison)

    tables_path = os.path.join(OUTPUT_DIR, "tables.tex")
    with open(tables_path, "w") as f:
        f.write(
            build_correlation_table_mean(
                results_by_comparison, label="tab:rank_corr_internal_mean"
            )
        )

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
