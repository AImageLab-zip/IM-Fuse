#!/usr/bin/env python3
"""Single table combining the official-split BraTS2018-vs-BraTS2023 rank
correlation (rank_correlation.py) and the internal-cohort BraTS18-chp-vs-
BraTS25-pre-chp rank correlation (rank_correlation_internal.py), one block
per comparison, separated by a double rule.

Both blocks use the same per-region/metric layout (Region, Metric, Spearman
rho, Kendall tau, median |delta rank|); each model set/count differs between
the two comparisons (see the printed model-set report), so "n" is called out
per block instead of as a table column.

Reuses rank_correlation.py's stats-building and per-region/metric
correlation helpers (official_stats_by_model, build_common_set,
compute_correlations) for both blocks -- pointed at `models` for the BraTS
block and `internal_models` for the internal block -- rather than
re-deriving any of that pipeline.

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
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "rank_correlation_combined"
)

BRATS_LABEL = "BraTS2018 vs BraTS2023 (official test split)"
INTERNAL_LABEL = "MB-96 internal cohort: BraTS18 chp vs BraTS25-pre chp"


def build_block(results: dict[tuple[str, str], dict]) -> list[tuple[str, str, float, float, float]]:
    """[(region, metric, spearman_rho, kendall_tau, median_abs_rank_change), ...]
    in gst.REGIONS x gst.METRICS order, dropping each cell's n and p-values --
    those live in the per-block console report / caption instead."""
    rows = []
    for region in gst.REGIONS:
        for metric in gst.METRICS:
            r = results[(region, metric)]
            rho, _p_rho = r["spearman"]
            tau, _p_tau = r["kendall"]
            rows.append((region, metric, rho, tau, r["median_abs_rank_change"]))
    return rows


def print_combined_table(
    blocks: list[tuple[str, int, list[tuple[str, str, float, float, float]]]],
) -> None:
    """blocks: [(section_label, n, rows), ...] -- prints one rich table with a
    bold section-header row (spanning n in its own column text) before each
    block's region/metric rows."""
    table = Table(title="BraTS official-split vs MB-96 internal-cohort rank correlation")
    table.add_column("Region")
    table.add_column("Metric")
    table.add_column("Spearman ρ", justify="right")
    table.add_column("Kendall τ", justify="right")
    table.add_column("Median |Δrank|", justify="right")
    for i, (section_label, n, rows) in enumerate(blocks):
        if i > 0:
            table.add_section()
        table.add_row(f"[bold cyan]{section_label} (n={n})[/bold cyan]", "", "", "", "")
        for region, metric, rho, tau, med in rows:
            table.add_row(region, metric, f"{rho:.3f}", f"{tau:.3f}", f"{med:.1f}")
    Console().print(table)


def build_combined_latex_table(
    blocks: list[tuple[str, int, list[tuple[str, str, float, float, float]]]],
    *,
    label: str = "tab:rank_corr_combined",
) -> str:
    body = []
    for i, (section_label, n, rows) in enumerate(blocks):
        if i > 0:
            # Double rule between blocks, as requested.
            body.append(r"\midrule")
            body.append(r"\midrule")
        body.append(f"\\multicolumn{{5}}{{l}}{{\\textbf{{{section_label} (n={n})}}}} \\\\")
        body.append(r"\midrule")
        for region, metric, rho, tau, med in rows:
            body.append(f"{region} & {metric} & {rho:.3f} & {tau:.3f} & {med:.1f} \\\\")

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Spearman and Kendall rank correlation, plus median "
        r"absolute rank change, per tumor region and metric, for two "
        r"reliability comparisons: the official-test-split BraTS2018-vs-"
        r"BraTS2023 model ranking (top), and the MB-96 internal-cohort "
        r"ranking under the BraTS18-trained checkpoint vs. the "
        r"BraTS25-pre-trained checkpoint (bottom). Each block's $n$ is the "
        r"number of models with complete data for that comparison; the two "
        r"blocks are independent model sets.}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Region} & \textbf{Metric} & \textbf{Spearman $\rho$} & "
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

    # BraTS official-split block.
    brats_stats_18 = rc.official_stats_by_model(models, "18", display_names)
    brats_stats_23 = rc.official_stats_by_model(models, "23", display_names)
    brats_common, brats_only_18, brats_only_23 = rc.build_common_set(brats_stats_18, brats_stats_23)
    if not brats_common:
        print("No models complete on the official split for both datasets", file=sys.stderr)
        return
    print_model_set_report(brats_common, brats_only_18, brats_only_23)
    brats_results = rc.compute_correlations(brats_stats_18, brats_stats_23, brats_common)

    # MB-96 internal-cohort block.
    internal_stats_18 = rc.official_stats_by_model(internal_models, "18", display_names)
    internal_stats_23 = rc.official_stats_by_model(internal_models, "23", display_names)
    internal_common, internal_only_18, internal_only_23 = rc.build_common_set(
        internal_stats_18, internal_stats_23
    )
    if not internal_common:
        print("No models complete on the internal cohort under both checkpoints", file=sys.stderr)
        return
    print_model_set_report(internal_common, internal_only_18, internal_only_23)
    internal_results = rc.compute_correlations(internal_stats_18, internal_stats_23, internal_common)

    blocks = [
        (BRATS_LABEL, len(brats_common), build_block(brats_results)),
        (INTERNAL_LABEL, len(internal_common), build_block(internal_results)),
    ]

    print_combined_table(blocks)

    tables_path = os.path.join(OUTPUT_DIR, "tables.tex")
    with open(tables_path, "w") as f:
        f.write(build_combined_latex_table(blocks))

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
