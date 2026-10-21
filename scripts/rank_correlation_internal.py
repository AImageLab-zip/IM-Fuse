#!/usr/bin/env python3
"""Rank correlation between BraTS2018-checkpoint and BraTS2025-pre-checkpoint
model rankings on the INTERNAL (MB-96) cohort.

Same idea as rank_correlation.py's official-split BraTS2018-vs-BraTS2023
comparison, but over the internal cohort's two scoring runs instead: for
every model with COMPLETE internal-cohort results scored with both the
BraTS18-trained checkpoint (results_internal_fold*.xlsx under the `_18`
results dir) and the BraTS25-pre-trained checkpoint (same, under `_23`),
compute the Spearman and Kendall rank correlation between its two scores,
per tumor region (WT/TC/ET) and metric (Dice, HD95). A high correlation
supports the claim that which checkpoint scored the internal cohort doesn't
change the internal-cohort model ranking.

Reuses rank_correlation.py's stats-building/correlation helpers (they only
take a generic {model_norm: {dsnum: folds}} dict, so pointing them at
generate_summary_tables.discover's internal_models return value instead of
its models return value is enough -- no new pipeline needed) and
generate_summary_tables.py's discover/is_complete/aggregate underneath that.

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

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "rank_correlation_internal"
)


def print_model_set_report(common: list[str], only_18: list[str], only_23: list[str]) -> None:
    """Like rank_correlation.print_model_set_report, but labeled for the
    internal cohort's two checkpoints rather than the official-split
    BraTS2018/BraTS2023 comparison that function's wording assumes."""
    console = Console()
    console.print(
        f"[bold]Common models (internal cohort, both checkpoints):[/bold] {len(common)}"
    )
    console.print(", ".join(common))
    if only_18:
        console.print(
            f"[yellow]Only complete under the BraTS18-trained checkpoint:[/yellow] "
            f"{', '.join(only_18)}"
        )
    if only_23:
        console.print(
            f"[yellow]Only complete under the BraTS25-pre-trained checkpoint:[/yellow] "
            f"{', '.join(only_23)}"
        )


def print_correlation_table(results: dict[tuple[str, str], dict]) -> None:
    table = Table(
        title="MB-96 internal cohort: BraTS18 chp vs BraTS25-pre chp rank correlation"
    )
    table.add_column("Region", style="bold cyan")
    table.add_column("Metric")
    table.add_column("Spearman ρ", justify="right")
    table.add_column("Kendall τ", justify="right")
    table.add_column("|Δrank|", justify="right")
    for region in gst.REGIONS:
        for metric in gst.METRICS:
            r = results[(region, metric)]
            rho, _p_rho = r["spearman"]
            tau, _p_tau = r["kendall"]
            table.add_row(
                region,
                metric,
                f"{rho:.3f}",
                f"{tau:.3f}",
                f"{r['median_abs_rank_change']:.1f}",
            )
    Console().print(table)


def build_correlation_table(
    results: dict[tuple[str, str], dict], *, label: str = "tab:rank_corr_internal"
) -> str:
    body = []
    for region in gst.REGIONS:
        for metric in gst.METRICS:
            r = results[(region, metric)]
            rho, _p_rho = r["spearman"]
            tau, _p_tau = r["kendall"]
            body.append(
                f"{region} & {metric} & {rho:.3f} & "
                f"{tau:.3f} & {r['median_abs_rank_change']:.1f} \\\\"
            )
    n = next(iter(results.values()))["n"]
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Spearman and Kendall rank correlation, plus median "
        r"absolute rank change, between per-model MB-96 internal-cohort "
        r"scores under the BraTS18-trained checkpoint and under the "
        r"BraTS25-pre-trained checkpoint, per tumor region and metric "
        r"($n=" + str(n) + r"$ models with complete internal-cohort results "
        r"under both checkpoints). Median $|\Delta\text{rank}|$ is the "
        r"median, across those models, of the absolute difference between "
        r"a model's leaderboard rank under each checkpoint (rank 1 = "
        r"best), computed over the same $n$ models under both.}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Region} & \textbf{Metric} & \textbf{Spearman $\rho$} & "
        r"\textbf{Kendall $\tau$} & \textbf{$|\Delta\text{rank}|$} \\",
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
    results = rc.compute_correlations(stats_18, stats_23, common)
    print_correlation_table(results)

    tables_path = os.path.join(OUTPUT_DIR, "tables.tex")
    with open(tables_path, "w") as f:
        f.write(build_correlation_table(results))

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
