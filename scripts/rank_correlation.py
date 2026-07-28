#!/usr/bin/env python3
"""Rank correlation between BraTS2018 and BraTS2023 model rankings.

Turns "we ran the models twice" into a concrete reliability finding: for
every model with COMPLETE official-test-split results on both BraTS2018 and
BraTS2023 (see generate_summary_tables.is_complete), compute the Spearman
and Kendall rank correlation between its BraTS2018 score and its BraTS2023
score, per tumor region (WT/TC/ET) and metric (Dice, HD95). A high
correlation supports the claim that BraTS2018-era conclusions about
relative model performance still hold at BraTS2023 scale.

Only the official test-split scores are used (not the internal cohort).
Reuses generate_summary_tables.py's discovery/completeness/aggregation
pipeline rather than re-deriving it -- see official_stats_by_model below for
why generate_summary_tables.build_rows itself isn't reused directly.

Writes a LaTeX table plus a standalone main.tex into OUTPUT_DIR, compiling
it to PDF with tectonic (if available), and prints the same data as rich
console tables.
"""

from __future__ import annotations

import os
import statistics
import sys

from rich.console import Console
from rich.table import Table
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_reproduction_comparison as brc  # noqa: E402
import generate_summary_tables as gst  # noqa: E402  (reuse discover/is_complete/aggregate)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "rank_correlation"
)


def official_stats_by_model(
    models: dict, dsnum: str, display_names: dict
) -> dict[str, tuple[str, dict]]:
    """{model_norm: (display_name, stats)} for models with COMPLETE OFFICIAL-
    split results for this dsnum (internal-cohort completeness not required,
    unlike generate_summary_tables.build_rows -- that function requires both
    official AND internal completeness, which would wrongly shrink this
    official-split-only sample, and keys its output by display name rather
    than model_norm, which is the wrong join key across datasets)."""
    out = {}
    for model_norm, per_dsnum in models.items():
        if model_norm in gst.EXCLUDED_MODELS:
            continue
        folds = per_dsnum.get(dsnum)
        if not folds or not gst.is_complete(folds):
            continue
        display = display_names.get(model_norm) or gst.FALLBACK_DISPLAY_NAMES.get(
            model_norm, model_norm.upper()
        )
        out[model_norm] = (display, gst.aggregate(folds))
    return out


def build_common_set(
    stats_a: dict[str, tuple[str, dict]], stats_b: dict[str, tuple[str, dict]]
) -> tuple[list[str], list[str], list[str]]:
    """(common model_norms sorted by display name, only-in-a display names,
    only-in-b display names)."""
    common = sorted(set(stats_a) & set(stats_b), key=lambda m: stats_a[m][0].lower())
    only_a = sorted(stats_a[m][0] for m in set(stats_a) - set(stats_b))
    only_b = sorted(stats_b[m][0] for m in set(stats_b) - set(stats_a))
    return common, only_a, only_b


def _ranks_within(
    stats: dict[str, tuple[str, dict]], common: list[str], region: str, metric: str
) -> dict[str, int]:
    """{model_norm: rank}, rank 1 = best (max for Dice, min for HD95), computed
    over just the `common` models so BraTS2018 and BraTS2023 ranks are
    directly comparable (same denominator, same model set) -- matches the
    best-is-rank-1 convention used by generate_summary_tables.dice_official_ranks."""
    best_is_highest = metric == "Dice"
    order = sorted(
        common, key=lambda m: stats[m][1][region][metric][0], reverse=best_is_highest
    )
    return {model_norm: rank for rank, model_norm in enumerate(order, start=1)}


def median_abs_rank_change(
    stats_18: dict[str, tuple[str, dict]],
    stats_23: dict[str, tuple[str, dict]],
    common: list[str],
    region: str,
    metric: str,
) -> float:
    """Median, across the common models, of |rank on BraTS2018 - rank on
    BraTS2023| -- a directly interpretable companion to Spearman/Kendall:
    "half the models moved by at most N leaderboard positions"."""
    ranks_18 = _ranks_within(stats_18, common, region, metric)
    ranks_23 = _ranks_within(stats_23, common, region, metric)
    diffs = [abs(ranks_18[m] - ranks_23[m]) for m in common]
    return statistics.median(diffs)


def compute_correlations(
    stats_18: dict[str, tuple[str, dict]],
    stats_23: dict[str, tuple[str, dict]],
    common: list[str],
) -> dict[tuple[str, str], dict]:
    """{(region, metric): {"n", "spearman": (rho, p), "kendall": (tau, p),
    "median_abs_rank_change"}}."""
    results = {}
    for region in gst.REGIONS:
        for metric in gst.METRICS:
            vals_18 = [stats_18[m][1][region][metric][0] for m in common]
            vals_23 = [stats_23[m][1][region][metric][0] for m in common]
            rho, p_rho = spearmanr(vals_18, vals_23)
            tau, p_tau = kendalltau(vals_18, vals_23)
            results[(region, metric)] = {
                "n": len(common),
                "spearman": (rho, p_rho),
                "kendall": (tau, p_tau),
                "median_abs_rank_change": median_abs_rank_change(
                    stats_18, stats_23, common, region, metric
                ),
            }
    return results


def print_model_set_report(common: list[str], only_18: list[str], only_23: list[str]) -> None:
    console = Console()
    console.print(
        f"[bold]Common models (official split, both datasets):[/bold] {len(common)}"
    )
    console.print(", ".join(common))
    if only_18:
        console.print(f"[yellow]Only complete on BraTS2018:[/yellow] {', '.join(only_18)}")
    if only_23:
        console.print(f"[yellow]Only complete on BraTS2023:[/yellow] {', '.join(only_23)}")


def print_correlation_table(results: dict[tuple[str, str], dict]) -> None:
    table = Table(title="BraTS2018 vs BraTS2023 rank correlation (official test split)")
    table.add_column("Region", style="bold cyan")
    table.add_column("Metric")
    table.add_column("n", justify="right")
    table.add_column("Spearman ρ", justify="right")
    table.add_column("Kendall τ", justify="right")
    table.add_column("Median |Δrank|", justify="right")
    for region in gst.REGIONS:
        for metric in gst.METRICS:
            r = results[(region, metric)]
            rho, _p_rho = r["spearman"]
            tau, _p_tau = r["kendall"]
            table.add_row(
                region,
                metric,
                str(r["n"]),
                f"{rho:.3f}",
                f"{tau:.3f}",
                f"{r['median_abs_rank_change']:.1f}",
            )
    Console().print(table)


def build_correlation_table(results: dict[tuple[str, str], dict], *, label: str = "tab:rank_corr") -> str:
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
        r"absolute rank change, between per-model BraTS 2018 and BraTS 2023 "
        r"official-test-split scores, per tumor region and metric "
        r"($n=" + str(n) + r"$ models with complete official-split results "
        r"on both datasets). Median $|\Delta\text{rank}|$ is the median, "
        r"across those models, of the absolute difference between a "
        r"model's leaderboard rank on BraTS 2018 and on BraTS 2023 (rank 1 "
        r"= best), computed over the same $n$ models on both datasets.}",
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

    models, _internal_models, dsnums = gst.discover(gst.RESULTS_DIR)
    if "18" not in dsnums or "23" not in dsnums:
        print(
            f"Need both dsnum='18' and dsnum='23' under {gst.RESULTS_DIR}; found {dsnums}",
            file=sys.stderr,
        )
        return

    stats_18 = official_stats_by_model(models, "18", display_names)
    stats_23 = official_stats_by_model(models, "23", display_names)
    common, only_18, only_23 = build_common_set(stats_18, stats_23)
    if not common:
        print(
            "No models complete on the official split for both BraTS2018 and 2023",
            file=sys.stderr,
        )
        return

    print_model_set_report(common, only_18, only_23)
    results = compute_correlations(stats_18, stats_23, common)
    print_correlation_table(results)

    tables_path = os.path.join(OUTPUT_DIR, "tables.tex")
    with open(tables_path, "w") as f:
        f.write(build_correlation_table(results))

    main_tex = r"""\documentclass[12pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{lmodern}
\pagestyle{plain}
\begin{document}
\input{tables.tex}
\end{document}
"""
    main_path = os.path.join(OUTPUT_DIR, "main.tex")
    with open(main_path, "w") as f:
        f.write(main_tex)

    print(f"Wrote {tables_path}")
    print(f"Wrote {main_path}")
    gst.compile_pdf(main_path)


if __name__ == "__main__":
    main()
