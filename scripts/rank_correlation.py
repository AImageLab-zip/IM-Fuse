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

# Excluded from every comparison in this script (on top of gst.EXCLUDED_MODELS):
# these four have confirmed reproduction issues independent of dataset-scale or
# legacy-fidelity questions (RFNet: 20x learning-rate mismatch vs. the legacy
# training script; RobustSeg: 12x batch-size/LR scale-up vs. legacy; MIFPN:
# smaller but similar scale-up; U-HVED: InstanceNorm affine-weight
# initialization bug), so including them would confound the correlation
# numbers with those known, unrelated bugs rather than measuring what this
# script is actually meant to measure.
#EXTRA_EXCLUDED_MODELS = {"uhved", "rfnet", "robustseg", "mifpn"}
EXTRA_EXCLUDED_MODELS={}

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
        if model_norm in gst.EXCLUDED_MODELS or model_norm in EXTRA_EXCLUDED_MODELS:
            continue
        folds = per_dsnum.get(dsnum)
        if not folds or not gst.is_complete(folds):
            continue
        display = display_names.get(model_norm) or gst.FALLBACK_DISPLAY_NAMES.get(
            model_norm, model_norm.upper()
        )
        out[model_norm] = (display, gst.aggregate(folds))
    return out


def build_single_fold_stats(
    models: dict, dsnum: str, fold_label: str, display_names: dict
) -> dict[str, tuple[str, dict]]:
    """{model_norm: (display_name, {region: {"Dice": (mean, std), "HD95": (mean, std)}})}
    using only `fold_label`'s (e.g. "fold1") per-modality-combo values --
    unlike official_stats_by_model, which averages across all of
    gst.EXPECTED_FOLDS via gst.aggregate. gst.aggregate can't be reused here
    since it hardcodes iterating EXPECTED_FOLDS and would KeyError on a folds
    dict containing just one fold. "std" here is dispersion across the 15
    modality combinations (no repeated folds to vary over), same convention
    as build_legacy_stats. A model is included if this one fold has every
    region/metric populated across all 15 modality-presence combinations."""
    out = {}
    for model_norm, per_dsnum in models.items():
        if model_norm in gst.EXCLUDED_MODELS or model_norm in EXTRA_EXCLUDED_MODELS:
            continue
        fold_data = per_dsnum.get(dsnum, {}).get(fold_label)
        if not fold_data:
            continue
        complete = all(
            fold_data.get(mkey, {}).get(region) is not None
            and fold_data.get(mkey, {}).get(f"{region}_hd95") is not None
            for mkey in gst.MODALITY_KEY_ORDER
            for region in gst.REGIONS
        )
        if not complete:
            continue

        stats = {}
        for region in gst.REGIONS:
            stats[region] = {}
            for metric, suffix in (("Dice", ""), ("HD95", "_hd95")):
                key = region + suffix
                vals = [fold_data[mkey][key] for mkey in gst.MODALITY_KEY_ORDER]
                std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                stats[region][metric] = (statistics.fmean(vals), std)

        display = display_names.get(model_norm) or gst.FALLBACK_DISPLAY_NAMES.get(
            model_norm, model_norm.upper()
        )
        out[model_norm] = (display, stats)
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


def build_legacy_stats(
    legacy: dict, ds_label: str, display_names: dict
) -> dict[str, tuple[str, dict]]:
    """{model_norm: (display_name, {region: {"Dice": (mean, std)}})} from the
    published legacy sheet (build_reproduction_comparison.parse_legacy),
    averaged over whichever of the 15 modality combinations have a value.
    Legacy only has Dice (no HD95, see that module's docstring), and is a
    single reported number per combination rather than repeated folds, so
    the "std" here is dispersion across modality combinations, not folds --
    still shaped like generate_summary_tables.aggregate's output so the same
    rank/correlation helpers below can be reused unchanged. A model is only
    included if it has at least one value for every region in gst.REGIONS,
    matching official_stats_by_model's all-regions-present completeness bar."""
    per_model_region: dict[str, dict[str, list[float]]] = {}
    for region, by_mkey in legacy.get(ds_label, {}).items():
        for by_model in by_mkey.values():
            for model_norm, value in by_model.items():
                if value is None:
                    continue
                per_model_region.setdefault(model_norm, {}).setdefault(region, []).append(value)

    out = {}
    for model_norm, region_vals in per_model_region.items():
        if model_norm in gst.EXCLUDED_MODELS or model_norm in EXTRA_EXCLUDED_MODELS:
            continue
        stats = {}
        for region in gst.REGIONS:
            vals = region_vals.get(region, [])
            if not vals:
                break
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            stats[region] = {"Dice": (statistics.fmean(vals), std)}
        else:
            display = display_names.get(model_norm) or gst.FALLBACK_DISPLAY_NAMES.get(
                model_norm, model_norm.upper()
            )
            out[model_norm] = (display, stats)
    return out


def _signed_diffs(
    legacy_stats: dict[str, tuple[str, dict]],
    repro_stats: dict[str, tuple[str, dict]],
    common: list[str],
    region: str,
) -> list[float]:
    """[(reproduced Dice - legacy Dice), ...] for this region, one per common
    model -- shared by mean_signed_diff and std_signed_diff so both read off
    the same underlying sample."""
    return [
        repro_stats[m][1][region]["Dice"][0] - legacy_stats[m][1][region]["Dice"][0]
        for m in common
    ]


def mean_signed_diff(
    legacy_stats: dict[str, tuple[str, dict]],
    repro_stats: dict[str, tuple[str, dict]],
    common: list[str],
    region: str,
) -> float:
    """Mean, across the common models, of (reproduced Dice - legacy Dice) for
    this region -- signed, so it shows systematic bias (reproduction reading
    higher or lower than legacy on average), unlike median_abs_rank_change
    which only measures leaderboard-position churn."""
    return statistics.fmean(_signed_diffs(legacy_stats, repro_stats, common, region))


def std_signed_diff(
    legacy_stats: dict[str, tuple[str, dict]],
    repro_stats: dict[str, tuple[str, dict]],
    common: list[str],
    region: str,
) -> float:
    """Population std, across the common models, of (reproduced Dice - legacy
    Dice) for this region -- how much the per-model bias varies around
    mean_signed_diff, not just its average."""
    diffs = _signed_diffs(legacy_stats, repro_stats, common, region)
    return statistics.pstdev(diffs) if len(diffs) > 1 else 0.0


def compute_legacy_correlations(
    legacy_stats: dict[str, tuple[str, dict]],
    repro_stats: dict[str, tuple[str, dict]],
    common: list[str],
) -> dict[str, dict]:
    """Like compute_correlations, restricted to Dice (legacy has no HD95).
    {region: {"n", "spearman": rho, "kendall": tau, "median_abs_rank_change",
    "mean_diff", "std_diff"}}."""
    results = {}
    for region in gst.REGIONS:
        vals_legacy = [legacy_stats[m][1][region]["Dice"][0] for m in common]
        vals_repro = [repro_stats[m][1][region]["Dice"][0] for m in common]
        rho, _p_rho = spearmanr(vals_legacy, vals_repro)
        tau, _p_tau = kendalltau(vals_legacy, vals_repro)
        results[region] = {
            "n": len(common),
            "spearman": rho,
            "kendall": tau,
            "median_abs_rank_change": median_abs_rank_change(
                legacy_stats, repro_stats, common, region, "Dice"
            ),
            "mean_diff": mean_signed_diff(legacy_stats, repro_stats, common, region),
            "std_diff": std_signed_diff(legacy_stats, repro_stats, common, region),
        }
    return results


def _mean_metric(
    stats: dict[str, tuple[str, dict]], model_norm: str, metric: str = "Dice"
) -> float:
    """A model's Dice (or HD95) averaged across gst.REGIONS -- the
    single-number summary used by the mean-of-3-classes ranking (as opposed
    to ranking each region separately)."""
    return statistics.fmean(stats[model_norm][1][region][metric][0] for region in gst.REGIONS)


def _ranks_within_mean(
    stats: dict[str, tuple[str, dict]], common: list[str], metric: str = "Dice"
) -> dict[str, int]:
    """Like _ranks_within, but rank 1 = best mean-across-regions Dice/HD95
    (higher is better for Dice, lower for HD95) rather than a single
    region's score."""
    best_is_highest = metric == "Dice"
    order = sorted(
        common, key=lambda m: _mean_metric(stats, m, metric), reverse=best_is_highest
    )
    return {model_norm: rank for rank, model_norm in enumerate(order, start=1)}


def median_abs_rank_change_mean(
    stats_legacy: dict[str, tuple[str, dict]],
    stats_repro: dict[str, tuple[str, dict]],
    common: list[str],
    metric: str = "Dice",
) -> float:
    ranks_legacy = _ranks_within_mean(stats_legacy, common, metric)
    ranks_repro = _ranks_within_mean(stats_repro, common, metric)
    diffs = [abs(ranks_legacy[m] - ranks_repro[m]) for m in common]
    return statistics.median(diffs)


def compute_legacy_correlations_mean(
    legacy_stats: dict[str, tuple[str, dict]],
    repro_stats: dict[str, tuple[str, dict]],
    common: list[str],
    metric: str = "Dice",
) -> dict:
    """Like compute_legacy_correlations, but ranks/correlates each model's
    Dice (or HD95, via `metric`) averaged across WT/TC/ET rather than one
    region at a time -- "does the reproduction preserve the legacy paper's
    overall ranking", as opposed to its per-region ranking. Dice and HD95
    are never averaged together -- callers wanting both must call this once
    per metric and keep the two results in separate tables."""
    vals_legacy = [_mean_metric(legacy_stats, m, metric) for m in common]
    vals_repro = [_mean_metric(repro_stats, m, metric) for m in common]
    rho, _p_rho = spearmanr(vals_legacy, vals_repro)
    tau, _p_tau = kendalltau(vals_legacy, vals_repro)
    diffs = [r - l for l, r in zip(vals_legacy, vals_repro)]
    mean_diff = statistics.fmean(diffs)
    std_diff = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0
    return {
        "n": len(common),
        "spearman": rho,
        "kendall": tau,
        "median_abs_rank_change": median_abs_rank_change_mean(
            legacy_stats, repro_stats, common, metric
        ),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
    }


def print_legacy_correlation_table_mean(results_by_comparison: dict[str, dict]) -> None:
    table = Table(
        title="Legacy vs reproduced rank correlation "
        "(official test split, Dice averaged over WT/TC/ET)"
    )
    table.add_column("Comparison", style="bold cyan")
    table.add_column("n", justify="right")
    table.add_column("Spearman ρ", justify="right")
    table.add_column("Kendall τ", justify="right")
    table.add_column("Median |Δrank|", justify="right")
    table.add_column("Mean±Std Diff (repro-legacy)", justify="right")
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


def build_legacy_correlation_table_mean(
    results_by_comparison: dict[str, dict], *, label: str = "tab:rank_corr_legacy_mean"
) -> str:
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
        r"rank change, and mean$\pm$std signed Dice difference (reproduced "
        r"minus legacy, positive = reproduction reads higher; std is the "
        r"population std of that per-model difference across common "
        r"models), between the published legacy Dice scores and this "
        r"reproduction's official-test-split Dice scores, with each "
        r"model's Dice averaged over WT/TC/ET rather than compared per "
        r"region. Confirms whether the reproduction preserved the legacy "
        r"paper's overall model ranking on each dataset, and whether it "
        r"reads systematically higher or lower.}",
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


def print_legacy_correlation_table(
    results_by_comparison: dict[str, dict[str, dict]],
) -> None:
    table = Table(title="Legacy vs reproduced rank correlation (official test split, Dice)")
    table.add_column("Comparison", style="bold cyan")
    table.add_column("Region")
    table.add_column("n", justify="right")
    table.add_column("Spearman ρ", justify="right")
    table.add_column("Kendall τ", justify="right")
    table.add_column("Median |Δrank|", justify="right")
    table.add_column("Mean±Std Diff (repro-legacy)", justify="right")
    for comparison, results in results_by_comparison.items():
        for region in gst.REGIONS:
            r = results[region]
            table.add_row(
                comparison,
                region,
                str(r["n"]),
                f"{r['spearman']:.3f}",
                f"{r['kendall']:.3f}",
                f"{r['median_abs_rank_change']:.1f}",
                f"{r['mean_diff']:+.2f}±{r['std_diff']:.2f}",
            )
    Console().print(table)


def build_legacy_correlation_table(
    results_by_comparison: dict[str, dict[str, dict]], *, label: str = "tab:rank_corr_legacy"
) -> str:
    body = []
    for comparison, results in results_by_comparison.items():
        for region in gst.REGIONS:
            r = results[region]
            body.append(
                f"{comparison} & {region} & {r['spearman']:.3f} & "
                f"{r['kendall']:.3f} & {r['median_abs_rank_change']:.1f} & "
                f"{r['mean_diff']:+.2f}$\\pm${r['std_diff']:.2f} \\\\"
            )
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Spearman and Kendall rank correlation, median absolute "
        r"rank change, and mean$\pm$std signed Dice difference (reproduced "
        r"minus legacy, positive = reproduction reads higher; std is the "
        r"population std of that per-model difference across common "
        r"models), between the published legacy Dice scores and this "
        r"reproduction's official-test-split Dice scores, per tumor "
        r"region. Confirms whether the reproduction preserved the legacy "
        r"paper's relative model ranking on each dataset, and whether it "
        r"reads systematically higher or lower.}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Comparison} & \textbf{Region} & \textbf{Spearman $\rho$} & "
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

    legacy = {}
    display_names = {}
    if os.path.exists(gst.LEGACY_FILE):
        legacy, display_names = brc.parse_legacy(gst.LEGACY_FILE)

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

    include_lines = [r"\input{tables.tex}"]

    if legacy:
        legacy_stats_18 = build_legacy_stats(legacy, "BRATS2018", display_names)
        legacy_stats_23 = build_legacy_stats(legacy, "BRATS2023", display_names)

        results_by_comparison = {}
        results_by_comparison_mean = {}
        for label, legacy_stats, repro_stats in (
            ("Legacy18 vs BraTS18", legacy_stats_18, stats_18),
            ("Legacy23 vs BraTS23", legacy_stats_23, stats_23),
        ):
            common_l, only_legacy, only_repro = build_common_set(legacy_stats, repro_stats)
            if not common_l:
                print(f"{label}: no common models, skipping", file=sys.stderr)
                continue
            print_model_set_report(common_l, only_legacy, only_repro)
            results_by_comparison[label] = compute_legacy_correlations(
                legacy_stats, repro_stats, common_l
            )
            results_by_comparison_mean[label] = compute_legacy_correlations_mean(
                legacy_stats, repro_stats, common_l
            )

        if results_by_comparison:
            print_legacy_correlation_table(results_by_comparison)
            legacy_tables_path = os.path.join(OUTPUT_DIR, "tables_legacy.tex")
            with open(legacy_tables_path, "w") as f:
                f.write(build_legacy_correlation_table(results_by_comparison))
            include_lines.append(r"\clearpage")
            include_lines.append(r"\input{tables_legacy.tex}")
            print(f"Wrote {legacy_tables_path}")

        if results_by_comparison_mean:
            print_legacy_correlation_table_mean(results_by_comparison_mean)
            legacy_tables_mean_path = os.path.join(OUTPUT_DIR, "tables_legacy_mean.tex")
            with open(legacy_tables_mean_path, "w") as f:
                f.write(build_legacy_correlation_table_mean(results_by_comparison_mean))
            include_lines.append(r"\clearpage")
            include_lines.append(r"\input{tables_legacy_mean.tex}")
            print(f"Wrote {legacy_tables_mean_path}")

        # Legacy23 vs BraTS23 restricted to fold1 only (not the fold1/3/5
        # mean used everywhere else above) -- e.g. to see whether the mean
        # signed diff is already visible on a single split, or only emerges
        # once folds are averaged.
        stats_23_fold1 = build_single_fold_stats(models, "23", "fold1", display_names)
        common_f1, only_legacy_f1, only_repro_f1 = build_common_set(
            legacy_stats_23, stats_23_fold1
        )
        if common_f1:
            print_model_set_report(common_f1, only_legacy_f1, only_repro_f1)
            results_fold1 = {
                "Legacy23 vs BraTS23 (fold1)": compute_legacy_correlations(
                    legacy_stats_23, stats_23_fold1, common_f1
                )
            }
            print_legacy_correlation_table(results_fold1)
            legacy_fold1_path = os.path.join(OUTPUT_DIR, "tables_legacy_23_fold1.tex")
            with open(legacy_fold1_path, "w") as f:
                f.write(
                    build_legacy_correlation_table(
                        results_fold1, label="tab:rank_corr_legacy_23_fold1"
                    )
                )
            include_lines.append(r"\clearpage")
            include_lines.append(r"\input{tables_legacy_23_fold1.tex}")
            print(f"Wrote {legacy_fold1_path}")
        else:
            print("Legacy23 vs BraTS23 (fold1): no common models, skipping", file=sys.stderr)
    else:
        print(f"No legacy file at {gst.LEGACY_FILE}; skipping legacy comparison", file=sys.stderr)

    main_tex = (
        "\\documentclass[12pt]{article}\n"
        "\\usepackage[margin=0.8in]{geometry}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{lmodern}\n"
        "\\pagestyle{plain}\n"
        "\\begin{document}\n"
        + "\n".join(include_lines)
        + "\n\\end{document}\n"
    )
    main_path = os.path.join(OUTPUT_DIR, "main.tex")
    with open(main_path, "w") as f:
        f.write(main_tex)

    print(f"Wrote {tables_path}")
    print(f"Wrote {main_path}")
    gst.compile_pdf(main_path)


if __name__ == "__main__":
    main()
