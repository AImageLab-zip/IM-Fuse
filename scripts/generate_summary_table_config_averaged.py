#!/usr/bin/env python3
"""Single summary table averaging generate_summary_tables.py's three
per-region (WT/TC/ET) tables into one overall table -- like
generate_summary_table_mean.py, but computing the std the way
generate_summary_tables.aggregate() does (per-configuration, then averaged
over configurations) instead of collapsing straight to 3 fold-level numbers.

For each model/group/metric: Dice (or HD95) is first averaged across
WT/TC/ET for every (modality configuration, fold) pair. Then, for each of
the 15 modality configurations, its own mean and std are computed across
the three folds. The reported mean/std are those 15 per-configuration means
and stds, themselves averaged over all 15 configurations -- see
aggregate_config_level_mean.

This differs from generate_summary_table_mean.py's aggregate_mean_over_folds,
which averages away the modality-configuration axis at the same time as the
region axis (leaving only 3 fold-level numbers) before ever computing a std,
so its std reflects fold-to-fold variability. Here, std is computed within
each configuration first (matching gst.aggregate's per-region convention,
just applied to the region-averaged score), so it instead reflects
run-to-run variability at fixed configuration, averaged over configurations.

Reuses generate_summary_tables.py's discovery/completeness pipeline
(discover, is_complete, EXCLUDED_MODELS, FALLBACK_DISPLAY_NAMES,
CITATION_KEYS, tex_escape, GROUPS, compile_pdf) and
generate_summary_table_mean.py's row-building/table-rendering shape
(build_mean_rows/build_table), swapping in aggregate_config_level_mean as
the aggregation function.

Writes one `.tex` file with the single collapsed table plus a standalone
main.tex into OUTPUT_DIR, then compiles it to PDF with tectonic (if available).
"""

from __future__ import annotations

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_summary_tables as gst  # noqa: E402  (reuse discover/is_complete/compile_pdf)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "summary_table_config_averaged"
)

METRICS = gst.METRICS  # ["Dice", "HD95"]
REGIONS = gst.REGIONS  # ["WT", "TC", "ET"]
EXPECTED_FOLDS = gst.EXPECTED_FOLDS  # ["fold1", "fold3", "fold5"]
MODALITY_KEY_ORDER = gst.MODALITY_KEY_ORDER  # 15 combinations


def aggregate_config_level_mean(folds: dict) -> dict:
    """{"Dice": (mean, std), "HD95": (mean, std)} for a single model/group.

    For each of the 15 modality configurations: Dice (or HD95) is first
    averaged across WT/TC/ET for each fold, then that configuration's own
    mean and population std are computed across the 3 folds. The reported
    mean/std are those 15 per-configuration means/stds, themselves averaged
    over all 15 configurations -- gst.aggregate's
    per-configuration-then-average-over-configurations convention, applied
    to the region-averaged score rather than one region at a time."""
    out = {}
    for metric, suffix in (("Dice", ""), ("HD95", "_hd95")):
        combo_means, combo_stds = [], []
        for mkey in MODALITY_KEY_ORDER:
            vals = [
                statistics.fmean(folds[fold][mkey][region + suffix] for region in REGIONS)
                for fold in EXPECTED_FOLDS
            ]
            combo_means.append(statistics.fmean(vals))
            combo_stds.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
        out[metric] = (statistics.fmean(combo_means), statistics.fmean(combo_stds))
    return out


def build_mean_rows(models: dict, internal_models: dict, display_names: dict):
    """Like generate_summary_table_mean.build_mean_rows, but each GROUPS
    entry's stats are computed with aggregate_config_level_mean instead of
    aggregate_mean_over_folds."""
    all_model_norms = set(models) | set(internal_models)
    rows = []
    for model_norm in sorted(all_model_norms):
        if model_norm in gst.EXCLUDED_MODELS:
            continue

        group_stats = {}
        any_present = False
        for label, source, dsnum in gst.GROUPS:
            source_dict = models if source == "official" else internal_models
            folds = source_dict.get(model_norm, {}).get(dsnum)
            if folds and gst.is_complete(folds):
                group_stats[label] = aggregate_config_level_mean(folds)
                any_present = True
            else:
                group_stats[label] = None
        if not any_present:
            continue

        display = display_names.get(model_norm) or gst.FALLBACK_DISPLAY_NAMES.get(
            model_norm, model_norm.upper()
        )
        display = gst.tex_escape(display)
        key = gst.CITATION_KEYS.get(model_norm)
        if key:
            display = f"{display}~\\cite{{{key}}}"
        rows.append((display, group_stats))
    rows.sort(key=lambda r: r[0].lower())
    return rows


def best_per_group(rows, metric, label):
    """Index of the best row for (metric, group label): max for Dice, min
    for HD95. Rows lacking that group (None) are excluded."""
    values = [(i, r[1][label][metric][0]) for i, r in enumerate(rows) if r[1][label] is not None]
    if not values:
        return None
    if metric == "Dice":
        return max(values, key=lambda t: t[1])[0]
    return min(values, key=lambda t: t[1])[0]


def second_best_per_group(rows, metric, label, best_idx):
    """Index of the runner-up row for (metric, group label), excluding the
    best and any row lacking that group."""
    values = [
        (i, r[1][label][metric][0])
        for i, r in enumerate(rows)
        if r[1][label] is not None and i != best_idx
    ]
    if not values:
        return None
    if metric == "Dice":
        return max(values, key=lambda t: t[1])[0]
    return min(values, key=lambda t: t[1])[0]


def dice_ranks_per_group(rows, label) -> dict[int, int]:
    """{row_index: rank}, rank 1 = highest region-averaged Dice for this
    group, among rows that have data for this group."""
    present = [i for i, r in enumerate(rows) if r[1][label] is not None]
    order = sorted(present, key=lambda i: -rows[i][1][label]["Dice"][0])
    return {row_i: rank for rank, row_i in enumerate(order, start=1)}


def build_table(rows, caption, label):
    """One table, all models, columns are, per GROUPS entry, a Dice-based
    rank then region-averaged Dice/HD95 mean+-std."""
    if not rows:
        return ""

    group_labels = [g[0] for g in gst.GROUPS]
    bests = {
        (lbl, metric): best_per_group(rows, metric, lbl)
        for lbl in group_labels
        for metric in METRICS
    }
    runners_up = {
        (lbl, metric): second_best_per_group(rows, metric, lbl, bests[(lbl, metric)])
        for lbl in group_labels
        for metric in METRICS
    }
    ranks_by_group = {lbl: dice_ranks_per_group(rows, lbl) for lbl in group_labels}

    body = []
    for i, (name, group_stats) in enumerate(rows):
        cells = [name]
        for lbl in group_labels:
            stats = group_stats[lbl]
            if stats is None:
                cells.extend(["--", "--", "--"])
                continue
            rank = ranks_by_group[lbl].get(i)
            cells.append(str(rank) if rank is not None else "--")
            for metric in METRICS:
                mean, std = stats[metric]
                text = f"{mean:.1f}$\\pm${std:.1f}"
                if bests[(lbl, metric)] == i:
                    text = f"\\textbf{{{text}}}"
                elif runners_up[(lbl, metric)] == i:
                    text = f"\\underline{{{text}}}"
                cells.append(text)
        body.append(" & ".join(cells) + r" \\")

    n_sub = 3  # Rank, Dice, HD95

    group_header = " & ".join(
        f"\\multicolumn{{{n_sub}}}{{c}}{{\\textbf{{{lbl}}}}}" for lbl in group_labels
    )
    group_cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + i * n_sub}-{1 + (i + 1) * n_sub}}}" for i in range(len(group_labels))
    )
    sub_header = " & ".join("R & Dice & HD95" for _ in group_labels)
    colspec = "l" + "ccc" * len(group_labels)

    lines = [
        r"\begin{table*}[!ht]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\fontsize{8}{10}\selectfont",
        r"\setlength{\tabcolsep}{3pt}",
        r"\rowcolors{3}{gray!8}{white}",
        f"\\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        f" & {group_header} \\\\",
        group_cmidrules,
        f"\\textbf{{Model}} & {sub_header} \\\\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    display_names = {}
    if os.path.exists(gst.LEGACY_FILE):
        _, display_names = gst.brc.parse_legacy(gst.LEGACY_FILE)

    models, internal_models, dsnums = gst.discover(gst.RESULTS_DIR)
    if not dsnums:
        print("No result directories found in", gst.RESULTS_DIR)
        return

    mean_rows = build_mean_rows(models, internal_models, display_names)
    if not mean_rows:
        print("No model has at least one complete group (BraTS18/BraTS25-pre/MB-96); nothing to write")
        return

    caption = (
        "Overall segmentation performance reported as mean $\\pm$ standard deviation Dice (\\%) "
        "and HD95 (mm). Dice and HD95 were first computed independently for WT, TC, and ET and "
        "then averaged across the three regions. For each of the 15 non-empty modality "
        "configurations, the mean and standard deviation were subsequently computed across the "
        "three splits. The values reported in the table were obtained by averaging these "
        "configuration-level means and standard deviations over all 15 configurations. "
        "BraTS18/BraTS25-pre are the official test splits; MB-96 - BraTS18 chp/MB-96 - "
        "BraTS25-pre chp are the internal cohort, scored with the BraTS18-trained and "
        "BraTS25-pre-trained checkpoints respectively. "
        "Rank is by Dice, among models with data for that group. Bold marks the best model "
        "per column, underline the runner-up."
    )
    table = build_table(mean_rows, caption, "tab:config_averaged")

    tables_path = os.path.join(OUTPUT_DIR, "tables.tex")
    with open(tables_path, "w") as f:
        f.write(table)

    # NOTE: this wrapper is a throwaway local-preview document only, to eyeball
    # the table as a standalone PDF -- see generate_summary_tables.py's main()
    # for why it must never be copied into the paper's actual .tex source.
    main_tex = r"""\documentclass[12pt]{article}
\usepackage[paperwidth=10in, paperheight=13in, margin=0.6in]{geometry}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{makecell}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage[table]{xcolor}
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
