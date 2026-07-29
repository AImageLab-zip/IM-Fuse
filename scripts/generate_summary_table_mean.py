#!/usr/bin/env python3
"""Single summary table averaging generate_summary_tables.py's three
per-region (WT/TC/ET) tables into one overall table.

Reuses generate_summary_tables.py's discovery/completeness pipeline (discover,
is_complete, EXCLUDED_MODELS, FALLBACK_DISPLAY_NAMES, CITATION_KEYS,
tex_escape) but NOT its aggregate() -- that function's "average of per-combo
means, average of per-combo stds" convention isn't what this table wants.
Instead, for each model/group/metric, every one of the 15 modality
combinations and all three tumor regions (WT/TC/ET) is averaged away WITHIN
each fold first, leaving exactly one number per fold (3 numbers, matching
EXPECTED_FOLDS); the reported mean and std are then this model's plain mean
and population std across just those 3 per-fold numbers -- see
aggregate_mean_over_folds.

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
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "summary_table_mean"
)

METRICS = gst.METRICS  # ["Dice", "HD95"]
REGIONS = gst.REGIONS  # ["WT", "TC", "ET"]
EXPECTED_FOLDS = gst.EXPECTED_FOLDS  # ["fold1", "fold3", "fold5"]
MODALITY_KEY_ORDER = gst.MODALITY_KEY_ORDER  # 15 combinations


def aggregate_mean_over_folds(folds: dict) -> dict:
    """{"Dice": (mean, std), "HD95": (mean, std)} for a single model/group.

    Within each fold, averages over all 15 modality combinations AND all
    three regions (WT/TC/ET) at once, leaving one plain number per fold (3
    numbers total). The reported mean/std are then this model's mean and
    population std across just those 3 fold-level numbers -- unlike
    gst.aggregate, which averages per-combo (mean, std) pairs computed across
    folds and never reduces to a 3-value sample."""
    out = {}
    for metric, suffix in (("Dice", ""), ("HD95", "_hd95")):
        fold_values = []
        for fold in EXPECTED_FOLDS:
            vals = [
                folds[fold][mkey][region + suffix]
                for mkey in MODALITY_KEY_ORDER
                for region in REGIONS
            ]
            fold_values.append(statistics.fmean(vals))
        out[metric] = (statistics.fmean(fold_values), statistics.pstdev(fold_values))
    return out


def build_mean_rows(models: dict, internal_models: dict, display_names: dict):
    """Like generate_summary_tables.build_class_rows, but each GROUPS entry's
    stats are computed with aggregate_mean_over_folds directly from the raw
    per-fold data instead of gst.aggregate -- so the region axis is collapsed
    as part of aggregation rather than as a separate post-hoc step, and the
    3-fold std described above is preserved rather than lost to an
    intermediate per-region std."""
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
                group_stats[label] = aggregate_mean_over_folds(folds)
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
        "Mean $\\pm$ std Dice (\\%) and HD95 (mm), per dataset, first averaged within each of "
        "our three folds over all modality-presence combinations and the three tumor regions "
        "(WT/TC/ET), then reported as the mean and std across those three per-fold numbers. "
        "BraTS18/BraTS25-pre are the official test splits; MB-96 - BraTS18 chp/MB-96 - "
        "BraTS25-pre chp are the internal cohort, scored with the BraTS18-trained and "
        "BraTS25-pre-trained checkpoints respectively. "
        "Rank is by Dice, among models with data for that group. Bold marks the best model "
        "per column, underline the runner-up."
    )
    table = build_table(mean_rows, caption, "tab:mean_regions")

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
