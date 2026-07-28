#!/usr/bin/env python3
"""Build per-dataset summary tables: models x (WT, TC, ET), each region
showing overall Dice and HD95 as mean +/- std over ALL modality-presence
combinations and folds.

For every dataset found under RESULTS_DIR (BraTS 2018 / 2023 / 2025, ...)
this emits two tables:
  - the official test-set table (results_fold*.xlsx)
  - the internal-dataset table, scored with the same checkpoints against the
    internal cohort (results_internal_fold*.xlsx)

A model is only included in a given table if it has COMPLETE results for it:
all of EXPECTED_FOLDS present, each covering all 15 modality-presence
combinations, with every WT/TC/ET Dice and HD95 value populated. Partial
results are silently dropped from that table (a model can appear in the
official table but be missing from the internal one, or vice versa).

Per model/region, the reported value is:
  mean over the 15 modality combinations of (mean over folds), "+-"
  mean over the 15 modality combinations of (std over folds)
i.e. the column-wise mean of the "Mean" and "Std" columns that
build_reproduction_comparison.py computes per modality row.

Writes one `.tex` file with all tables plus a standalone `main.tex` wrapper
into OUTPUT_DIR, then compiles it to PDF with `tectonic` (if available).
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict

import openpyxl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_reproduction_comparison as brc  # noqa: E402  (reuse parse_legacy + constants)

RESULTS_DIR = brc.RESULTS_DIR
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "summary_tables"
)
LEGACY_FILE = os.path.join(RESULTS_DIR, brc.LEGACY_FILE)

REGIONS = brc.REGIONS  # ["WT", "TC", "ET"]
REGION_LABELS = {"WT": "Whole Tumor", "TC": "Tumor Core", "ET": "Enhancing Tumor"}
METRICS = ["Dice", "HD95"]
EXPECTED_FOLDS = brc.EXPECTED_FOLDS  # ["fold1", "fold3", "fold5"]
MODALITY_KEY_ORDER = brc.MODALITY_KEY_ORDER  # 15 combinations
DATASET_LABELS = {"18": "BraTS 2018", "23": "BraTS 2023", "25": "BraTS 2025"}
EXCLUDED_MODELS = {brc.normalize(n) for n in ("tinymimosaweighted", "shaspec", "mstkdnet", "m3ae")}
# Fallback display names for models not present in the legacy display-name
# file (kept consistent with generate_flops_table.py's DISPLAY_NAMES).
FALLBACK_DISPLAY_NAMES = {"tinymimosa": "TinyMimosa", "manymimosas": "ManyMimosas"}


def dataset_label(dsnum: str) -> str:
    return DATASET_LABELS.get(dsnum, f"Dataset {dsnum}")


# --------------------------------------------------------------------------- #
# Discovery: <model>_<dsnum> directories, official + internal fold files.
# --------------------------------------------------------------------------- #
def discover(results_dir):
    """Return (models, internal_models, dsnums).

    models / internal_models: {model_norm: {dsnum: {fold_label: parsed_dict}}}
    parsed_dict: {mkey: {"WT": v, "WT_hd95": v, "TC": v, ...}} or {} on parse failure.
    """
    models = defaultdict(dict)
    internal_models = defaultdict(dict)
    dsnums = set()

    entries = [
        e for e in sorted(os.listdir(results_dir))
        if os.path.isdir(os.path.join(results_dir, e))
    ]
    for entry in entries:
        m = re.match(r"^(.+)_(\d{2})$", entry)
        if not m:
            continue
        model_raw, dsnum = m.group(1), m.group(2)
        full = os.path.join(results_dir, entry)

        fold_files = sorted(
            glob.glob(os.path.join(full, "results_fold*.xlsx")),
            key=lambda p: int(re.search(r"fold(\d+)", p).group(1)),
        )
        internal_fold_files = sorted(
            glob.glob(os.path.join(full, "results_internal_fold*.xlsx")),
            key=lambda p: int(re.search(r"fold(\d+)", p).group(1)),
        )
        if not fold_files and not internal_fold_files:
            continue

        dsnums.add(dsnum)
        norm = brc.normalize(model_raw)

        if fold_files:
            folds = {}
            for ff in fold_files:
                folds[brc.fold_label(os.path.basename(ff))] = brc.parse_result_file(ff)
            models[norm][dsnum] = folds
        if internal_fold_files:
            internal_folds = {}
            for ff in internal_fold_files:
                internal_folds[brc.fold_label(os.path.basename(ff))] = brc.parse_result_file(ff)
            internal_models[norm][dsnum] = internal_folds

    return models, internal_models, sorted(dsnums)


# --------------------------------------------------------------------------- #
# Completeness + aggregation.
# --------------------------------------------------------------------------- #
def is_complete(folds: dict) -> bool:
    """True iff every EXPECTED_FOLDS is present and fully populated."""
    for fold in EXPECTED_FOLDS:
        entry = folds.get(fold)
        if not entry:
            return False
        for mkey in MODALITY_KEY_ORDER:
            row = entry.get(mkey)
            if not row:
                return False
            for region in REGIONS:
                if row.get(region) is None or row.get(f"{region}_hd95") is None:
                    return False
    return True


def aggregate(folds: dict) -> dict:
    """{region: {"Dice": (mean, std), "HD95": (mean, std)}}, averaged over the
    15 modality combinations of the per-combination (mean, std) over folds."""
    out = {}
    for region in REGIONS:
        out[region] = {}
        for metric, suffix in (("Dice", ""), ("HD95", "_hd95")):
            key = region + suffix
            combo_means, combo_stds = [], []
            for mkey in MODALITY_KEY_ORDER:
                vals = [folds[fold][mkey][key] for fold in EXPECTED_FOLDS]
                combo_means.append(statistics.fmean(vals))
                combo_stds.append(statistics.pstdev(vals))
            out[region][metric] = (statistics.fmean(combo_means), statistics.fmean(combo_stds))
    return out


SOURCES = ["BraTS", "Internal"]


def build_rows(models: dict, internal_models: dict, dsnum: str, display_names: dict):
    """Return [(display_name, {region: {metric: {"BraTS": (mean,std), "Internal": (mean,std)}}}), ...].

    Only models with COMPLETE results on both the official test split and the
    internal cohort (for this dataset) are included, since every row needs
    both subcolumns populated."""
    rows = []
    for model_norm in models:
        if model_norm in EXCLUDED_MODELS:
            continue
        official_folds = models.get(model_norm, {}).get(dsnum)
        internal_folds = internal_models.get(model_norm, {}).get(dsnum)
        if not official_folds or not is_complete(official_folds):
            continue
        if not internal_folds or not is_complete(internal_folds):
            continue

        official_agg = aggregate(official_folds)
        internal_agg = aggregate(internal_folds)
        stats = {
            region: {
                metric: {"BraTS": official_agg[region][metric], "Internal": internal_agg[region][metric]}
                for metric in METRICS
            }
            for region in REGIONS
        }
        display = display_names.get(model_norm) or FALLBACK_DISPLAY_NAMES.get(model_norm, model_norm.upper())
        rows.append((display, stats))
    rows.sort(key=lambda r: r[0].lower())
    return rows


# --------------------------------------------------------------------------- #
# LaTeX rendering.
# --------------------------------------------------------------------------- #
def tex_escape(s: str) -> str:
    return re.sub(r"([&%$#_{}])", r"\\\1", s)


def best_per_column(rows, region, metric, source):
    """Index of the best row for (region, metric, source): max for Dice, min for HD95."""
    values = [(i, r[1][region][metric][source][0]) for i, r in enumerate(rows)]
    if not values:
        return None
    if metric == "Dice":
        return max(values, key=lambda t: t[1])[0]
    return min(values, key=lambda t: t[1])[0]


def second_best_per_column(rows, region, metric, source, best_idx):
    """Index of the runner-up row for (region, metric, source), excluding the best."""
    values = [(i, r[1][region][metric][source][0]) for i, r in enumerate(rows) if i != best_idx]
    if not values:
        return None
    if metric == "Dice":
        return max(values, key=lambda t: t[1])[0]
    return min(values, key=lambda t: t[1])[0]


def dice_official_ranks(rows, region) -> dict[int, int]:
    """{row_index: rank}, rank 1 = highest Dice on the official BraTS split for
    this region. Ties broken by the rows' existing (alphabetical) order."""
    order = sorted(range(len(rows)), key=lambda i: -rows[i][1][region]["Dice"]["BraTS"][0])
    return {row_i: rank for rank, row_i in enumerate(order, start=1)}


def build_table(rows, region, caption, label):
    """One table for a single region: rows are models, columns are a Dice(BraTS)
    rank, then Dice/HD95 x BraTS/Internal."""
    if not rows:
        return ""

    bests = {
        (metric, source): best_per_column(rows, region, metric, source)
        for metric in METRICS
        for source in SOURCES
    }
    runners_up = {
        (metric, source): second_best_per_column(rows, region, metric, source, bests[(metric, source)])
        for metric in METRICS
        for source in SOURCES
    }
    ranks = dice_official_ranks(rows, region)

    body = []
    for i, (name, stats) in enumerate(rows):
        cells = [str(ranks[i]), tex_escape(name)]
        for metric in METRICS:
            for source in SOURCES:
                mean, std = stats[region][metric][source]
                text = f"{mean:.1f}$\\pm${std:.1f}"
                if bests[(metric, source)] == i:
                    text = f"\\textbf{{{text}}}"
                elif runners_up[(metric, source)] == i:
                    text = f"\\underline{{{text}}}"
                cells.append(text)
        body.append(" & ".join(cells) + r" \\")

    n_sub = len(SOURCES)  # columns per metric (2)

    # Level 1: Dice / HD95, each spanning its 2 sources. Rank + Model are the
    # first two (unspanned) columns.
    metric_header = " & ".join(
        f"\\multicolumn{{{n_sub}}}{{c}}{{\\textbf{{{m}}}}}" for m in METRICS
    )
    metric_cmidrules = " ".join(
        f"\\cmidrule(lr){{{3 + i * n_sub}-{2 + (i + 1) * n_sub}}}" for i in range(len(METRICS))
    )

    # Level 2: BraTS / Internal.
    source_header = " & ".join(f"{s}" for _ in METRICS for s in SOURCES)

    colspec = "cl" + "".join("cc" for _ in range(len(METRICS)))

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\rowcolors{3}{gray!8}{white}",
        f"\\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        f"\\textbf{{Rank}} & \\textbf{{Model}} & {metric_header} \\\\",
        metric_cmidrules,
        f" & & {source_header} \\\\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    display_names = {}
    if os.path.exists(LEGACY_FILE):
        _, display_names = brc.parse_legacy(LEGACY_FILE)

    models, internal_models, dsnums = discover(RESULTS_DIR)
    if not dsnums:
        print("No result directories found in", RESULTS_DIR)
        return

    sections = []
    for dsnum in dsnums:
        label = dataset_label(dsnum)
        sections.append(f"\\section*{{{label}}}")

        rows = build_rows(models, internal_models, dsnum, display_names)
        if not rows:
            print(f"{label}: no model complete on both splits, skipping tables")
            sections.append(r"\clearpage")
            continue

        for region in REGIONS:
            caption = (
                f"{label}, {REGION_LABELS[region]} ({region}): mean $\\pm$ std Dice (\\%) and "
                "HD95 (mm) over all modality-presence combinations and folds, evaluated on the "
                "official BraTS test split and on the internal cohort (using the same "
                "checkpoints trained on this dataset). Rank is by Dice on the official BraTS "
                "split. Bold marks the best model per column."
            )
            table = build_table(rows, region, caption, f"tab:{dsnum}_{region.lower()}")
            sections.append(table)

        sections.append(r"\clearpage")

    tables_path = os.path.join(OUTPUT_DIR, "tables.tex")
    with open(tables_path, "w") as f:
        f.write("\n".join(sections))

    # NOTE: this wrapper is a throwaway local-preview document only, to eyeball
    # the tables as a standalone PDF. It is NOT AAAI-compliant (geometry and
    # \input are both explicitly banned by the AAAI author instructions) and
    # must never be copied into the paper's actual .tex source -- only the
    # \begin{table}...\end{table} blocks from tables.tex should be pasted in.
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

    compile_pdf(main_path)


def compile_pdf(main_path):
    tectonic = shutil.which("tectonic") or os.path.expanduser("~/.local/bin/tectonic")
    if not (tectonic and os.access(tectonic, os.X_OK)):
        print("tectonic not found on PATH or in ~/.local/bin; skipping PDF compile", file=sys.stderr)
        return

    result = subprocess.run(
        [tectonic, os.path.basename(main_path)],
        cwd=os.path.dirname(main_path),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        print("tectonic compile failed", file=sys.stderr)
        return

    pdf_path = os.path.splitext(main_path)[0] + ".pdf"
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
