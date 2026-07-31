#!/usr/bin/env python3
"""Turn `reproduction-comparison.xlsx` into LaTeX comparison tables.

Reads the workbook produced by `build_reproduction_comparison.py` (one sheet
per model, each holding a BRATS2018 and a BRATS2023 table) and emits, for
every (dataset, region, metric) combination, a `table*` with models as rows
and modality-presence combinations as columns. Each cell stacks the legacy
value on top of the reproduced mean +/- std (Dice tables) or shows just the
reproduced mean +/- std (HD95 tables, since the legacy workbook has no HD95
numbers).

Additionally emits an "MB-96" section: results of the same checkpoints
(trained on BRATS2018 or BRATS2023) evaluated against the internal dataset,
read from each sheet's "Internal Mean" / "Internal Std" columns. There is no
legacy reference for this dataset, so these tables only ever show the
reproduced mean +/- std.

Writes one `.tex` file with all tables plus a standalone `main.tex` wrapper
into OUTPUT_DIR, then compiles it to PDF with `tectonic`.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

import openpyxl

RESULTS_DIR = "/work/phd_mimose/results"
WORKBOOK = os.path.join(RESULTS_DIR, "reproduction-comparison.xlsx")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "latex_tables")

DATASETS = ["BRATS2018", "BRATS2023"]
REGIONS = ["WT", "TC", "ET"]
METRICS = ["Dice", "HD95"]
# Column order matches build_reproduction_comparison.py's MODALITY_COLS
# (Fl, T1, T1c, T2) -- only the display labels change here, to the BraTS2023
# modality names: Fl->T2f, T1->T1n, T1c->T1c, T2->T2w.
MODALITY_COLS = ["T2f", "T1n", "T1c", "T2w"]

# Sheet names (normalized: lowercased, non-alphanumeric stripped) to leave
# out of every table entirely. "summary" is the workbook overview tab, not a
# model -- it has the same dataset/region headers but no data rows, so
# leaving it in silently added a spurious all-"--" row to every table.
EXCLUDED_MODELS = {
    "summary",
    "manymimosas",
    "mambavitakd",
    "mstkdnet",
    "olduhved",
    "tinymimosa",
    "tinymimosaweighted",
    "shaspec",
    "m3ae",
    "mcpl",
}


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def squircle(name: str, present: bool) -> str:
    # "minimum width/height" is only a floor -- glyph metrics differ across
    # T2f/T1n/T1c/T2w (bold w/n are wider than f/c, and f's descender changes
    # the natural depth), so content that exceeds the floor grows past it
    # unevenly. "text width/height/depth" force a truly fixed box regardless
    # of content, so every badge is pixel-identical.
    if present:
        draw, font, color = "draw=black, line width=0.4pt", r"\bfseries\scriptsize", "black"
    else:
        draw, font, color = "draw=none", r"\scriptsize", "gray!55"
    return (
        f"\\tikz[baseline=-0.5ex]{{\\node[rounded corners=2.5pt, {draw}, "
        f"text width=1.7em, text height=0.75em, text depth=0.15em, "
        f"inner sep=1pt, align=center, text={color}, font={font}] {{{name}}};}}"
    )


def modality_label(mkey: tuple) -> str:
    """2x2 grid of modality abbreviations (MODALITY_COLS order, row-major):
    present ones get a thin outlined badge, missing ones are plain light
    gray text -- self-explanatory at a glance, no legend needed."""
    cells = [squircle(name, present) for name, present in zip(MODALITY_COLS, mkey)]
    top = f"{cells[0]} {cells[1]}"
    bottom = f"{cells[2]} {cells[3]}"
    return f"\\makecell{{{top} \\\\[3pt] {bottom}}}"


def tex_escape(s: str) -> str:
    return re.sub(r"([&%$#_{}])", r"\\\1", s)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_sheet(ws):
    """Return {dataset: {region: {metric: [(mkey, legacy, mean, std), ...]}}}."""
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    out = {}

    i = 0
    while i < len(rows):
        row = rows[i]
        if row and isinstance(row[0], str) and row[0] in DATASETS:
            dataset = row[0]
            group_row = rows[i + 1]
            sub_row = rows[i + 2]

            # Map each column >= 4 to its (metric, region) group.
            col_group = {}
            current = None
            for c, cell in enumerate(group_row):
                if isinstance(cell, str) and "·" in cell:
                    metric, region = [p.strip() for p in cell.split("·")]
                    current = (metric, region)
                if current is not None and c >= 4:
                    col_group[c] = current

            # Within each group, find the Legacy / Mean / Std / Internal
            # Mean / Internal Std sub-columns.
            group_cols = {}  # (metric, region) -> {'Legacy': c, 'Mean': c, ...}
            for c, cell in enumerate(sub_row):
                if c < 4 or c not in col_group:
                    continue
                label = str(cell).strip() if cell is not None else ""
                if label in ("Legacy", "Mean", "Std", "Internal Mean", "Internal Std"):
                    group_cols.setdefault(col_group[c], {})[label] = c

            # Data rows until a blank row or the trailing "MEAN" summary row
            # (its bool(row[0])==True, bool(None)==False for cols 1-3 would
            # otherwise collide with the real Flair-only mkey and silently
            # clobber it, since by_key construction keeps the last entry).
            r = i + 3
            data = {region: {metric: [] for metric in METRICS} for region in REGIONS}
            while r < len(rows) and any(v is not None for v in rows[r]):
                drow = rows[r]
                if isinstance(drow[0], str) and drow[0].strip().upper() == "MEAN":
                    break
                mkey = tuple(bool(drow[k]) for k in range(4))
                for (metric, region), cols in group_cols.items():
                    legacy = drow[cols["Legacy"]] if "Legacy" in cols else None
                    mean = drow[cols["Mean"]] if "Mean" in cols else None
                    std = drow[cols["Std"]] if "Std" in cols else None
                    imean = drow[cols["Internal Mean"]] if "Internal Mean" in cols else None
                    istd = drow[cols["Internal Std"]] if "Internal Std" in cols else None
                    data[region][metric].append((mkey, legacy, mean, std, imean, istd))
                r += 1

            out[dataset] = data
            i = r
        else:
            i += 1

    return out


def load_all(path):
    """Return {model_display: {dataset: {region: {metric: [rows]}}}}."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    models = {}
    for name in wb.sheetnames:
        if normalize_name(name) in EXCLUDED_MODELS:
            continue
        models[name] = parse_sheet(wb[name])
    return models


# --------------------------------------------------------------------------- #
# LaTeX rendering
# --------------------------------------------------------------------------- #
def fmt_num(v):
    return f"{v:.1f}" if isinstance(v, (int, float)) else None


def cell_text(legacy, mean, std, source):
    if source == "legacy":
        legacy_s = fmt_num(legacy)
        return legacy_s if legacy_s is not None else "--"

    # source == "repro"
    mean_s = fmt_num(mean)
    std_s = fmt_num(std)
    if mean_s is not None and std_s is not None:
        return f"{mean_s}$\\pm${std_s}"
    return mean_s if mean_s is not None else "--"


def build_table(models, dataset, region, metric, source, use_internal=False):
    mkeys = None
    rows_tex = []
    model_names = sorted(
        (name for name in models if dataset in models[name]), key=str.lower
    )

    for name in model_names:
        entries = models[name][dataset][region][metric]
        if mkeys is None:
            mkeys = [e[0] for e in entries]
        by_key = {e[0]: e for e in entries}

        cells = []
        legacies, means = [], []
        for mkey in mkeys:
            _, legacy, mean, std, imean, istd = by_key.get(
                mkey, (mkey, None, None, None, None, None)
            )
            if use_internal:
                legacy, mean, std = None, imean, istd
            cells.append(cell_text(legacy, mean, std, source))
            if isinstance(legacy, (int, float)):
                legacies.append(legacy)
            if isinstance(mean, (int, float)):
                means.append(mean)

        avg_legacy = sum(legacies) / len(legacies) if legacies else None
        avg_mean = sum(means) / len(means) if means else None
        avg_cell = cell_text(avg_legacy, avg_mean, None, source)

        row = " & ".join([tex_escape(name)] + cells + [avg_cell])
        rows_tex.append(row + r" \\")

    if mkeys is None:
        return ""

    header_mods = " & ".join(modality_label(k) for k in mkeys)

    # A thin gray rule between every score column, plus a heavier black rule
    # right before Avg to set it apart as the summary column.
    thin_rule = "!{\\color{gray!35}\\vrule width 0.4pt}"
    colspec = "l" + f"{thin_rule}c" * len(mkeys) + "!{\\vrule width 0.6pt}c"

    if use_internal:
        unit = "\\%" if metric == "Dice" else ""
        caption = (
            f"{metric}{unit} on the {region} region, MB-96 dataset "
            f"(checkpoints trained on {dataset}; mean $\\pm$ std over folds; "
            "no legacy reference available for this dataset)."
        )
        label = f"tab:mb96_{dataset.lower()}_{region.lower()}_{metric.lower()}"
    elif source == "legacy":
        caption = f"Legacy {metric}\\% on the {region} region, {dataset} dataset."
        label = f"tab:{dataset.lower()}_{region.lower()}_{metric.lower()}_{source}"
    elif metric == "Dice":
        caption = (
            f"Reproduced {metric}\\% on the {region} region, {dataset} dataset "
            "(mean $\\pm$ std over the reproduced folds)."
        )
        label = f"tab:{dataset.lower()}_{region.lower()}_{metric.lower()}_{source}"
    else:
        caption = (
            f"Reproduced {metric} on the {region} region, {dataset} dataset "
            "(mean $\\pm$ std over folds; no legacy reference available for this metric)."
        )
        label = f"tab:{dataset.lower()}_{region.lower()}_{metric.lower()}_{source}"

    lines = [
        r"\begin{table*}[!ht]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begingroup",
        r"\renewcommand{\arraystretch}{1.3}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\rowcolors{2}{white}{gray!6}",
        f"\\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        f"\\textbf{{Model}} & {header_mods} & \\textbf{{Avg.}} \\\\",
        r"\midrule",
        *rows_tex,
        r"\bottomrule",
        r"\end{tabular}",
        r"\endgroup",
        r"}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    models = load_all(WORKBOOK)

    sections = []
    for dataset in DATASETS:
        sections.append(f"\\section*{{{dataset}}}")
        for region in REGIONS:
            for metric in METRICS:
                sources = ("legacy", "repro") if metric == "Dice" else ("repro",)
                for source in sources:
                    table = build_table(models, dataset, region, metric, source)
                    if table:
                        sections.append(table)
        sections.append(r"\clearpage")

    sections.append(r"\section*{MB-96}")
    for dataset in DATASETS:
        for region in REGIONS:
            for metric in METRICS:
                table = build_table(
                    models, dataset, region, metric, source="repro", use_internal=True
                )
                if table:
                    sections.append(table)
    sections.append(r"\clearpage")

    tables_path = os.path.join(OUTPUT_DIR, "tables.tex")
    with open(tables_path, "w") as f:
        f.write("\n".join(sections))

    main_tex = r"""\documentclass[10pt]{article}
\usepackage[margin=1.2cm]{geometry}
\usepackage{array}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{makecell}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage[table]{xcolor}
\usepackage{tikz}
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
