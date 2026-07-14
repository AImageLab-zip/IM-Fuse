#!/usr/bin/env python3
"""Turn `reproduction-comparison.xlsx` into LaTeX comparison tables.

Reads the workbook produced by `build_reproduction_comparison.py` (one sheet
per model, each holding a BRATS2018 and a BRATS2023 table) and emits, for
every (dataset, region, metric) combination, a `table*` with models as rows
and modality-presence combinations as columns. Each cell stacks the legacy
value on top of the reproduced mean +/- std (Dice tables) or shows just the
reproduced mean +/- std (HD95 tables, since the legacy workbook has no HD95
numbers).

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
MODALITY_COLS = ["Fl", "T1", "T1c", "T2"]


def modality_label(mkey: tuple) -> str:
    return "+".join(name for name, present in zip(MODALITY_COLS, mkey) if present)


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

            # Within each group, find the Legacy / Mean / Std sub-columns.
            group_cols = {}  # (metric, region) -> {'Legacy': c, 'Mean': c, 'Std': c}
            for c, cell in enumerate(sub_row):
                if c < 4 or c not in col_group:
                    continue
                label = str(cell).strip() if cell is not None else ""
                if label in ("Legacy", "Mean", "Std"):
                    group_cols.setdefault(col_group[c], {})[label] = c

            # Data rows until a blank row.
            r = i + 3
            data = {region: {metric: [] for metric in METRICS} for region in REGIONS}
            while r < len(rows) and any(v is not None for v in rows[r]):
                drow = rows[r]
                mkey = tuple(bool(drow[k]) for k in range(4))
                for (metric, region), cols in group_cols.items():
                    legacy = drow[cols["Legacy"]] if "Legacy" in cols else None
                    mean = drow[cols["Mean"]] if "Mean" in cols else None
                    std = drow[cols["Std"]] if "Std" in cols else None
                    data[region][metric].append((mkey, legacy, mean, std))
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
        models[name] = parse_sheet(wb[name])
    return models


# --------------------------------------------------------------------------- #
# LaTeX rendering
# --------------------------------------------------------------------------- #
def fmt_num(v):
    return f"{v:.1f}" if isinstance(v, (int, float)) else None


def cell_text(legacy, mean, std, metric):
    legacy_s = fmt_num(legacy)
    mean_s = fmt_num(mean)
    std_s = fmt_num(std)
    repro_s = f"{mean_s}$\\pm${std_s}" if mean_s is not None and std_s is not None else (
        mean_s if mean_s is not None else None
    )

    if metric == "Dice":
        if legacy_s is None and repro_s is None:
            return "--"
        if legacy_s is None:
            return f"\\makecell{{-- \\\\ {repro_s}}}"
        if repro_s is None:
            return f"\\makecell{{{legacy_s} \\\\ --}}"
        return f"\\makecell{{{legacy_s} \\\\ {repro_s}}}"

    # HD95: legacy is never available, show reproduced value only.
    return repro_s if repro_s is not None else "--"


def build_table(models, dataset, region, metric):
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
            _, legacy, mean, std = by_key.get(mkey, (mkey, None, None, None))
            cells.append(cell_text(legacy, mean, std, metric))
            if isinstance(legacy, (int, float)):
                legacies.append(legacy)
            if isinstance(mean, (int, float)):
                means.append(mean)

        avg_legacy = sum(legacies) / len(legacies) if legacies else None
        avg_mean = sum(means) / len(means) if means else None
        avg_cell = cell_text(avg_legacy, avg_mean, None, metric)

        row = " & ".join([tex_escape(name)] + cells + [avg_cell])
        rows_tex.append(row + r" \\")

    if mkeys is None:
        return ""

    ncols = len(mkeys) + 1
    header_mods = " & ".join(f"\\textbf{{{modality_label(k)}}}" for k in mkeys)

    caption = (
        f"Reproduction comparison ({metric}\\%) on the {region} region, "
        f"{dataset} dataset. Each cell shows the legacy value and, below it, "
        f"the mean $\\pm$ std over the reproduced folds."
        if metric == "Dice"
        else (
            f"Reproduced {metric} on the {region} region, {dataset} dataset "
            "(mean $\\pm$ std over folds; no legacy reference available for this metric)."
        )
    )
    label = f"tab:{dataset.lower()}_{region.lower()}_{metric.lower()}"

    lines = [
        r"\begin{table*}[!ht]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begingroup",
        r"\renewcommand{\arraystretch}{1.3}",
        r"\setlength{\tabcolsep}{4pt}",
        f"\\begin{{tabular}}{{l{'c' * ncols}}}",
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
                table = build_table(models, dataset, region, metric)
                if table:
                    sections.append(table)
        sections.append(r"\clearpage")

    tables_path = os.path.join(OUTPUT_DIR, "tables.tex")
    with open(tables_path, "w") as f:
        f.write("\n".join(sections))

    main_tex = r"""\documentclass[10pt]{article}
\usepackage[margin=1.2cm]{geometry}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{makecell}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{xcolor}
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
