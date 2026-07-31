#!/usr/bin/env python3
"""Emit a single LaTeX table -- one row per reproduced model -- listing its
most important training hyperparameters.

"Reproduced model" reuses generate_latex_tables.py's own definition: every
sheet in reproduction-comparison.xlsx whose normalized name is not in its
EXCLUDED_MODELS set. For each one, reads the packaged config it was actually
trained with (src/mimose/data/configs/<model>_<dataset>.yaml -- preferring
brats18, falling back to brats23/brats25 for models without one) and pulls
the hyperparameter values out of the raw YAML text (regex, not a full
parse, so values like 0.00003 are shown exactly as authored instead of
Python's float repr mangling them into scientific notation).

Momentum/Nesterov only apply to SGD and Betas only to adaptive optimizers, so
those columns show "--" for models using the other family.

Writes hyperparams.tex + hyperparams_main.tex into the same OUTPUT_DIR as
generate_latex_tables.py, then compiles hyperparams_main.tex to PDF.
"""

from __future__ import annotations

import os
import re
import sys

import openpyxl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_latex_tables as glt  # noqa: E402 (reuse WORKBOOK/EXCLUDED_MODELS/normalize_name/compile_pdf)

CONFIGS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src", "mimose", "data", "configs"
)
OUTPUT_DIR = glt.OUTPUT_DIR

DATASET_PREFERENCE = ["18", "23", "25"]

FIELDS = [
    ("Optimizer", "optimizer"),
    ("Learning rate", "lr"),
    ("Batch size", "batch_size"),
    ("Weight decay", "weight_decay"),
    ("Scheduler", "scheduler"),
    ("Epochs", "num_epochs"),
]

SGD_FIELDS = [("Momentum", "momentum"), ("Nesterov", "nesterov")]
ADAPTIVE_OPTIMIZERS = {"adam", "adamw", "radam"}


def tex_escape(s: str) -> str:
    return re.sub(r"([&%$#_{}])", r"\\\1", s)


def scalar(text: str, key: str) -> str | None:
    m = re.search(rf"(?m)^{re.escape(key)}: *(.+?) *$", text)
    if not m or m.group(1).strip() == "null":
        return None
    return m.group(1).strip()


def reproduced_models() -> list[str]:
    wb = openpyxl.load_workbook(glt.WORKBOOK, read_only=True)
    names = [n for n in wb.sheetnames if glt.normalize_name(n) not in glt.EXCLUDED_MODELS]
    return sorted(names, key=str.lower)


def find_config(display_name: str) -> tuple[str, str] | tuple[None, None]:
    norm = glt.normalize_name(display_name)
    for dsnum in DATASET_PREFERENCE:
        path = os.path.join(CONFIGS_DIR, f"{norm}_{dsnum}.yaml")
        if os.path.isfile(path):
            return path, dsnum
    return None, None


def load_hparams(path: str) -> dict:
    text = open(path).read()
    hparams = {}
    for label, key in FIELDS:
        value = scalar(text, key)
        if value is not None:
            hparams[label] = value

    if "Learning rate" in hparams and "Batch size" in hparams:
        try:
            per_sample_lr = float(hparams["Learning rate"]) / float(hparams["Batch size"])
        except ValueError:
            pass
        else:
            # Rebuild in place so the row keeps Learning rate's original
            # position instead of jumping to the end of the table.
            hparams = {
                ("Learning rate / batch size" if k == "Learning rate" else k): (
                    f"{per_sample_lr:.10g}" if k == "Learning rate" else v
                )
                for k, v in hparams.items()
                if k != "Batch size"
            }
    hparams.pop("Batch size", None)

    optimizer = hparams.get("Optimizer", "").strip("'\"").lower()
    if optimizer == "sgd":
        for label, key in SGD_FIELDS:
            value = scalar(text, key)
            if value is not None:
                hparams[label] = value
    elif optimizer in ADAPTIVE_OPTIMIZERS:
        value = scalar(text, "betas")
        if value is not None:
            hparams["Betas"] = value

    patch_size = re.search(r"(?m)^ +patch_size: *(.+?) *$", text)
    if patch_size and patch_size.group(1).strip() != "null":
        hparams["Patch size"] = patch_size.group(1).strip()

    return hparams


COLUMNS = [
    "Optimizer",
    "Learning rate / batch size",
    "Weight decay",
    "Scheduler",
    "Epochs",
    "Momentum",
    "Nesterov",
    "Betas",
    "Patch size",
]


def build_table(rows: list[tuple[str, str, dict]]) -> str:
    header = " & ".join([r"\textbf{Model}", r"\textbf{Config}"] + [f"\\textbf{{{c}}}" for c in COLUMNS])
    body = []
    for display_name, dsnum, hparams in rows:
        cells = [tex_escape(display_name), f"BraTS{dsnum}"]
        cells += [tex_escape(hparams.get(c, "--")) for c in COLUMNS]
        body.append(" & ".join(cells) + r" \\")

    # A thin gray rule between every column, matching generate_latex_tables.py's
    # comparison tables so the two report sections read as one system.
    thin_rule = "!{\\color{gray!35}\\vrule width 0.4pt}"
    colspec = "l" + f"{thin_rule}c" * (1 + len(COLUMNS))
    return "\n".join(
        [
            r"\begin{table*}[!ht]",
            r"\centering",
            r"\caption{Training hyperparameters for every reproduced model.}",
            r"\label{tab:hparams}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begingroup",
            r"\renewcommand{\arraystretch}{1.3}",
            r"\setlength{\tabcolsep}{4pt}",
            r"\rowcolors{2}{white}{gray!6}",
            f"\\begin{{tabular}}{{{colspec}}}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\endgroup",
            r"}",
            r"\end{table*}",
            "",
        ]
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []
    missing = []
    for name in reproduced_models():
        path, dsnum = find_config(name)
        if path is None:
            missing.append(name)
            continue
        rows.append((name, dsnum, load_hparams(path)))

    tables_path = os.path.join(OUTPUT_DIR, "hyperparams.tex")
    with open(tables_path, "w") as f:
        f.write(build_table(rows))

    main_tex = r"""\documentclass[10pt]{article}
\usepackage[margin=1.2cm]{geometry}
\usepackage{array}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[table]{xcolor}
\usepackage{lmodern}
\pagestyle{plain}
\begin{document}
\input{hyperparams.tex}
\end{document}
"""
    main_path = os.path.join(OUTPUT_DIR, "hyperparams_main.tex")
    with open(main_path, "w") as f:
        f.write(main_tex)

    print(f"Wrote {tables_path}")
    print(f"Wrote {main_path}")
    if missing:
        print(f"No config found for: {', '.join(missing)}", file=sys.stderr)

    glt.compile_pdf(main_path)


if __name__ == "__main__":
    main()
