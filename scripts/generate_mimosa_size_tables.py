#!/usr/bin/env python3
"""Build the same class-specific (per-region) Dice/HD95 summary tables as
generate_summary_tables.py, but restricted to the mimosa_[size] model family
(mimosa_base, mimosa_large, mimosa_tiny, ...) that generate_summary_tables.py
explicitly excludes.

Reuses all discovery/aggregation/rendering logic from generate_summary_tables.py
and just swaps which models pass the row filter.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_summary_tables as gst  # noqa: E402

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "summary_tables_mimosa_size"
)

FALLBACK_DISPLAY_NAMES = {
    "mimosabase": "MiMoSe-Base",
    "mimosagargantuan": "MiMoSe-Gargantuan",
    "mimosahuge": "MiMoSe-Huge",
    "mimosalarge": "MiMoSe-Large",
    "mimosamedium": "MiMoSe-Medium",
    "mimosamicro": "MiMoSe-Micro",
    "mimosasmall": "MiMoSe-Small",
    "mimosatiny": "MiMoSe-Tiny",
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    display_names = {}
    if os.path.exists(gst.LEGACY_FILE):
        _, display_names = gst.brc.parse_legacy(gst.LEGACY_FILE)

    models, internal_models, dsnums = gst.discover(gst.RESULTS_DIR)
    if not dsnums:
        print("No result directories found in", gst.RESULTS_DIR)
        return

    # Invert generate_summary_tables.py's filter: keep only mimosa_[size]
    # models instead of excluding them.
    all_norms = set(models) | set(internal_models)
    gst.EXCLUDED_MODELS = all_norms - gst.MIMOSA_SIZE_MODELS
    gst.FALLBACK_DISPLAY_NAMES = FALLBACK_DISPLAY_NAMES

    rows = gst.build_class_rows(models, internal_models, display_names)
    if not rows:
        print("No mimosa_[size] model has at least one complete group; nothing to write")
        return

    sections = []
    for region in gst.REGIONS:
        caption = (
            f"{gst.REGION_LABELS[region]} ({region}), mimosa\\_[size] variants: mean $\\pm$ std "
            "Dice (\\%) and HD95 (mm) over all modality-presence combinations and folds, per dataset. "
            "BraTS18/BraTS25-pre are the official test splits, averaged over our three folds; "
            "MB-96 - BraTS18 chp/MB-96 - BraTS25-pre chp are the internal cohort, scored with the "
            "BraTS18-trained and BraTS25-pre-trained checkpoints respectively, also averaged over the "
            "three fold checkpoints. Rank is by Dice, among models with data for that group. Bold marks "
            "the best model per column, underline the runner-up."
        )
        table = gst.build_table(rows, region, caption, f"tab:mimosasize:{region.lower()}")
        sections.append(table)

    tables_path = os.path.join(OUTPUT_DIR, "tables.tex")
    with open(tables_path, "w") as f:
        f.write("\n".join(sections))

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
