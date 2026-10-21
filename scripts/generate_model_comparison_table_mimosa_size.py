#!/usr/bin/env python3
"""Same accuracy-vs-cost comparison table as generate_model_comparison_table.py,
but for the mimosa_[size] family (mimosa_base, mimosa_large, mimosa_tiny, ...)
instead of the paper's 18 literature baselines.

Reuses all discovery/aggregation/rendering logic from
generate_model_comparison_table.py, swapping the model set and injecting
display names for the mimosa_[size] models (absent from
generate_flops_table.py's DISPLAY_NAMES, which only covers the literature
baselines plus TinyMimosa/ManyMimosas).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_flops_table as gft  # noqa: E402
import generate_model_comparison_table as gmc  # noqa: E402
import generate_summary_tables as gst  # noqa: E402

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "model_comparison_table_mimosa_size"
)

MODEL_SET = gst.MIMOSA_SIZE_MODELS

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


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gft.DISPLAY_NAMES = {**gft.DISPLAY_NAMES, **FALLBACK_DISPLAY_NAMES}

    # mimosa_[size] flops runs are written as flops_gpu.csv/.txt (see
    # sbatch_files/flops/flops_mimosa_size_gpu.sh), not the flops.csv used by
    # the literature baselines. No CPU run exists yet for this family, so
    # cpu_filename is left at its default and every lat_cpu cell shows n/a.
    rows = gmc.build_rows(MODEL_SET, gpu_filename="flops_gpu.csv")
    if not rows:
        print("No data found for the mimosa_[size] models", file=sys.stderr)
        return

    # Ascending by FLOPs (rather than gmc.build_rows' alphabetical order),
    # since this family is naturally ordered by compute cost.
    rows.sort(key=lambda r: (r[1]["flops"] is None, r[1]["flops"]))

    gmc.print_console_table(rows)

    tables_path = os.path.join(OUTPUT_DIR, "tables.tex")
    with open(tables_path, "w") as f:
        f.write(
            gmc.build_latex_table(
                rows,
                label="tab:model_comparison_mimosa_size",
                scope="the mimosa\\_[size] models",
            )
        )

    main_tex = r"""\documentclass[12pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{booktabs}
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

    gft.compile_pdf(main_path)


if __name__ == "__main__":
    main()
