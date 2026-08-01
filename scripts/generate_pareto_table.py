#!/usr/bin/env python3
"""Pareto-efficiency table across every model with complete accuracy +
efficiency data: the 18 literature baselines (see
generate_model_comparison_table.py) plus all 8 mimosa_[size] variants (see
generate_model_comparison_table_mimosa_size.py), together.

Model A "dominates" model B if A is at least as good as B on every one of
the 5 objectives below, and strictly better on at least one -- i.e. there is
no reason to ever prefer B over A. A model is "Pareto-efficient" (on the
frontier) iff nothing dominates it: every other model that beats it on some
objective is worse than it on another, so picking it is a genuine
accuracy/cost trade-off rather than a strictly worse choice.

Objectives (5): Dice (maximize), HD95, GPU latency, peak GPU memory, FLOPs
(all minimize). CPU latency is shown for reference but excluded from the
dominance computation, since it's "n/a" for IM-Fuse and for every
mimosa_[size] model (no CPU flops run exists for that family yet) --
including it would make those models incomparable on that axis.

Reuses generate_model_comparison_table.py's build_rows (accuracy + cost data
collection) for both families rather than re-deriving it.
"""

from __future__ import annotations

import os
import sys

from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_flops_table as gft  # noqa: E402
import generate_model_comparison_table as gmc  # noqa: E402
import generate_model_comparison_table_mimosa_size as gmcm  # noqa: E402
import generate_performance_table as gpt  # noqa: E402
import generate_summary_tables as gst  # noqa: E402

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "pareto_table"
)

# (metric key, maximize?) -- the 5 dominance objectives, in display order.
OBJECTIVES = [
    ("dice", True),
    ("hd95", False),
    ("lat_gpu", False),
    ("mem", False),
    ("flops", False),
]

INSUFFICIENT_DATA = "insufficient data"


def collect_all_rows() -> list[tuple[str, str, dict[str, float | None]]]:
    """[(display_name, family, metrics_dict), ...] for both families."""
    rows = []
    for display, m in gmc.build_rows(gmc.MODEL_SET):
        rows.append((display, "Baseline", m))

    gft.DISPLAY_NAMES = {**gft.DISPLAY_NAMES, **gmcm.FALLBACK_DISPLAY_NAMES}
    for display, m in gmc.build_rows(gst.MIMOSA_SIZE_MODELS, gpu_filename="flops_gpu.csv"):
        rows.append((display, "MimosaSize", m))

    return rows


def has_all_objectives(m: dict[str, float | None]) -> bool:
    return all(m.get(key) is not None for key, _maximize in OBJECTIVES)


def dominates(a: dict[str, float | None], b: dict[str, float | None]) -> bool:
    """True iff a dominates b: at least as good as b on every objective, and
    strictly better on at least one. Assumes both have all objectives."""
    at_least_as_good = True
    strictly_better = False
    for key, maximize in OBJECTIVES:
        x, y = a[key], b[key]
        if maximize:
            if x < y:
                at_least_as_good = False
            if x > y:
                strictly_better = True
        else:
            if x > y:
                at_least_as_good = False
            if x < y:
                strictly_better = True
    return at_least_as_good and strictly_better


def compute_pareto(
    rows: list[tuple[str, str, dict[str, float | None]]],
) -> dict[str, list[str]]:
    """{display_name: [dominator_display_names]}. Empty list -> Pareto-efficient.
    Models lacking any objective are omitted from this dict entirely (they
    get "insufficient data" in the table, and take no part in dominance)."""
    complete = [(display, m) for display, _family, m in rows if has_all_objectives(m)]
    dominators = {}
    for display, m in complete:
        dominators[display] = [
            other_display
            for other_display, other_m in complete
            if other_display != display and dominates(other_m, m)
        ]
    return dominators


def print_console_table(
    rows: list[tuple[str, str, dict[str, float | None]]], dominators: dict[str, list[str]]
) -> None:
    table = Table(title="Pareto efficiency — accuracy vs. computational cost")
    table.add_column("Model", style="bold cyan")
    table.add_column("Family")
    table.add_column("Dice [%]", justify="right")
    table.add_column("HD95 [mm]", justify="right")
    table.add_column("Lat. [s] GPU", justify="right")
    table.add_column("Lat. [s] CPU", justify="right")
    table.add_column("Mem. [MiB]", justify="right")
    table.add_column("FLOPs [T]", justify="right")
    table.add_column("Pareto?")
    table.add_column("Dominated by")

    for display, family, m in rows:
        if display not in dominators:
            pareto, dominated_by = INSUFFICIENT_DATA, "--"
        else:
            direct = dominators[display]
            pareto = "Yes" if not direct else "No"
            dominated_by = ", ".join(direct) if direct else "--"
        table.add_row(
            display,
            family,
            gmc.fmt_pct(m["dice"]),
            gmc.fmt_mm(m["hd95"]),
            gpt.fmt_seconds(m["lat_gpu"]),
            gpt.fmt_seconds(m["lat_cpu"]),
            gpt.fmt_mib(m["mem"]),
            gpt.fmt_tflops(m["flops"]),
            pareto,
            dominated_by,
        )
    Console().print(table)


def build_latex_table(
    rows: list[tuple[str, str, dict[str, float | None]]],
    dominators: dict[str, list[str]],
    *,
    label: str = "tab:pareto",
) -> str:
    if not rows:
        return ""

    body = []
    for display, family, m in rows:
        if display not in dominators:
            pareto, dominated_by = INSUFFICIENT_DATA, "--"
        else:
            direct = dominators[display]
            pareto = "Yes" if not direct else "No"
            dominated_by = ", ".join(gft.tex_escape(d) for d in direct) if direct else "--"
        cells = [
            gft.tex_escape(display),
            family,
            gmc.fmt_pct(m["dice"]),
            gmc.fmt_mm(m["hd95"]),
            gpt.fmt_seconds(m["lat_gpu"]),
            gpt.fmt_seconds(m["lat_cpu"]),
            gpt.fmt_mib(m["mem"]),
            gpt.fmt_tflops(m["flops"]),
            pareto,
            dominated_by,
        ]
        body.append(" & ".join(cells) + r" \\")

    lines = [
        r"\begin{table*}[!ht]",
        r"\centering",
        r"\caption{Pareto efficiency across the 18 literature baselines and the "
        r"8 mimosa\_[size] variants. Model A \emph{dominates} model B if A is "
        r"at least as good as B on every one of Dice, HD95, GPU latency, peak "
        r"GPU memory, and FLOPs, and strictly better on at least one -- i.e.\ "
        r"there is no reason to prefer B. A model is \emph{Pareto-efficient} "
        r"(\textbf{Pareto? = Yes}) iff nothing dominates it. CPU latency is "
        r"shown for reference only and excluded from the dominance test (`n/a' "
        r"for IM-Fuse and every mimosa\_[size] model, no CPU flops run yet for "
        r"that family). `Dominated by' lists every model that directly "
        r"dominates a non-efficient row.}",
        f"\\label{{{label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\rowcolors{3}{gray!8}{white}",
        r"\begin{tabular}{llccccccll}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Family} & \textbf{Dice [\%]} & \textbf{HD95 [mm]} & "
        r"\textbf{Lat.\ GPU [s]} & \textbf{Lat.\ CPU [s]} & \textbf{Mem.\ [MiB]} & "
        r"\textbf{FLOPs [T]} & \textbf{Pareto?} & \textbf{Dominated by} \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = collect_all_rows()
    if not rows:
        print("No model data found for either family", file=sys.stderr)
        return
    rows.sort(key=lambda r: (r[1], r[0].lower()))

    dominators = compute_pareto(rows)

    print_console_table(rows, dominators)

    tables_path = os.path.join(OUTPUT_DIR, "tables.tex")
    with open(tables_path, "w") as f:
        f.write(build_latex_table(rows, dominators))

    main_tex = r"""\documentclass[12pt]{article}
\usepackage[paperwidth=13in, paperheight=9in, margin=0.6in]{geometry}
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

    n_pareto = sum(1 for display, _f, _m in rows if display in dominators and not dominators[display])
    n_complete = len(dominators)
    print(f"Pareto-efficient: {n_pareto}/{n_complete} models with complete data ({len(rows) - n_complete} skipped, insufficient data)")

    gft.compile_pdf(main_path)


if __name__ == "__main__":
    main()
