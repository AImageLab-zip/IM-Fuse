#!/usr/bin/env python3
"""Single compact performance table combining GPU and CPU `mimose flops`
results: one row per model with parameter count, FLOPs, GPU peak memory, and
GPU/CPU whole-volume latency side by side.

Unlike generate_flops_table.py / generate_flops_table_cpu.py (which each
report patch/tile AND whole-volume costs, GPU-only or CPU-only), this script
reports only the whole-volume cost (the end-to-end inference figure that
matters for a top-line comparison table) and merges the GPU (`flops.csv`,
written by `mimose flops --device cuda`) and CPU (`flops_cpu.csv`, written
by `mimose flops --device cpu`) sibling CSVs per model into one row. FLOPs
and params are read from whichever of the two is present (ptflops' op count
is architecture/input-shape derived, not hardware-dependent, so GPU and CPU
runs report the same figure -- GPU is preferred when both exist purely for
consistency). Peak memory has no CPU equivalent (mimose.flops never measures
it there -- see generate_flops_table_cpu.py's docstring) so that column is
GPU-only. A model missing one side's CSV, or a measurement that side never
produces, still gets a row, with "n/a" (NOT_SUPPORTED) in the columns that
side would have filled.

All FLOPs are reported in a fixed TFLOPs unit (unlike generate_flops_table's
auto T/G/M/K scaling) and latency in a fixed-seconds unit, so columns stay
directly comparable across every row without per-cell unit switching.

Reuses generate_flops_table.py's discovery/parsing (discover_flops_csvs,
load_scoped_rows, DISPLAY_NAMES, PAPER_TABLE1_MODELS, tex_escape,
ranked_indices, compile_pdf) rather than re-deriving that pipeline.

Model order is alphabetical (case-insensitive), and TinyMimosa/ManyMimosas
are dropped from every table (unrestricted and restricted alike) -- same
convention this session applied to generate_flops_table_cpu.py and
modality_contribution.py.

Writes `tables.tex` (all discovered models) and `tables_restricted.tex`
(paper Table 1 (tab:models) models only) plus a standalone
preview main.tex into OUTPUT_DIR, compiling it to PDF with tectonic (if
available), and prints the same data as rich console tables.
"""

from __future__ import annotations

import os
import sys

from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_flops_table as gft  # noqa: E402

RESULTS_DIR = gft.RESULTS_DIR
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "performance_table"
)

# Dropped from every table, restricted or not.
EXCLUDED_DISPLAY_NAMES = {"TinyMimosa", "ManyMimosas"}


def collect_performance_rows(
    results_dir: str, *, allowed_models: set[str] | None = None
) -> list[tuple[str, dict[str, float | None]]]:
    """[(display_name, {"params", "flops", "mem_gpu", "lat_gpu", "lat_cpu"}), ...],
    one row per model_norm found in either flops.csv or flops_cpu.csv's
    whole_volume scope (a model missing one side just gets None there),
    sorted case-insensitively by display name."""
    gpu_csvs = gft.discover_flops_csvs(results_dir, filename="flops.csv")
    cpu_csvs = gft.discover_flops_csvs(results_dir, filename="flops_cpu.csv")

    rows = []
    for model_norm in sorted(set(gpu_csvs) | set(cpu_csvs)):
        if allowed_models is not None and model_norm not in allowed_models:
            continue
        gpu_whole = gft.load_scoped_rows(gpu_csvs[model_norm]).get("whole_volume") if model_norm in gpu_csvs else None
        cpu_whole = gft.load_scoped_rows(cpu_csvs[model_norm]).get("whole_volume") if model_norm in cpu_csvs else None
        if gpu_whole is None and cpu_whole is None:
            print(f"{model_norm}: no whole_volume row in flops.csv or flops_cpu.csv, skipping", file=sys.stderr)
            continue

        reference = gpu_whole or cpu_whole
        flops_row = gpu_whole or cpu_whole
        display = gft.DISPLAY_NAMES.get(model_norm, model_norm.upper())
        if display in EXCLUDED_DISPLAY_NAMES:
            continue

        rows.append(
            (
                display,
                {
                    "params": float(reference["params"]),
                    "flops": float(flops_row["mean_flops"]) if flops_row else None,
                    "mem_gpu": float(gpu_whole["mean_peak_memory_bytes"])
                    if gpu_whole and gpu_whole.get("mean_peak_memory_bytes")
                    else None,
                    "lat_gpu": float(gpu_whole["mean_latency_seconds"])
                    if gpu_whole and gpu_whole.get("mean_latency_seconds")
                    else None,
                    "lat_cpu": float(cpu_whole["mean_latency_seconds"])
                    if cpu_whole and cpu_whole.get("mean_latency_seconds")
                    else None,
                },
            )
        )
    rows.sort(key=lambda r: r[0].lower())
    return rows


# Rendered for any cell whose underlying measurement is absent (model
# missing that side's CSV, or a measurement that side never produces --
# e.g. CPU peak memory is never measured, see module docstring) -- distinct
# from a bare "-", so it's unambiguous that the score wasn't just omitted.
NOT_SUPPORTED = "n/a"


def fmt_params(value: float | None) -> str:
    formatted = gft.fmt_count(value)
    return NOT_SUPPORTED if formatted == "-" else formatted


def fmt_tflops(value: float | None) -> str:
    return f"{value / 1e12:,.3f}" if value is not None else NOT_SUPPORTED


def fmt_mib(value: float | None) -> str:
    return f"{value / (1024 ** 2):,.0f}" if value is not None else NOT_SUPPORTED


def fmt_seconds(value: float | None) -> str:
    return f"{value:,.2f}" if value is not None else NOT_SUPPORTED


def print_console_table(rows: list[tuple[str, dict[str, float | None]]], *, title_suffix: str = "") -> None:
    table = Table(title="Performance summary — whole-volume forward, GPU + CPU" + title_suffix)
    table.add_column("Model", style="bold cyan")
    table.add_column("Params", justify="right")
    table.add_column("FLOPs [T]", justify="right")
    table.add_column("Mem. [MiB]", justify="right")
    table.add_column("Lat. [s] GPU", justify="right")
    table.add_column("Lat. [s] CPU", justify="right")

    for display, m in rows:
        table.add_row(
            display,
            fmt_params(m["params"]),
            fmt_tflops(m["flops"]),
            fmt_mib(m["mem_gpu"]),
            fmt_seconds(m["lat_gpu"]),
            fmt_seconds(m["lat_cpu"]),
        )
    Console().print(table)


def build_latex_table(
    rows: list[tuple[str, dict[str, float | None]]], *, label: str = "tab:performance", restricted: bool = False
) -> str:
    if not rows:
        return ""

    params = [r[1]["params"] for r in rows]
    flops = [r[1]["flops"] for r in rows]
    mem = [r[1]["mem_gpu"] for r in rows]
    lat_gpu = [r[1]["lat_gpu"] for r in rows]
    lat_cpu = [r[1]["lat_cpu"] for r in rows]

    best_params, second_params = gft.ranked_indices(params)
    best_flops, second_flops = gft.ranked_indices(flops)
    best_mem, second_mem = gft.ranked_indices(mem)
    best_lat_gpu, second_lat_gpu = gft.ranked_indices(lat_gpu)
    best_lat_cpu, second_lat_cpu = gft.ranked_indices(lat_cpu)

    def cell(value_text: str, i: int, best: int | None, second: int | None) -> str:
        if value_text == NOT_SUPPORTED:
            return value_text
        if i == best:
            return f"\\textbf{{{value_text}}}"
        if i == second:
            return f"\\underline{{{value_text}}}"
        return value_text

    body = []
    for i, (display, m) in enumerate(rows):
        cells = [
            gft.tex_escape(display),
            cell(fmt_params(params[i]), i, best_params, second_params),
            cell(fmt_tflops(flops[i]), i, best_flops, second_flops),
            cell(fmt_mib(mem[i]), i, best_mem, second_mem),
            cell(fmt_seconds(lat_gpu[i]), i, best_lat_gpu, second_lat_gpu),
            cell(fmt_seconds(lat_cpu[i]), i, best_lat_cpu, second_lat_cpu),
        ]
        body.append(" & ".join(cells) + r" \\")

    scope_note = (
        r" Restricted to the 18 models in the paper's Table~1 (\texttt{tab:models})."
        if restricted
        else ""
    )
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Per-model computational cost on the whole input volume "
        r"(aggregated via sliding-window inference for patch-based models; "
        r"processed directly for models that consume the whole volume in "
        r"one shot), all four modalities present: parameter count, FLOPs, "
        r"GPU peak memory, and GPU/CPU wall-clock latency. Bold marks the "
        r"most efficient model per column, underline the runner-up. "
        r"`n/a' marks models that do not support CPU inference."
        f"{scope_note}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\rowcolors{3}{gray!8}{white}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Model} & "
        r"\rotatebox{90}{\footnotesize\textbf{Params}} & "
        r"\rotatebox{90}{\footnotesize\textbf{FLOPs [T]}} & "
        r"\rotatebox{90}{\footnotesize\textbf{Mem. [MiB]}} & "
        r"\rotatebox{90}{\footnotesize\textbf{GPU [s]}} & "
        r"\rotatebox{90}{\footnotesize\textbf{CPU [s]}} \\",
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

    rows = collect_performance_rows(RESULTS_DIR)
    if not rows:
        print(f"No flops.csv/flops_cpu.csv files found under {RESULTS_DIR}/<model>_23/", file=sys.stderr)
        return

    restricted_rows = collect_performance_rows(RESULTS_DIR, allowed_models=gft.PAPER_TABLE1_MODELS)

    print_console_table(rows)
    if restricted_rows:
        print_console_table(restricted_rows, title_suffix=" (paper Table 1 only)")
    else:
        print("No paper Table 1 models found under RESULTS_DIR; skipping restricted table", file=sys.stderr)

    tables_path = os.path.join(OUTPUT_DIR, "tables.tex")
    with open(tables_path, "w") as f:
        f.write(build_latex_table(rows))

    restricted_path = os.path.join(OUTPUT_DIR, "tables_restricted.tex")
    with open(restricted_path, "w") as f:
        f.write(build_latex_table(restricted_rows, label="tab:performance_restricted", restricted=True))

    main_tex = r"""\documentclass[12pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{booktabs}
\usepackage[table]{xcolor}
\usepackage{lmodern}
\usepackage{graphicx}
\pagestyle{plain}
\begin{document}
\input{tables.tex}
\clearpage
\input{tables_restricted.tex}
\end{document}
"""
    main_path = os.path.join(OUTPUT_DIR, "main.tex")
    with open(main_path, "w") as f:
        f.write(main_tex)

    print(f"Wrote {tables_path}")
    print(f"Wrote {restricted_path}")
    print(f"Wrote {main_path}")

    gft.compile_pdf(main_path)


if __name__ == "__main__":
    main()
