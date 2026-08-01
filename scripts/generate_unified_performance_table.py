#!/usr/bin/env python3
"""One performance table (params, FLOPs, GPU peak memory, GPU/CPU whole-volume
latency) covering both families in a single figure: the paper's 18 literature
baselines (generate_performance_table.py's restricted set) first, then --
visually separated by a second \\midrule -- the 8 mimosa_[size] models
(generate_model_comparison_table_mimosa_size.py's family), ordered ascending
by FLOPs within their own block.

Reuses collect_performance_rows/fmt_*/NOT_SUPPORTED from
generate_performance_table.py (gpt) for both families -- only the on-disk
flops filename differs (mimosa_[size] writes flops_gpu.csv/flops_cpu.csv
instead of flops.csv/flops_cpu.csv, see flops_mimosa_size_gpu.sh and the
flops_mimosa_<size>_cpu.sh sbatch scripts) -- and ranked_indices/tex_escape
from generate_flops_table.py (gft). Best/runner-up are ranked globally across
all 26 rows, not per block, since this is meant to read as one table.
"""

from __future__ import annotations

import os
import sys

from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_flops_table as gft  # noqa: E402
import generate_performance_table as gpt  # noqa: E402
import generate_summary_tables as gst  # noqa: E402

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "unified_performance_table"
)

MIMOSA_SIZE_DISPLAY_NAMES = {
    "mimosabase": "MiMoSe-Base",
    "mimosagargantuan": "MiMoSe-Gargantuan",
    "mimosahuge": "MiMoSe-Huge",
    "mimosalarge": "MiMoSe-Large",
    "mimosamedium": "MiMoSe-Medium",
    "mimosamicro": "MiMoSe-Micro",
    "mimosasmall": "MiMoSe-Small",
    "mimosatiny": "MiMoSe-Tiny",
}


def collect_unified_rows() -> tuple[list[tuple[str, dict]], int]:
    """(rows, split_index): rows is the 18 literature-baseline rows (paper
    Table 1, alphabetical) followed by the 8 mimosa_[size] rows (ascending by
    FLOPs); split_index is len(literature rows), i.e. where the group
    \\midrule goes."""
    literature_rows = gpt.collect_performance_rows(gpt.RESULTS_DIR, allowed_models=gft.PAPER_TABLE1_MODELS)

    gft.DISPLAY_NAMES = {**gft.DISPLAY_NAMES, **MIMOSA_SIZE_DISPLAY_NAMES}
    mimosa_size_rows = gpt.collect_performance_rows(
        gpt.RESULTS_DIR,
        allowed_models=gst.MIMOSA_SIZE_MODELS,
        gpu_filename="flops_gpu.csv",
        cpu_filename="flops_cpu.csv",
    )
    mimosa_size_rows.sort(key=lambda r: (r[1]["flops"] is None, r[1]["flops"]))

    return literature_rows + mimosa_size_rows, len(literature_rows)


def print_console_table(rows: list[tuple[str, dict]], split_index: int) -> None:
    table = Table(title="Unified performance summary — whole-volume forward, GPU + CPU")
    table.add_column("Model", style="bold cyan")
    table.add_column("Params", justify="right")
    table.add_column("FLOPs [T]", justify="right")
    table.add_column("Mem. [MiB]", justify="right")
    table.add_column("Lat. [s] GPU", justify="right")
    table.add_column("Lat. [s] CPU", justify="right")

    for i, (display, m) in enumerate(rows):
        if i == split_index:
            table.add_section()
        table.add_row(
            display,
            gpt.fmt_params(m["params"]),
            gpt.fmt_tflops(m["flops"]),
            gpt.fmt_mib(m["mem_gpu"]),
            gpt.fmt_seconds(m["lat_gpu"]),
            gpt.fmt_seconds(m["lat_cpu"]),
        )
    Console().print(table)


def build_latex_table(rows: list[tuple[str, dict]], split_index: int, *, label: str = "tab:performance_unified") -> str:
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
        if value_text == gpt.NOT_SUPPORTED:
            return value_text
        if i == best:
            return f"\\textbf{{{value_text}}}"
        if i == second:
            return f"\\underline{{{value_text}}}"
        return value_text

    body = []
    for i, (display, m) in enumerate(rows):
        if i == split_index:
            body.append(r"\midrule")
        cells = [
            gft.tex_escape(display),
            cell(gpt.fmt_params(params[i]), i, best_params, second_params),
            cell(gpt.fmt_tflops(flops[i]), i, best_flops, second_flops),
            cell(gpt.fmt_mib(mem[i]), i, best_mem, second_mem),
            cell(gpt.fmt_seconds(lat_gpu[i]), i, best_lat_gpu, second_lat_gpu),
            cell(gpt.fmt_seconds(lat_cpu[i]), i, best_lat_cpu, second_lat_cpu),
        ]
        body.append(" & ".join(cells) + r" \\")

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Per-model computational cost on the whole input volume "
        r"(aggregated via sliding-window inference for patch-based models; "
        r"processed directly for models that consume the whole volume in "
        r"one shot), all four modalities present: parameter count, FLOPs, "
        r"GPU peak memory, and GPU/CPU wall-clock latency. Bold marks the "
        r"most efficient model per column, underline the runner-up, ranked "
        r"across the whole table. Top block: the paper's 18 literature "
        r"baselines (Table~1, \texttt{tab:models}). Bottom block: the "
        r"mimosa\_[size] family, ordered by ascending FLOPs. "
        r"`n/a' marks models/measurements that do not exist.}",
        f"\\label{{{label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
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

    rows, split_index = collect_unified_rows()
    if not rows:
        print(f"No flops data found under {gpt.RESULTS_DIR}", file=sys.stderr)
        return

    print_console_table(rows, split_index)

    tables_path = os.path.join(OUTPUT_DIR, "tables.tex")
    with open(tables_path, "w") as f:
        f.write(build_latex_table(rows, split_index))

    main_tex = r"""\documentclass[12pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{booktabs}
\usepackage[table]{xcolor}
\usepackage{lmodern}
\usepackage{graphicx}
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
