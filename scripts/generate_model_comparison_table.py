#!/usr/bin/env python3
"""Single comparison table for the paper's 18 baseline models (Table 1 /
tab:models): model name | avg Dice | avg HD95 | GPU latency | CPU latency |
peak memory | FLOPs.

Explicitly excludes TinyMimosa/ManyMimosas (this paper's own MissingBench-arch
lightweight models) and every mimosa_[size] variant -- this is a
like-for-like comparison of the literature baselines only, not this paper's
own contributions.

Reuses:
  - generate_summary_tables.py (gst) for Dice/HD95: discover() + is_complete()
    + aggregate(), averaged over every complete GROUPS entry (BraTS18/
    BraTS25-pre/MB-96 - BraTS18 chp/MB-96 - BraTS25-pre chp) and over WT/TC/ET,
    into one Dice number and one HD95 number per model. Dice and HD95 stay
    separate columns (never merged into a single score), per this project's
    convention.
  - generate_flops_table.py (gft) + generate_performance_table.py (gpt) for
    FLOPs / GPU peak memory / GPU+CPU latency, whole-volume scope only (the
    end-to-end inference figure, not the single patch/tile cost).

Writes tables.tex (LaTeX) plus a standalone main.tex preview into OUTPUT_DIR,
compiled to PDF with tectonic if available, and prints a rich console table.
Bold marks the best model per column (highest Dice, lowest everything else),
underline the runner-up.
"""

from __future__ import annotations

import os
import statistics
import sys

from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_flops_table as gft  # noqa: E402
import generate_performance_table as gpt  # noqa: E402
import generate_summary_tables as gst  # noqa: E402

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "model_comparison_table"
)

# The paper's 18 baseline models: gft.PAPER_TABLE1_MODELS minus TinyMimosa/
# ManyMimosas (this paper's own MissingBench-arch contributions). Never
# includes any mimosa_[size] variant -- those were never part of
# PAPER_TABLE1_MODELS to begin with (see generate_mimosa_size_tables.py /
# build_reproduction_comparison_mimosa_size.py for that family instead).
MODEL_SET = gft.PAPER_TABLE1_MODELS - {"tinymimosa", "manymimosas"}


def collect_dice_hd95(model_set: set[str]) -> dict[str, dict[str, float | None]]:
    """{model_norm: {"dice": avg or None, "hd95": avg or None}}, averaged over
    every complete GROUPS entry (see gst.GROUPS) and over WT/TC/ET."""
    models, internal_models, _dsnums = gst.discover(gst.RESULTS_DIR)
    out = {}
    for model_norm in model_set:
        dice_vals, hd95_vals = [], []
        for _label, source, dsnum in gst.GROUPS:
            source_dict = models if source == "official" else internal_models
            folds = source_dict.get(model_norm, {}).get(dsnum)
            if not folds or not gst.is_complete(folds):
                continue
            stats = gst.aggregate(folds)
            for region in gst.REGIONS:
                dice_vals.append(stats[region]["Dice"][0])
                hd95_vals.append(stats[region]["HD95"][0])
        out[model_norm] = {
            "dice": statistics.fmean(dice_vals) if dice_vals else None,
            "hd95": statistics.fmean(hd95_vals) if hd95_vals else None,
        }
    return out


def build_rows(
    model_set: set[str],
    *,
    gpu_filename: str = "flops.csv",
    cpu_filename: str = "flops_cpu.csv",
) -> list[tuple[str, dict[str, float | None]]]:
    """[(display_name, {"dice", "hd95", "lat_gpu", "lat_cpu", "mem", "flops"}), ...],
    sorted case-insensitively by display name. gpu_filename/cpu_filename are
    forwarded to gpt.collect_performance_rows (see its docstring)."""
    dice_hd95 = collect_dice_hd95(model_set)
    perf_rows = gpt.collect_performance_rows(
        gpt.RESULTS_DIR, allowed_models=model_set, gpu_filename=gpu_filename, cpu_filename=cpu_filename
    )

    rows = []
    for display, perf in perf_rows:
        model_norm = next(n for n in model_set if gft.DISPLAY_NAMES.get(n, n.upper()) == display)
        dh = dice_hd95.get(model_norm, {})
        rows.append(
            (
                display,
                {
                    "dice": dh.get("dice"),
                    "hd95": dh.get("hd95"),
                    "lat_gpu": perf["lat_gpu"],
                    "lat_cpu": perf["lat_cpu"],
                    "mem": perf["mem_gpu"],
                    "flops": perf["flops"],
                },
            )
        )
    rows.sort(key=lambda r: r[0].lower())
    return rows


def ranked_indices_max(values: list[float | None]) -> tuple[int | None, int | None]:
    """(index of largest, index of second-largest) among non-None values."""
    available = [(i, v) for i, v in enumerate(values) if v is not None]
    order = sorted(available, key=lambda t: -t[1])
    best = order[0][0] if order else None
    runner_up = order[1][0] if len(order) > 1 else None
    return best, runner_up


def fmt_pct(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else gpt.NOT_SUPPORTED


def fmt_mm(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else gpt.NOT_SUPPORTED


def print_console_table(rows: list[tuple[str, dict[str, float | None]]]) -> None:
    table = Table(title="Model comparison — accuracy vs. computational cost")
    table.add_column("Model", style="bold cyan")
    table.add_column("Dice [%]", justify="right")
    table.add_column("HD95 [mm]", justify="right")
    table.add_column("Lat. [s] GPU", justify="right")
    table.add_column("Lat. [s] CPU", justify="right")
    table.add_column("Mem. [MiB]", justify="right")
    table.add_column("FLOPs [T]", justify="right")

    for display, m in rows:
        table.add_row(
            display,
            fmt_pct(m["dice"]),
            fmt_mm(m["hd95"]),
            gpt.fmt_seconds(m["lat_gpu"]),
            gpt.fmt_seconds(m["lat_cpu"]),
            gpt.fmt_mib(m["mem"]),
            gpt.fmt_tflops(m["flops"]),
        )
    Console().print(table)


def build_latex_table(
    rows: list[tuple[str, dict[str, float | None]]],
    *,
    label: str = "tab:model_comparison",
    scope: str = "the paper's 18 baseline models",
) -> str:
    if not rows:
        return ""

    dice = [r[1]["dice"] for r in rows]
    hd95 = [r[1]["hd95"] for r in rows]
    lat_gpu = [r[1]["lat_gpu"] for r in rows]
    lat_cpu = [r[1]["lat_cpu"] for r in rows]
    mem = [r[1]["mem"] for r in rows]
    flops = [r[1]["flops"] for r in rows]

    best_dice, second_dice = ranked_indices_max(dice)
    best_hd95, second_hd95 = gft.ranked_indices(hd95)
    best_lat_gpu, second_lat_gpu = gft.ranked_indices(lat_gpu)
    best_lat_cpu, second_lat_cpu = gft.ranked_indices(lat_cpu)
    best_mem, second_mem = gft.ranked_indices(mem)
    best_flops, second_flops = gft.ranked_indices(flops)

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
        cells = [
            gft.tex_escape(display),
            cell(fmt_pct(dice[i]), i, best_dice, second_dice),
            cell(fmt_mm(hd95[i]), i, best_hd95, second_hd95),
            cell(gpt.fmt_seconds(lat_gpu[i]), i, best_lat_gpu, second_lat_gpu),
            cell(gpt.fmt_seconds(lat_cpu[i]), i, best_lat_cpu, second_lat_cpu),
            cell(gpt.fmt_mib(mem[i]), i, best_mem, second_mem),
            cell(gpt.fmt_tflops(flops[i]), i, best_flops, second_flops),
        ]
        body.append(" & ".join(cells) + r" \\")

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        f"\\caption{{Accuracy vs. computational cost for {scope}. "
        r"Dice (\%) and HD95 (mm) are averaged over all "
        r"modality-presence combinations, folds, WT/TC/ET, and every complete "
        r"dataset/checkpoint group (BraTS18, BraTS25-pre, and the internal "
        r"MB-96 cohort scored with both checkpoints). FLOPs, peak GPU memory, "
        r"and GPU/CPU latency are for a whole-volume forward pass with all "
        r"four modalities present. Bold marks the best model per column "
        r"(highest Dice, lowest everything else), underline the runner-up. "
        r"`n/a' marks models that do not support CPU inference.}",
        f"\\label{{{label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\rowcolors{3}{gray!8}{white}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Dice [\%]} & \textbf{HD95 [mm]} & "
        r"\textbf{Lat.\ GPU [s]} & \textbf{Lat.\ CPU [s]} & \textbf{Mem.\ [MiB]} & "
        r"\textbf{FLOPs [T]} \\",
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

    rows = build_rows(MODEL_SET)
    if not rows:
        print("No data found for the 18 baseline models", file=sys.stderr)
        return

    print_console_table(rows)

    tables_path = os.path.join(OUTPUT_DIR, "tables.tex")
    with open(tables_path, "w") as f:
        f.write(build_latex_table(rows))

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
