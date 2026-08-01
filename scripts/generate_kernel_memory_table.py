#!/usr/bin/env python3
"""Kernel-launch-count and memory-pressure table for the paper's 18 models.

Reads `flops_gpu.csv` (written by `mimose flops --device cuda`, see
sbatch_files/flops/*.sh), the sibling of `flops_gpu.txt` that
`write_flops_summary_csv` produces next to each model's report -- see
`measure_kernel_launches_and_memory` in mimose/flops.py. Kernel-launch count
and peak *reserved* CUDA memory are GPU-only concepts (no discrete
kernel-launch model, and no `torch.cuda.max_memory_reserved` equivalent, on
CPU), so unlike generate_flops_table.py this script has no CPU counterpart.

Peak *reserved* memory is the allocator's cached/reserved footprint, a
better proxy for actual device memory pressure than the peak *allocated*
figure alone: the gap between the two ("Mem Overhead") reflects allocator
fragmentation/caching overhead that allocated-only figures hide.

Restricted, by default, to PAPER_TABLE1_MODELS (see generate_flops_table.py)
-- the paper's Table 1 (tab:models) 18 models plus TinyMimosa and
ManyMimosas. Reports the whole-volume pass only (aggregated via a model's
own sliding-window predict() for patch-based models, or processed directly
otherwise), since that's the comparable end-to-end cost across models with
different patching strategies.

Prints one console table (sorted by kernel-launch count, fewest first) and
writes one standalone LaTeX table (`tables.tex`) into OUTPUT_DIR. Bold marks
the best (lowest) value per column; underline marks the runner-up.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_flops_table as gft  # noqa: E402

from rich.console import Console
from rich.table import Table

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "kernel_memory_table"
)
CSV_FILENAME = "flops_gpu.csv"


def fmt_kernel_launches(value: str | float | None) -> str:
    if not value and value != 0:
        return "-"
    return gft.fmt_count(float(value))


def fmt_overhead_pct(reserved: str | float | None, allocated: str | float | None) -> str:
    if not reserved or not allocated or float(allocated) == 0:
        return "-"
    pct = (float(reserved) - float(allocated)) / float(allocated) * 100
    return f"{pct:.1f}%"


def collect_kernel_memory_rows(
    results_dir: str, *, allowed_models: set[str] | None = None
) -> list[tuple[str, dict[str, str]]]:
    """[(display_name, whole_volume_row), ...], sorted by kernel-launch count
    (fewest first). Skips models whose flops_gpu.csv lacks kernel/reserved-
    memory data (older runs predating measure_kernel_launches_and_memory)."""
    rows = []
    for model_norm, csv_path in gft.discover_flops_csvs(
        results_dir, filename=CSV_FILENAME
    ).items():
        if allowed_models is not None and model_norm not in allowed_models:
            continue
        scoped = gft.load_scoped_rows(csv_path)
        whole = scoped.get("whole_volume")
        if whole is None:
            print(f"{model_norm}: {CSV_FILENAME} has no whole_volume row, skipping", file=sys.stderr)
            continue
        if not whole.get("mean_kernel_launches"):
            print(
                f"{model_norm}: {CSV_FILENAME} predates kernel-launch/reserved-memory "
                "profiling, skipping (re-run its sbatch flops job)",
                file=sys.stderr,
            )
            continue
        display = gft.DISPLAY_NAMES.get(model_norm, model_norm.upper())
        rows.append((display, whole))
    rows.sort(key=lambda r: float(r[1]["mean_kernel_launches"]))
    return rows


def print_console_table(
    rows: list[tuple[str, dict[str, str]]], *, title_suffix: str = ""
) -> None:
    table = Table(title="Kernel launches & memory pressure — whole-volume forward" + title_suffix)
    table.add_column("Model", style="bold cyan")
    table.add_column("Kernel Launches", justify="right")
    table.add_column("Peak Allocated", justify="right")
    table.add_column("Peak Reserved", justify="right")
    table.add_column("Mem Overhead", justify="right")
    table.add_column("GFLOPs", justify="right")
    table.add_column("Latency", justify="right")

    for display, whole in rows:
        table.add_row(
            display,
            fmt_kernel_launches(whole.get("mean_kernel_launches")),
            gft.fmt_bytes(whole.get("mean_peak_memory_bytes")),
            gft.fmt_bytes(whole.get("mean_peak_reserved_memory_bytes")),
            fmt_overhead_pct(
                whole.get("mean_peak_reserved_memory_bytes"),
                whole.get("mean_peak_memory_bytes"),
            ),
            gft.fmt_count(float(whole["mean_flops"])),
            gft.fmt_seconds(whole.get("mean_latency_seconds")),
        )

    Console().print(table)


def build_latex_table(
    rows: list[tuple[str, dict[str, str]]],
    *,
    label: str = "tab:kernel_memory",
    restricted: bool = False,
) -> str:
    if not rows:
        return ""

    kernel_launches = [float(r[1]["mean_kernel_launches"]) for r in rows]
    peak_allocated = [
        float(r[1]["mean_peak_memory_bytes"]) if r[1].get("mean_peak_memory_bytes") else None
        for r in rows
    ]
    peak_reserved = [
        float(r[1]["mean_peak_reserved_memory_bytes"])
        if r[1].get("mean_peak_reserved_memory_bytes")
        else None
        for r in rows
    ]

    best_kernel, second_kernel = gft.ranked_indices(kernel_launches)
    best_alloc, second_alloc = gft.ranked_indices(peak_allocated)
    best_reserved, second_reserved = gft.ranked_indices(peak_reserved)

    def cell(value_text: str, i: int, best: int | None, second: int | None) -> str:
        if value_text == "-":
            return value_text
        if i == best:
            return f"\\textbf{{{value_text}}}"
        if i == second:
            return f"\\underline{{{value_text}}}"
        return value_text

    body = []
    for i, (display, whole) in enumerate(rows):
        cells = [
            gft.tex_escape(display),
            cell(fmt_kernel_launches(whole.get("mean_kernel_launches")), i, best_kernel, second_kernel),
            cell(gft.fmt_bytes(whole.get("mean_peak_memory_bytes")), i, best_alloc, second_alloc),
            cell(
                gft.fmt_bytes(whole.get("mean_peak_reserved_memory_bytes")),
                i,
                best_reserved,
                second_reserved,
            ),
            fmt_overhead_pct(
                whole.get("mean_peak_reserved_memory_bytes"),
                whole.get("mean_peak_memory_bytes"),
            ),
        ]
        body.append(" & ".join(cells) + r" \\")

    scope_note = (
        r" Restricted to the 18 models in the paper's Table~1 "
        r"(\texttt{tab:models}) plus TinyMimosa and ManyMimosas."
        if restricted
        else ""
    )
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{GPU kernel-launch count and memory pressure per model, "
        r"whole-volume forward pass (aggregated via sliding-window inference "
        r"for patch-based models; processed directly for models that consume "
        r"the whole volume in one shot), all four modalities present. Peak "
        r"Reserved is the CUDA allocator's cached footprint "
        r"(\texttt{torch.cuda.max\_memory\_reserved}); Mem Overhead is the "
        r"gap between it and Peak Allocated "
        r"(\texttt{torch.cuda.max\_memory\_allocated}), i.e.\ allocator "
        r"fragmentation/caching overhead. Bold marks the most efficient "
        r"model per column, underline the runner-up."
        f"{scope_note}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\rowcolors{2}{gray!8}{white}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Kernel Launches} & \textbf{Peak Allocated} & "
        r"\textbf{Peak Reserved} & \textbf{Mem Overhead} \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def run(
    *,
    allowed_models: set[str] | None,
    output_dir: str,
    label: str,
    title_suffix: str = "",
    restricted: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    rows = collect_kernel_memory_rows(gft.RESULTS_DIR, allowed_models=allowed_models)
    if not rows:
        print(
            f"No {CSV_FILENAME} files with kernel-launch/reserved-memory data found "
            f"under {gft.RESULTS_DIR}/<model>_23/",
            file=sys.stderr,
        )
        return

    print_console_table(rows, title_suffix=title_suffix)

    tables_path = os.path.join(output_dir, "tables.tex")
    with open(tables_path, "w") as f:
        f.write(build_latex_table(rows, label=label, restricted=restricted))
    print(f"Wrote {tables_path}")


def main() -> None:
    run(
        allowed_models=gft.PAPER_TABLE1_MODELS,
        output_dir=OUTPUT_DIR,
        label="tab:kernel_memory",
        title_suffix=" (paper Table 1 only)",
        restricted=True,
    )


if __name__ == "__main__":
    main()
