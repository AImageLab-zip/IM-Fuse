#!/usr/bin/env python3
"""Full profiler table for the paper's 18 models -- every metric `mimose
flops` produces, patch and whole-volume side by side.

Reads `flops_gpu.csv` (written by `mimose flops --device cuda`, see
sbatch_files/flops/*.sh), the sibling of `flops_gpu.txt` that
`write_flops_summary_csv` produces next to each model's report (see
`measure_kernel_launches_and_memory` and `write_flops_summary_csv` in
mimose/flops.py). Where generate_flops_table.py reports params/FLOPs/peak-
allocated-memory/latency and generate_kernel_memory_table.py adds kernel
launches/peak reserved memory but only for the whole-volume pass, this
script reports every column that CSV carries -- FLOPs, peak allocated
memory, peak reserved memory, memory overhead, kernel launches, and
latency -- for both the single patch/tile pass and the whole-volume pass, in
one table per model.

Kernel-launch count and peak reserved memory are GPU-only concepts (no
discrete kernel-launch model, and no torch.cuda.max_memory_reserved
equivalent, on CPU), so like generate_kernel_memory_table.py this script has
no CPU counterpart and requires flops_gpu.csv specifically.

Restricted, by default, to PAPER_TABLE1_MODELS (see generate_flops_table.py)
-- the paper's Table 1 (tab:models) 18 models plus TinyMimosa and
ManyMimosas.

Prints one console table (sorted by whole-volume GFLOPs, most efficient
model first) and writes one standalone LaTeX table (`tables.tex`) into
OUTPUT_DIR. Bold marks the best (lowest) value per column; underline marks
the runner-up. Unpatched models show "-" in the patch columns, since there
is no separate single-patch cost to report for them. Mem Overhead columns
are derived (Peak Reserved - Peak Allocated) and are not ranked.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_flops_table as gft  # noqa: E402
import generate_kernel_memory_table as gkmt  # noqa: E402

from rich.console import Console
from rich.table import Table

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "full_profile_table"
)
CSV_FILENAME = "flops_gpu.csv"


def collect_full_profile_rows(
    results_dir: str, *, allowed_models: set[str] | None = None
) -> list[tuple[str, dict[str, dict[str, str] | None]]]:
    """[(display_name, {"whole": row, "patch": row|None}), ...], sorted by
    whole-volume FLOPs (most efficient first). Skips models whose
    flops_gpu.csv lacks kernel-launch/reserved-memory data (older runs
    predating measure_kernel_launches_and_memory)."""
    rows = []
    for model_norm, csv_path in gft.discover_flops_csvs(
        results_dir, filename=CSV_FILENAME
    ).items():
        if allowed_models is not None and model_norm not in allowed_models:
            continue
        scoped = gft.load_scoped_rows(csv_path)
        whole = scoped.get("whole_volume")
        patch = scoped.get("patch")
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
        rows.append((display, {"whole": whole, "patch": patch}))
    rows.sort(key=lambda r: float(r[1]["whole"]["mean_flops"]))
    return rows


def print_console_table(
    rows: list[tuple[str, dict[str, dict[str, str] | None]]],
    *,
    title_suffix: str = "",
) -> None:
    table = Table(
        title="Full profiler summary — patch/tile vs. whole-volume forward" + title_suffix
    )
    table.add_column("Model", style="bold cyan")
    table.add_column("Params", justify="right")
    for scope_label in ("Patch", "Whole-Vol"):
        table.add_column(f"{scope_label} GFLOPs", justify="right")
        table.add_column(f"{scope_label} Peak Alloc", justify="right")
        table.add_column(f"{scope_label} Peak Reserved", justify="right")
        table.add_column(f"{scope_label} Mem Overhead", justify="right")
        table.add_column(f"{scope_label} Kernels", justify="right")
        table.add_column(f"{scope_label} Latency", justify="right")

    for display, scopes in rows:
        whole, patch = scopes["whole"], scopes["patch"]
        cells = [display, gft.fmt_count(float(whole["params"]))]
        for row in (patch, whole):
            if row is None:
                cells.extend(["-"] * 6)
                continue
            cells.extend(
                [
                    gft.fmt_count(float(row["mean_flops"])),
                    gft.fmt_bytes(row.get("mean_peak_memory_bytes")),
                    gft.fmt_bytes(row.get("mean_peak_reserved_memory_bytes")),
                    gkmt.fmt_overhead_pct(
                        row.get("mean_peak_reserved_memory_bytes"),
                        row.get("mean_peak_memory_bytes"),
                    ),
                    gkmt.fmt_kernel_launches(row.get("mean_kernel_launches")),
                    gft.fmt_seconds(row.get("mean_latency_seconds")),
                ]
            )
        table.add_row(*cells)

    Console().print(table)


def build_latex_table(
    rows: list[tuple[str, dict[str, dict[str, str] | None]]],
    *,
    label: str = "tab:full_profile",
    restricted: bool = False,
) -> str:
    if not rows:
        return ""

    def scoped_values(scope: str, field: str) -> list[float | None]:
        values = []
        for _, scopes in rows:
            row = scopes[scope]
            value = row.get(field) if row else None
            values.append(float(value) if value else None)
        return values

    params = [float(r[1]["whole"]["params"]) for r in rows]
    metrics = {}
    for scope in ("patch", "whole"):
        metrics[(scope, "flops")] = scoped_values(scope, "mean_flops")
        metrics[(scope, "alloc")] = scoped_values(scope, "mean_peak_memory_bytes")
        metrics[(scope, "reserved")] = scoped_values(scope, "mean_peak_reserved_memory_bytes")
        metrics[(scope, "kernel")] = scoped_values(scope, "mean_kernel_launches")
        metrics[(scope, "latency")] = scoped_values(scope, "mean_latency_seconds")

    ranks = {"params": gft.ranked_indices(params)}
    for key, values in metrics.items():
        ranks[key] = gft.ranked_indices(values)

    def cell(value_text: str, i: int, key: str) -> str:
        if value_text == "-":
            return value_text
        best, second = ranks[key]
        if i == best:
            return f"\\textbf{{{value_text}}}"
        if i == second:
            return f"\\underline{{{value_text}}}"
        return value_text

    body = []
    for i, (display, scopes) in enumerate(rows):
        cells = [gft.tex_escape(display), cell(gft.fmt_count(params[i]), i, "params")]
        for scope, row in (("patch", scopes["patch"]), ("whole", scopes["whole"])):
            if row is None:
                cells.extend(["-"] * 6)
                continue
            cells.extend(
                [
                    cell(gft.fmt_count(float(row["mean_flops"])), i, (scope, "flops")),
                    cell(gft.fmt_bytes(row.get("mean_peak_memory_bytes")), i, (scope, "alloc")),
                    cell(
                        gft.fmt_bytes(row.get("mean_peak_reserved_memory_bytes")),
                        i,
                        (scope, "reserved"),
                    ),
                    gkmt.fmt_overhead_pct(
                        row.get("mean_peak_reserved_memory_bytes"),
                        row.get("mean_peak_memory_bytes"),
                    ),
                    cell(
                        gkmt.fmt_kernel_launches(row.get("mean_kernel_launches")),
                        i,
                        (scope, "kernel"),
                    ),
                    cell(gft.fmt_seconds(row.get("mean_latency_seconds")), i, (scope, "latency")),
                ]
            )
        body.append(" & ".join(cells) + r" \\")

    scope_note = (
        r" Restricted to the 18 models in the paper's Table~1 "
        r"(\texttt{tab:models}) plus TinyMimosa and ManyMimosas."
        if restricted
        else ""
    )
    lines = [
        r"\begin{table*}[!ht]",
        r"\centering",
        r"\caption{Full computational profile per model: parameter count, "
        r"and FLOPs / peak allocated memory / peak reserved memory / memory "
        r"overhead (reserved vs.\ allocated) / GPU kernel launches / wall-"
        r"clock latency, for a single patch/tile forward pass versus the "
        r"whole input volume (aggregated via sliding-window inference for "
        r"patch-based models; `-' where a model has no separate patch cost, "
        r"i.e.\ it consumes the whole volume directly). All measurements use "
        r"all four modalities present. Bold marks the most efficient model "
        r"per column, underline the runner-up; Mem Overhead is derived and "
        r"not ranked."
        f"{scope_note}}}",
        f"\\label{{{label}}}",
        r"\tiny",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\rowcolors{3}{gray!8}{white}",
        r"\begin{tabular}{lc*{6}{c}*{6}{c}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Params} & "
        r"\multicolumn{6}{c}{\textbf{Single Patch/Tile}} & "
        r"\multicolumn{6}{c}{\textbf{Whole Volume}} \\",
        r"\cmidrule(lr){3-8} \cmidrule(lr){9-14}",
        r" & & GFLOPs & Peak Alloc & Peak Reserved & Mem Overhead & Kernels & Latency"
        r" & GFLOPs & Peak Alloc & Peak Reserved & Mem Overhead & Kernels & Latency \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
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

    rows = collect_full_profile_rows(gft.RESULTS_DIR, allowed_models=allowed_models)
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
        label="tab:full_profile",
        title_suffix=" (paper Table 1 only)",
        restricted=True,
    )


if __name__ == "__main__":
    main()
