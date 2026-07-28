#!/usr/bin/env python3
"""Group `mimose flops` results into one summary table.

Reads the `flops.csv` sibling that `mimose flops` writes next to each
model's `flops.txt` (see `write_flops_summary_csv` in `mimose.flops`),
one per model directory under RESULTS_DIR (`<model>_23/flops.csv`,
written by sbatch_files/flops_18models.sh). Each row in that CSV is
tagged with a `scope` of "patch" (a single fixed-size patch/tile
through the network) or "whole_volume" (the full input: aggregated via
a model's own sliding-window predict() for patch-based models, or
processed directly for models that consume the whole volume in one
shot), and carries params, FLOPs, peak memory, and mean wall-clock
latency (`mean_latency_seconds`, timed separately from FLOPs counting
via `measure_latency` -- see `mimose.flops`). This script reports both
scopes side by side per model, since patched and unpatched models are
not comparable on a single FLOPs/latency figure -- a patched model's
"patch" cost looks artificially cheap next to an unpatched model's
whole-volume cost.

Prints four console tables (all sorted by whole-volume GFLOPs, most
efficient model first): the full set of discovered models with patch
and whole-volume columns side by side, the same restricted to
PAPER_TABLE1_MODELS (the paper's Table 1 (tab:models) 18 models plus
TinyMimosa and ManyMimosas, the paper's own MissingBench-arch
contribution -- excluding only models that aren't part of the paper at
all), and a whole-volume-only variant of each (for contexts where the
patch/tile figure isn't wanted, e.g. it makes patch-based models look
artificially cheap next to models that only ever report a whole-volume
cost).
Writes all four as separate LaTeX tables (`tables.tex`,
`tables_whole_volume_only.tex`, `tables_restricted.tex`,
`tables_restricted_whole_volume_only.tex`) plus a standalone preview
`main.tex` into OUTPUT_DIR, compiling it to PDF with `tectonic` if
available. Bold marks the best (lowest) value per column; underline
marks the runner-up. Unpatched models show "-" in the patch columns of
the combined tables, since there is no separate single-patch cost to
report for them.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import sys

from rich.console import Console
from rich.table import Table

RESULTS_DIR = "/work/phd_mimose/results"
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "flops_table"
)

# Paper Table 1 (tab:models) display names, plus the MissingBench-arch
# lightweight models not yet in that table. Models found on disk but not
# listed here fall back to their raw config-stem name, upper-cased.
DISPLAY_NAMES = {
    "a2fseg": "A2FSeg",
    "dcseg": "DC-Seg",
    "imfuse": "IM-Fuse",
    "ims2trans": "IMS2Trans",
    "inoutfusion": "InOutFusion",
    "lckd": "LCKD",
    "m2ftrans": "M2FTrans",
    "m3fecon": "M3FeCon",
    "mifpn": "MIFPN",
    "mmformer": "mmFormer",
    "mmmvit": "MMMViT",
    "rfl": "RFL",
    "rfnet": "RFNet",
    "robustseg": "RobustSeg",
    "sfusion": "SFusion",
    "srmnet": "SRMNet",
    "uhved": "U-HVED",
    "unetmfi": "UNET-MFI",
    "tinymimosa": "TinyMimosa",
    "manymimosas": "ManyMimosas",
    "manymimosaskd": "ManyMimosasKD",
}

# The paper's Table 1 (tab:models) 18 models, plus TinyMimosa and
# ManyMimosas (the paper's own MissingBench-arch contribution). Excludes
# only the models that aren't part of the paper at all (manymimosaskd,
# mcpl, shaspec, m3ae, mambavitakd, mstkdnet, tinymimosaweighted). Used for
# the "restricted" tables.
PAPER_TABLE1_MODELS = {
    "a2fseg",
    "dcseg",
    "imfuse",
    "ims2trans",
    "inoutfusion",
    "lckd",
    "m2ftrans",
    "m3fecon",
    "mifpn",
    "mmformer",
    "mmmvit",
    "rfl",
    "rfnet",
    "robustseg",
    "sfusion",
    "srmnet",
    "uhved",
    "unetmfi",
    "tinymimosa",
    "manymimosas",
}


def discover_flops_csvs(results_dir: str, *, filename: str = "flops.csv") -> dict[str, str]:
    """{model_norm: path/to/<filename>} for every <model>_23/<filename> found."""
    found = {}
    if not os.path.isdir(results_dir):
        return found
    for entry in sorted(os.listdir(results_dir)):
        m = re.match(r"^(.+)_23$", entry)
        if not m:
            continue
        csv_path = os.path.join(results_dir, entry, filename)
        if os.path.isfile(csv_path):
            found[m.group(1)] = csv_path
    return found


def load_scoped_rows(csv_path: str) -> dict[str, dict[str, str]]:
    """{"patch": row, "whole_volume": row} for whichever scopes are present."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row["scope"]: row for row in csv.DictReader(f)}


def collect_rows(
    results_dir: str, *, allowed_models: set[str] | None = None, filename: str = "flops.csv"
) -> list[tuple[str, dict[str, dict[str, str] | None]]]:
    rows = []
    for model_norm, csv_path in discover_flops_csvs(results_dir, filename=filename).items():
        if allowed_models is not None and model_norm not in allowed_models:
            continue
        scoped = load_scoped_rows(csv_path)
        whole = scoped.get("whole_volume")
        patch = scoped.get("patch")
        if whole is None:
            print(f"{model_norm}: {filename} has no whole_volume row, skipping", file=sys.stderr)
            continue
        display = DISPLAY_NAMES.get(model_norm, model_norm.upper())
        rows.append((display, {"whole": whole, "patch": patch}))
    rows.sort(key=lambda r: float(r[1]["whole"]["mean_flops"]))
    return rows


def fmt_count(value: float | None) -> str:
    if value is None:
        return "-"
    for unit, threshold in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if value >= threshold:
            return f"{value / threshold:.2f}{unit}"
    return f"{value:.0f}"


def fmt_bytes(value: str | None) -> str:
    if not value:
        return "-"
    mib = float(value) / (1024 ** 2)
    return f"{mib:.0f} MiB"


def fmt_seconds(value: str | float | None) -> str:
    if not value:
        return "-"
    seconds = float(value)
    if seconds >= 1:
        return f"{seconds:.2f}s"
    return f"{seconds * 1e3:.1f}ms"


def fmt_seconds_with_std(value: str | float | None, std: str | float | None) -> str:
    """Like fmt_seconds, plus a "± std" suffix in the same unit when a std is
    available (older flops_cpu.csv/flops.csv files predate that column)."""
    formatted = fmt_seconds(value)
    if formatted == "-" or not std:
        return formatted
    seconds = float(value)
    std_seconds = float(std)
    if seconds >= 1:
        return f"{formatted} ± {std_seconds:.2f}s"
    return f"{formatted} ± {std_seconds * 1e3:.1f}ms"


def print_console_table(
    rows: list[tuple[str, dict[str, dict[str, str] | None]]],
    *,
    title_suffix: str = "",
) -> None:
    table = Table(
        title="FLOPs summary — single patch/tile forward vs. whole-volume forward"
        + title_suffix
    )
    table.add_column("Model", style="bold cyan")
    table.add_column("Params", justify="right")
    table.add_column("Patch GFLOPs", justify="right")
    table.add_column("Patch Mem", justify="right")
    table.add_column("Patch Latency", justify="right")
    table.add_column("Whole-Vol GFLOPs", justify="right")
    table.add_column("Whole-Vol Mem", justify="right")
    table.add_column("Whole-Vol Latency", justify="right")

    for display, scopes in rows:
        whole, patch = scopes["whole"], scopes["patch"]
        table.add_row(
            display,
            fmt_count(float(whole["params"])),
            fmt_count(float(patch["mean_flops"])) if patch else "-",
            fmt_bytes(patch["mean_peak_memory_bytes"]) if patch else "-",
            fmt_seconds(patch.get("mean_latency_seconds")) if patch else "-",
            fmt_count(float(whole["mean_flops"])),
            fmt_bytes(whole["mean_peak_memory_bytes"]),
            fmt_seconds(whole.get("mean_latency_seconds")),
        )

    Console().print(table)


def print_console_table_whole_only(
    rows: list[tuple[str, dict[str, dict[str, str] | None]]],
    *,
    title_suffix: str = "",
) -> None:
    table = Table(title="FLOPs summary — whole-volume forward only" + title_suffix)
    table.add_column("Model", style="bold cyan")
    table.add_column("Params", justify="right")
    table.add_column("GFLOPs", justify="right")
    table.add_column("Peak Memory", justify="right")
    table.add_column("Latency", justify="right")

    for display, scopes in rows:
        whole = scopes["whole"]
        table.add_row(
            display,
            fmt_count(float(whole["params"])),
            fmt_count(float(whole["mean_flops"])),
            fmt_bytes(whole["mean_peak_memory_bytes"]),
            fmt_seconds(whole.get("mean_latency_seconds")),
        )

    Console().print(table)


def tex_escape(s: str) -> str:
    return re.sub(r"([&%$#_{}])", r"\\\1", s)


def ranked_indices(values: list[float | None]) -> tuple[int | None, int | None]:
    """(index of smallest, index of second-smallest) among non-None values."""
    available = [(i, v) for i, v in enumerate(values) if v is not None]
    order = sorted(available, key=lambda t: t[1])
    best = order[0][0] if order else None
    runner_up = order[1][0] if len(order) > 1 else None
    return best, runner_up


def build_latex_table(
    rows: list[tuple[str, dict[str, dict[str, str] | None]]],
    *,
    label: str = "tab:flops",
    restricted: bool = False,
) -> str:
    if not rows:
        return ""

    params = [float(r[1]["whole"]["params"]) for r in rows]
    patch_flops = [
        float(r[1]["patch"]["mean_flops"]) if r[1]["patch"] else None for r in rows
    ]
    whole_flops = [float(r[1]["whole"]["mean_flops"]) for r in rows]
    patch_memory = [
        float(r[1]["patch"]["mean_peak_memory_bytes"])
        if r[1]["patch"] and r[1]["patch"]["mean_peak_memory_bytes"]
        else None
        for r in rows
    ]
    whole_memory = [
        float(r[1]["whole"]["mean_peak_memory_bytes"])
        if r[1]["whole"]["mean_peak_memory_bytes"]
        else None
        for r in rows
    ]
    patch_latency = [
        float(r[1]["patch"]["mean_latency_seconds"])
        if r[1]["patch"] and r[1]["patch"].get("mean_latency_seconds")
        else None
        for r in rows
    ]
    whole_latency = [
        float(r[1]["whole"]["mean_latency_seconds"])
        if r[1]["whole"].get("mean_latency_seconds")
        else None
        for r in rows
    ]

    best_params, second_params = ranked_indices(params)
    best_patch_flops, second_patch_flops = ranked_indices(patch_flops)
    best_whole_flops, second_whole_flops = ranked_indices(whole_flops)
    best_patch_mem, second_patch_mem = ranked_indices(patch_memory)
    best_whole_mem, second_whole_mem = ranked_indices(whole_memory)
    best_patch_lat, second_patch_lat = ranked_indices(patch_latency)
    best_whole_lat, second_whole_lat = ranked_indices(whole_latency)

    def cell(value_text: str, i: int, best: int | None, second: int | None) -> str:
        if value_text == "-":
            return value_text
        if i == best:
            return f"\\textbf{{{value_text}}}"
        if i == second:
            return f"\\underline{{{value_text}}}"
        return value_text

    body = []
    for i, (display, scopes) in enumerate(rows):
        whole, patch = scopes["whole"], scopes["patch"]
        cells = [
            tex_escape(display),
            cell(fmt_count(params[i]), i, best_params, second_params),
            cell(
                fmt_count(patch_flops[i]), i, best_patch_flops, second_patch_flops
            ),
            cell(
                fmt_bytes(patch["mean_peak_memory_bytes"]) if patch else "-",
                i,
                best_patch_mem,
                second_patch_mem,
            ),
            cell(
                fmt_seconds(patch.get("mean_latency_seconds")) if patch else "-",
                i,
                best_patch_lat,
                second_patch_lat,
            ),
            cell(fmt_count(whole_flops[i]), i, best_whole_flops, second_whole_flops),
            cell(
                fmt_bytes(whole["mean_peak_memory_bytes"]),
                i,
                best_whole_mem,
                second_whole_mem,
            ),
            cell(
                fmt_seconds(whole.get("mean_latency_seconds")),
                i,
                best_whole_lat,
                second_whole_lat,
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
        r"\caption{Computational cost per model: parameter count, and FLOPs / "
        r"peak memory / wall-clock latency for a single patch/tile forward "
        r"pass versus the whole input volume (aggregated via sliding-window "
        r"inference for patch-based models; `-' where a model has no "
        r"separate patch cost, i.e.\ it consumes the whole volume directly). "
        r"All measurements use all four modalities present. Bold marks the "
        r"most efficient model per column, underline the runner-up."
        f"{scope_note}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\rowcolors{3}{gray!8}{white}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Params} & "
        r"\multicolumn{3}{c}{\textbf{Single Patch/Tile}} & "
        r"\multicolumn{3}{c}{\textbf{Whole Volume}} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}",
        r" & & GFLOPs & Peak Mem & Latency & GFLOPs & Peak Mem & Latency \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def build_latex_table_whole_only(
    rows: list[tuple[str, dict[str, dict[str, str] | None]]],
    *,
    label: str = "tab:flops_whole_volume",
    restricted: bool = False,
) -> str:
    """Same models/ranking as build_latex_table, but only the whole-volume
    columns -- for contexts where the patch/tile number would be misleading
    or is simply not of interest (e.g. an inference-cost comparison, since
    unpatched models never had a patch number to compare against anyway)."""
    if not rows:
        return ""

    params = [float(r[1]["whole"]["params"]) for r in rows]
    whole_flops = [float(r[1]["whole"]["mean_flops"]) for r in rows]
    whole_memory = [
        float(r[1]["whole"]["mean_peak_memory_bytes"])
        if r[1]["whole"]["mean_peak_memory_bytes"]
        else None
        for r in rows
    ]
    whole_latency = [
        float(r[1]["whole"]["mean_latency_seconds"])
        if r[1]["whole"].get("mean_latency_seconds")
        else None
        for r in rows
    ]

    best_params, second_params = ranked_indices(params)
    best_whole_flops, second_whole_flops = ranked_indices(whole_flops)
    best_whole_mem, second_whole_mem = ranked_indices(whole_memory)
    best_whole_lat, second_whole_lat = ranked_indices(whole_latency)

    def cell(value_text: str, i: int, best: int | None, second: int | None) -> str:
        if value_text == "-":
            return value_text
        if i == best:
            return f"\\textbf{{{value_text}}}"
        if i == second:
            return f"\\underline{{{value_text}}}"
        return value_text

    body = []
    for i, (display, scopes) in enumerate(rows):
        whole = scopes["whole"]
        cells = [
            tex_escape(display),
            cell(fmt_count(params[i]), i, best_params, second_params),
            cell(fmt_count(whole_flops[i]), i, best_whole_flops, second_whole_flops),
            cell(
                fmt_bytes(whole["mean_peak_memory_bytes"]),
                i,
                best_whole_mem,
                second_whole_mem,
            ),
            cell(
                fmt_seconds(whole.get("mean_latency_seconds")),
                i,
                best_whole_lat,
                second_whole_lat,
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
        r"\caption{Computational cost per model on the whole input volume "
        r"(aggregated via sliding-window inference for patch-based models; "
        r"processed directly for models that consume the whole volume in "
        r"one shot), with all four modalities present. Bold marks the most "
        r"efficient model per column, underline the runner-up."
        f"{scope_note}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\rowcolors{3}{gray!8}{white}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Params} & \textbf{GFLOPs} & \textbf{Peak Memory} & \textbf{Latency} \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def compile_pdf(main_path: str) -> None:
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


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = collect_rows(RESULTS_DIR)
    if not rows:
        print(f"No flops.csv files found under {RESULTS_DIR}/<model>_23/", file=sys.stderr)
        return

    restricted_rows = collect_rows(RESULTS_DIR, allowed_models=PAPER_TABLE1_MODELS)

    print_console_table(rows)
    print_console_table_whole_only(rows)
    if restricted_rows:
        print_console_table(restricted_rows, title_suffix=" (paper Table 1 only)")
        print_console_table_whole_only(
            restricted_rows, title_suffix=" (paper Table 1 only)"
        )
    else:
        print(
            "No paper Table 1 models found under RESULTS_DIR; skipping restricted tables",
            file=sys.stderr,
        )

    tables_path = os.path.join(OUTPUT_DIR, "tables.tex")
    with open(tables_path, "w") as f:
        f.write(build_latex_table(rows))

    whole_only_path = os.path.join(OUTPUT_DIR, "tables_whole_volume_only.tex")
    with open(whole_only_path, "w") as f:
        f.write(build_latex_table_whole_only(rows))

    restricted_path = os.path.join(OUTPUT_DIR, "tables_restricted.tex")
    with open(restricted_path, "w") as f:
        f.write(
            build_latex_table(
                restricted_rows, label="tab:flops_restricted", restricted=True
            )
        )

    restricted_whole_only_path = os.path.join(
        OUTPUT_DIR, "tables_restricted_whole_volume_only.tex"
    )
    with open(restricted_whole_only_path, "w") as f:
        f.write(
            build_latex_table_whole_only(
                restricted_rows,
                label="tab:flops_whole_volume_restricted",
                restricted=True,
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
\clearpage
\input{tables_whole_volume_only.tex}
\clearpage
\input{tables_restricted.tex}
\clearpage
\input{tables_restricted_whole_volume_only.tex}
\end{document}
"""
    main_path = os.path.join(OUTPUT_DIR, "main.tex")
    with open(main_path, "w") as f:
        f.write(main_tex)

    print(f"Wrote {tables_path}")
    print(f"Wrote {whole_only_path}")
    print(f"Wrote {restricted_path}")
    print(f"Wrote {restricted_whole_only_path}")
    print(f"Wrote {main_path}")

    compile_pdf(main_path)


if __name__ == "__main__":
    main()
