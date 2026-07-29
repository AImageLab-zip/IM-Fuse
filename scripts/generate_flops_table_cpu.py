#!/usr/bin/env python3
"""CPU counterpart of generate_flops_table.py.

Reads `flops_cpu.csv` (written by `mimose flops --device cpu`, see
sbatch_files/flops/*.sh) instead of the GPU `flops.csv`, and builds the same
four LaTeX tables (full set + whole-volume-only, each combined and
restricted to the paper's Table 1 models). Peak memory is dropped from every
table here: `mimose flops` never measures it on CPU (there is no cheap,
reliable equivalent to CUDA's max_memory_allocated for plain CPU tensor
allocations -- see measure_pass's `device.type == "cuda"` guard in
mimose/flops.py), so the column would just be empty in every row.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_flops_table as gft  # noqa: E402

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "flops_table_cpu"
)
CSV_FILENAME = "flops_cpu.csv"


def sort_alphabetically(
    rows: list[tuple[str, dict[str, dict[str, str] | None]]],
) -> list[tuple[str, dict[str, dict[str, str] | None]]]:
    """Row order for this file only -- gft.collect_rows sorts by FLOPs (shared
    with generate_flops_table.py's GPU tables, which stays that way), so
    re-sort here rather than changing the shared helper. Case-insensitive so
    a lowercase-leading display name (e.g. "mmFormer") sorts next to its
    peers instead of after every upper-case name."""
    return sorted(rows, key=lambda r: r[0].lower())


# Dropped from the full (unrestricted) tables only -- the "restricted" tables
# are defined (see gft.PAPER_TABLE1_MODELS) as the paper's Table 1 PLUS these
# two, so excluding them there would contradict what "restricted" means.
EXCLUDED_DISPLAY_NAMES = {"TinyMimosa", "ManyMimosas"}


def exclude_models(
    rows: list[tuple[str, dict[str, dict[str, str] | None]]],
) -> list[tuple[str, dict[str, dict[str, str] | None]]]:
    return [r for r in rows if r[0] not in EXCLUDED_DISPLAY_NAMES]


def _cell(value_text: str, i: int, best: int | None, second: int | None) -> str:
    if value_text == "-":
        return value_text
    if i == best:
        return f"\\textbf{{{value_text}}}"
    if i == second:
        return f"\\underline{{{value_text}}}"
    return value_text


def build_latex_table_cpu(
    rows: list[tuple[str, dict[str, dict[str, str] | None]]],
    *,
    label: str = "tab:flops_cpu",
    restricted: bool = False,
) -> str:
    if not rows:
        return ""

    params = [float(r[1]["whole"]["params"]) for r in rows]
    patch_flops = [
        float(r[1]["patch"]["mean_flops"]) if r[1]["patch"] else None for r in rows
    ]
    whole_flops = [float(r[1]["whole"]["mean_flops"]) for r in rows]
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

    best_params, second_params = gft.ranked_indices(params)
    best_patch_flops, second_patch_flops = gft.ranked_indices(patch_flops)
    best_whole_flops, second_whole_flops = gft.ranked_indices(whole_flops)
    best_patch_lat, second_patch_lat = gft.ranked_indices(patch_latency)
    best_whole_lat, second_whole_lat = gft.ranked_indices(whole_latency)

    body = []
    for i, (display, scopes) in enumerate(rows):
        whole, patch = scopes["whole"], scopes["patch"]
        cells = [
            gft.tex_escape(display),
            _cell(gft.fmt_count(params[i]), i, best_params, second_params),
            _cell(gft.fmt_count(patch_flops[i]), i, best_patch_flops, second_patch_flops),
            _cell(
                gft.fmt_seconds_with_std(
                    patch.get("mean_latency_seconds"),
                    patch.get("latency_std_across_modalities_seconds"),
                )
                if patch
                else "-",
                i,
                best_patch_lat,
                second_patch_lat,
            ),
            _cell(gft.fmt_count(whole_flops[i]), i, best_whole_flops, second_whole_flops),
            _cell(
                gft.fmt_seconds_with_std(
                    whole.get("mean_latency_seconds"),
                    whole.get("latency_std_across_modalities_seconds"),
                ),
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
        r"\caption{CPU computational cost per model: parameter count, and "
        r"FLOPs / wall-clock latency for a single patch/tile forward pass "
        r"versus the whole input volume (aggregated via sliding-window "
        r"inference for patch-based models; `-' where a model has no "
        r"separate patch cost, i.e.\ it consumes the whole volume directly). "
        r"All measurements use all four modalities present, run on CPU only "
        r"(no peak-memory figures -- \texttt{mimose flops} does not measure "
        r"peak memory on CPU). Latency is shown as mean $\pm$ std across the 15 "
        r"modality-presence combinations, since execution time varies with "
        r"how many modalities are available. Bold marks the most efficient "
        r"model per "
        r"column, underline the runner-up."
        f"{scope_note}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\rowcolors{3}{gray!8}{white}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Params} & "
        r"\multicolumn{2}{c}{\textbf{Single Patch/Tile}} & "
        r"\multicolumn{2}{c}{\textbf{Whole Volume}} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r" & & GFLOPs & Latency & GFLOPs & Latency \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def build_latex_table_cpu_whole_only(
    rows: list[tuple[str, dict[str, dict[str, str] | None]]],
    *,
    label: str = "tab:flops_cpu_whole_volume",
    restricted: bool = False,
) -> str:
    if not rows:
        return ""

    params = [float(r[1]["whole"]["params"]) for r in rows]
    whole_flops = [float(r[1]["whole"]["mean_flops"]) for r in rows]
    whole_latency = [
        float(r[1]["whole"]["mean_latency_seconds"])
        if r[1]["whole"].get("mean_latency_seconds")
        else None
        for r in rows
    ]

    best_params, second_params = gft.ranked_indices(params)
    best_whole_flops, second_whole_flops = gft.ranked_indices(whole_flops)
    best_whole_lat, second_whole_lat = gft.ranked_indices(whole_latency)

    body = []
    for i, (display, scopes) in enumerate(rows):
        whole = scopes["whole"]
        cells = [
            gft.tex_escape(display),
            _cell(gft.fmt_count(params[i]), i, best_params, second_params),
            _cell(gft.fmt_count(whole_flops[i]), i, best_whole_flops, second_whole_flops),
            _cell(
                gft.fmt_seconds_with_std(
                    whole.get("mean_latency_seconds"),
                    whole.get("latency_std_across_modalities_seconds"),
                ),
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
        r"\caption{CPU computational cost per model on the whole input "
        r"volume (aggregated via sliding-window inference for patch-based "
        r"models; processed directly for models that consume the whole "
        r"volume in one shot), with all four modalities present, run on CPU "
        r"only (no peak-memory figures -- \texttt{mimose flops} does not "
        r"measure peak memory on CPU). Latency is shown as mean $\pm$ std "
        r"across the 15 modality-presence combinations, since execution "
        r"time varies with how many modalities are available. Bold marks "
        r"the most efficient model per column, underline the runner-up."
        f"{scope_note}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\rowcolors{3}{gray!8}{white}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Params} & \textbf{GFLOPs} & \textbf{Latency} \\",
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

    rows = gft.collect_rows(gft.RESULTS_DIR, filename=CSV_FILENAME)
    if not rows:
        print(
            f"No {CSV_FILENAME} files found under {gft.RESULTS_DIR}/<model>_23/",
            file=sys.stderr,
        )
        return
    rows = sort_alphabetically(exclude_models(rows))

    restricted_rows = gft.collect_rows(
        gft.RESULTS_DIR, allowed_models=gft.PAPER_TABLE1_MODELS, filename=CSV_FILENAME
    )
    restricted_rows = sort_alphabetically(restricted_rows)

    tables_path = os.path.join(OUTPUT_DIR, "tables.tex")
    with open(tables_path, "w") as f:
        f.write(build_latex_table_cpu(rows))

    whole_only_path = os.path.join(OUTPUT_DIR, "tables_whole_volume_only.tex")
    with open(whole_only_path, "w") as f:
        f.write(build_latex_table_cpu_whole_only(rows))

    restricted_path = os.path.join(OUTPUT_DIR, "tables_restricted.tex")
    with open(restricted_path, "w") as f:
        f.write(
            build_latex_table_cpu(
                restricted_rows, label="tab:flops_cpu_restricted", restricted=True
            )
        )

    restricted_whole_only_path = os.path.join(
        OUTPUT_DIR, "tables_restricted_whole_volume_only.tex"
    )
    with open(restricted_whole_only_path, "w") as f:
        f.write(
            build_latex_table_cpu_whole_only(
                restricted_rows,
                label="tab:flops_cpu_whole_volume_restricted",
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

    gft.compile_pdf(main_path)


if __name__ == "__main__":
    main()
