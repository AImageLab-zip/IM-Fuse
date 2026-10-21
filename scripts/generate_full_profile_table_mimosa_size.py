#!/usr/bin/env python3
"""mimosa_[size] counterpart of generate_full_profile_table.py.

Same full-profiler table (params, FLOPs, peak allocated/reserved memory,
memory overhead, kernel launches, and latency, patch and whole-volume side
by side), restricted instead to the mimosa_[size] family (mimosa_base,
mimosa_gargantuan, mimosa_huge, mimosa_large, mimosa_medium, mimosa_micro,
mimosa_small, mimosa_tiny -- see sbatch_files/flops/flops_mimosa_size_gpu.sh).
Sorted by whole-volume FLOPs like generate_full_profile_table.py's default
(most efficient model first). Display names use the short `mb_<size>` form
(e.g. `mb_base`) rather than gft.DISPLAY_NAMES' raw upper-cased fallback
(`MIMOSABASE`).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_flops_table as gft  # noqa: E402
import generate_full_profile_table as gfpt  # noqa: E402

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "full_profile_table_mimosa_size"
)

MIMOSA_SIZES = (
    "base",
    "gargantuan",
    "huge",
    "large",
    "medium",
    "micro",
    "small",
    "tiny",
)
MIMOSA_SIZE_MODELS = {gft.normalize(f"mimosa_{size}") for size in MIMOSA_SIZES}

# model_norm -> short display name, e.g. "mimosabase" -> "mb_base".
_DISPLAY_NAMES = {gft.normalize(f"mimosa_{size}"): f"mb_{size}" for size in MIMOSA_SIZES}


def rename_rows(
    rows: list[tuple[str, dict[str, dict[str, str] | None]]],
) -> list[tuple[str, dict[str, dict[str, str] | None]]]:
    return [
        (_DISPLAY_NAMES.get(gft.normalize(display), display), scopes)
        for display, scopes in rows
    ]


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = gfpt.collect_full_profile_rows(gft.RESULTS_DIR, allowed_models=MIMOSA_SIZE_MODELS)
    if not rows:
        print(
            f"No {gfpt.CSV_FILENAME} files with kernel-launch/reserved-memory data found "
            f"for the mimosa_[size] family under {gft.RESULTS_DIR}/<model>_23/",
            file=sys.stderr,
        )
        return
    rows = rename_rows(rows)

    gfpt.print_console_table(rows, title_suffix=" (mimosa_[size] family)")

    tables_path = os.path.join(OUTPUT_DIR, "tables.tex")
    with open(tables_path, "w") as f:
        f.write(gfpt.build_latex_table(rows, label="tab:full_profile_mimosa_size"))
    print(f"Wrote {tables_path}")


if __name__ == "__main__":
    main()
