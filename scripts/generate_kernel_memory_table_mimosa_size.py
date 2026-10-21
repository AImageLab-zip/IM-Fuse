#!/usr/bin/env python3
"""mimosa_[size] counterpart of generate_kernel_memory_table.py.

Same kernel-launch-count / memory-pressure table, restricted instead to the
mimosa_[size] family (mimosa_base, mimosa_gargantuan, mimosa_huge,
mimosa_large, mimosa_medium, mimosa_micro, mimosa_small, mimosa_tiny -- see
sbatch_files/flops/flops_mimosa_size_gpu.sh), sorted alphabetically by size
name rather than by kernel-launch count, since the point here is comparing
the family across its own size ladder rather than ranking against unrelated
architectures.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_flops_table as gft  # noqa: E402
import generate_kernel_memory_table as gkmt  # noqa: E402

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "kernel_memory_table_mimosa_size"
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

# Display name / sort order follows this fixed size ladder (small to huge),
# not gkmt's default kernel-launch-count sort, since the point of this table
# is comparing the family across its own size progression.
_SIZE_ORDER = {gft.normalize(f"mimosa_{size}"): i for i, size in enumerate(MIMOSA_SIZES)}


def sort_by_size(
    rows: list[tuple[str, dict[str, str]]],
) -> list[tuple[str, dict[str, str]]]:
    return sorted(rows, key=lambda r: _SIZE_ORDER.get(gft.normalize(r[0]), len(_SIZE_ORDER)))


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = gkmt.collect_kernel_memory_rows(gft.RESULTS_DIR, allowed_models=MIMOSA_SIZE_MODELS)
    if not rows:
        print(
            f"No {gkmt.CSV_FILENAME} files with kernel-launch/reserved-memory data found "
            f"for the mimosa_[size] family under {gft.RESULTS_DIR}/<model>_23/",
            file=sys.stderr,
        )
        return
    rows = sort_by_size(rows)

    gkmt.print_console_table(rows, title_suffix=" (mimosa_[size] family)")

    tables_path = os.path.join(OUTPUT_DIR, "tables.tex")
    with open(tables_path, "w") as f:
        f.write(gkmt.build_latex_table(rows, label="tab:kernel_memory_mimosa_size"))
    print(f"Wrote {tables_path}")


if __name__ == "__main__":
    main()
