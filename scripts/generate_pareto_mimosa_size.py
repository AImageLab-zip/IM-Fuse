#!/usr/bin/env python3
"""Pareto-frontier plots (whole-volume GPU latency vs. average Dice) for the
mimosa_size compute-scaling family.

Reads whole-volume latency from each mimosa_size model's `flops_gpu.csv`
(written by `mimose flops`, see scripts/generate_flops_table.py for the
paper-models counterpart -- mimosa_size models don't have a plain
`flops.csv`, only `flops_gpu.csv`/`flops_cpu.csv`) and average Dice from
`reproduction-comparison.xlsx` (same workbook/loader as
generate_latex_tables_mimosa_size.py), one plot per slice: BRATS2018,
BRATS2023, and MB-96 (internal dataset) using checkpoints trained on each
of those two.

Writes one PNG per slice into OUTPUT_DIR.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_flops_table as gft  # noqa: E402
import generate_latex_tables as glt  # noqa: E402
import generate_latex_tables_mimosa_size as gsize  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUTPUT_DIR = os.path.join(glt.RESULTS_DIR, "pareto_mimosa_size")

# One plot per (dataset, use_internal) slice: repro on BRATS2018/BRATS2023
# themselves, plus MB-96 generalization using each dataset's checkpoints.
SLICES = [
    ("BRATS2018", False, "BRATS2018", "brats2018"),
    ("BRATS2023", False, "BRATS2023", "brats2023"),
    ("BRATS2018", True, "MB-96 (BraTS18 checkpoints)", "mb96_brats2018"),
    ("BRATS2023", True, "MB-96 (BraTS25-pre checkpoints)", "mb96_brats2023"),
]


def load_latency(results_dir):
    """{mimosa_size display name: whole-volume GPU latency in seconds}.
    Mimosa_size models only ship flops_gpu.csv/flops_cpu.csv, not the plain
    flops.csv the paper-models discovery in generate_flops_table.py expects,
    so this reads the GPU one directly rather than going through
    collect_rows (which also brings in DISPLAY_NAMES formatting that isn't
    relevant here)."""
    out = {}
    for model_norm, csv_path in gft.discover_flops_csvs(
        results_dir, filename="flops_gpu.csv"
    ).items():
        if not model_norm.startswith("mimosa"):
            continue
        size = model_norm.removeprefix("mimosa")
        if size not in gsize.SIZE_DISPLAY:
            continue
        whole = gft.load_scoped_rows(csv_path).get("whole_volume")
        if whole is None or not whole.get("mean_latency_seconds"):
            continue
        out[gsize.SIZE_DISPLAY[size]] = float(whole["mean_latency_seconds"])
    return out


def avg_dice(models, dataset, use_internal):
    """{model_display: average Dice% across the WT/TC/ET region averages}."""
    region_avgs = {}
    for region in glt.REGIONS:
        _, row_records = glt.collect_row_records(
            models, dataset, region, "Dice", source="repro", use_internal=use_internal
        )
        for name, _cells, avg_cell, _diff in row_records:
            region_avgs.setdefault(name, []).append(avg_cell[1])
    return {
        name: sum(values) / len(values)
        for name, values in region_avgs.items()
        if all(v is not None for v in values)
    }


def pareto_frontier(points):
    """Names of the non-dominated points (maximize dice, minimize latency)."""
    frontier = []
    for name, (lat, dice) in points.items():
        dominated = any(
            other_lat <= lat
            and other_dice >= dice
            and (other_lat < lat or other_dice > dice)
            for other_name, (other_lat, other_dice) in points.items()
            if other_name != name
        )
        if not dominated:
            frontier.append(name)
    return frontier


def build_pareto_plot(latency, dice, title, out_path):
    points = {name: (latency[name], dice[name]) for name in dice if name in latency}
    if not points:
        print(f"No overlapping models for {title!r}, skipping", file=sys.stderr)
        return
    frontier = set(pareto_frontier(points))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    cmap = plt.get_cmap("viridis")
    n_sizes = len(gsize.MIMOSA_SIZES)
    for name in gsize.sort_by_size(points.keys()):
        lat, d = points[name]
        size = name.split("\\_", 1)[1]
        color = cmap(gsize.SIZE_ORDER[size] / max(n_sizes - 1, 1))
        on_frontier = name in frontier
        ax.scatter(
            lat,
            d,
            color=color,
            marker="o" if on_frontier else "x",
            s=70,
            zorder=3,
            edgecolors="black" if on_frontier else "none",
            linewidths=0.8,
        )
        ax.annotate(
            name.replace("mimosa\\_", "mimosa_"),
            (lat, d),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )

    frontier_pts = sorted((points[name] for name in frontier), key=lambda p: p[0])
    if len(frontier_pts) > 1:
        fx, fy = zip(*frontier_pts)
        ax.plot(fx, fy, color="black", linestyle="--", linewidth=1, alpha=0.6, zorder=2)

    ax.set_xlabel("Whole-volume GPU latency (s)")
    ax.set_ylabel("Average Dice (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    models = gsize.load_mimosa_size(glt.WORKBOOK)
    if not models:
        print("No mimosa_size sheets found in the workbook", file=sys.stderr)
        return

    latency = load_latency(glt.RESULTS_DIR)
    if not latency:
        print("No mimosa_size flops_gpu.csv files found", file=sys.stderr)
        return

    for dataset, use_internal, title, slug in SLICES:
        dice = avg_dice(models, dataset, use_internal)
        out_path = os.path.join(OUTPUT_DIR, f"pareto_{slug}.png")
        build_pareto_plot(latency, dice, f"mimosa_size: Dice vs. Latency -- {title}", out_path)


if __name__ == "__main__":
    main()
