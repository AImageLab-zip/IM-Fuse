#!/usr/bin/env python3
"""Rank models on a per-dataset Dice/HD95 leaderboard, built from the same
per-fold result workbooks as `generate_summary_tables.py` and
`build_reproduction_comparison.py`.

Emits one leaderboard per dataset found under RESULTS_DIR (BraTS 2018, 2023,
...) rather than pooling models across datasets, since a model that was only
run on one dataset shouldn't be penalized (or boosted) by an average that
partly reflects on which datasets it happened to be evaluated.

A model is scored on a dataset only if it has COMPLETE official-test-set
results for it (all of EXPECTED_FOLDS present, each covering all 15
modality-presence combinations, every WT/TC/ET Dice and HD95 value
populated) -- same completeness rule as generate_summary_tables.py. Per
dataset/model:
  Dice score = mean over WT/TC/ET of (mean over the 15 modality combinations
               of the per-combination mean over folds)
  HD95 score = the same, for HD95
This mirrors the "Dice score <ds>" / "HD95 score <ds>" rows that
build_reproduction_comparison.py writes into each model's tab.

Each dataset's leaderboard is sorted by Dice score (descending) by default;
pass --sort-by hd95 to sort by HD95 (ascending -- lower is better) instead.

Prints one rich table per dataset to the console and, unless --no-markdown
is passed, writes one Markdown file per dataset to
outputs/leaderboard_<dsnum>.md.
"""

from __future__ import annotations

import argparse
import os
import sys

from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_summary_tables as gst  # noqa: E402  (reuse discover/is_complete/aggregate)

RESULTS_DIR = gst.RESULTS_DIR
LEGACY_FILE = gst.LEGACY_FILE
REGIONS = gst.REGIONS
EXCLUDED_MODELS = gst.EXCLUDED_MODELS
DATASET_LABELS = gst.DATASET_LABELS
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")


def dataset_score(folds: dict) -> dict[str, float] | None:
    """{"dice": .., "hd95": ..} for one model/dataset, or None if incomplete."""
    if not gst.is_complete(folds):
        return None
    agg = gst.aggregate(folds)
    dice_means = [agg[region]["Dice"][0] for region in REGIONS]
    hd95_means = [agg[region]["HD95"][0] for region in REGIONS]
    return {"dice": sum(dice_means) / len(dice_means), "hd95": sum(hd95_means) / len(hd95_means)}


def build_leaderboard(models: dict, dsnum: str, display_names: dict) -> list[dict]:
    """One row per model with complete results on this dataset:
    {"name": display, "dice":, "hd95":}."""
    rows = []
    for model_norm, by_dataset in models.items():
        if model_norm in EXCLUDED_MODELS:
            continue
        folds = by_dataset.get(dsnum)
        if not folds:
            continue
        score = dataset_score(folds)
        if score is None:
            continue
        rows.append(
            {
                "name": display_names.get(model_norm, model_norm.upper()),
                "dice": score["dice"],
                "hd95": score["hd95"],
            }
        )
    return rows


def render_console(rows: list[dict], label: str, console: Console) -> None:
    table = Table(title=f"Model Leaderboard — {label}", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("Model")
    table.add_column("Dice", justify="right", style="bold")
    table.add_column("HD95", justify="right")

    for i, row in enumerate(rows, start=1):
        style = "bold green" if i == 1 else None
        table.add_row(str(i), row["name"], f"{row['dice']:.1f}", f"{row['hd95']:.1f}", style=style)

    console.print(table)


def render_markdown(rows: list[dict], label: str) -> str:
    lines = [
        f"# Model Leaderboard — {label}",
        "",
        "| # | Model | Dice | HD95 |",
        "|---|---|---|---|",
    ]
    for i, row in enumerate(rows, start=1):
        lines.append(f"| {i} | {row['name']} | {row['dice']:.1f} | {row['hd95']:.1f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sort-by", choices=["dice", "hd95"], default="dice",
        help="Rank by Dice (descending, higher is better) or HD95 (ascending, "
        "lower is better). Default: dice.",
    )
    parser.add_argument(
        "--no-markdown", action="store_true",
        help=f"Skip writing the per-dataset Markdown files to {OUTPUT_DIR}.",
    )
    args = parser.parse_args()

    console = Console()

    display_names = {}
    legacy_path = os.path.join(RESULTS_DIR, LEGACY_FILE)
    if os.path.exists(legacy_path):
        import build_reproduction_comparison as brc

        _, display_names = brc.parse_legacy(legacy_path)

    models, _internal_models, dsnums = gst.discover(RESULTS_DIR)
    if not dsnums:
        console.print(f"[red]No result directories found in {RESULTS_DIR}[/red]")
        return

    if not args.no_markdown:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    for dsnum in dsnums:
        label = DATASET_LABELS.get(dsnum, f"Dataset {dsnum}")
        rows = build_leaderboard(models, dsnum, display_names)
        if not rows:
            console.print(f"[yellow]{label}: no model has complete results, skipping.[/yellow]")
            continue

        if args.sort_by == "dice":
            rows.sort(key=lambda r: r["dice"], reverse=True)
        else:
            rows.sort(key=lambda r: r["hd95"])

        render_console(rows, label, console)

        if not args.no_markdown:
            out_path = os.path.join(OUTPUT_DIR, f"leaderboard_{dsnum}.md")
            with open(out_path, "w") as f:
                f.write(render_markdown(rows, label))
            console.print(f"Wrote {out_path}\n")


if __name__ == "__main__":
    main()
