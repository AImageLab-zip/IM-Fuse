#!/usr/bin/env python3
"""BraTS25-pre (= BraTS23), fold1-only 2D Pareto plot: whole-volume FLOPs vs.
average Dice.

The compute-cost twin of generate_pareto_plot_latency_fold1.py: identical data
collection and rendering (both reuse generate_pareto_plot.py's
collect_points/render_pareto_plot), with FLOPs on the x axis instead of
measured GPU latency. Every model is scored on BraTS25-pre (dsnum 23), fold1
only -- the single evaluation the MiMoSe-P budgets have, so the comparison is
like for like; see that script's docstring for why the unrestricted version
misleads.

FLOPs are the hardware-independent cost axis -- the same number on any device
-- where latency is what a particular GPU actually delivered, so a model's
Pareto status can differ between the two plots wherever an architecture maps
unusually well or badly onto the hardware.

Plotted in TFLOPs (raw counts / 1e12), matching the FLOPs column of
generate_pareto_table.py's tab:pareto.

Writes pareto_dice_vs_flops_fold1.png into generate_pareto_table.py's
OUTPUT_DIR.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_pareto_plot as gpp  # noqa: E402
import pareto_budgets as pbud  # noqa: E402

FOLDS = ["fold1"]

# Every model is scored on BraTS25-pre only -- the sole evaluation the
# MiMoSe-P budgets have (see pareto_budgets.GROUPS). Without this restriction
# the other families' Dice is a mean that also folds in BraTS18 and the
# internal MB-96 cohort, which score several points lower, so the budgets
# would appear ~6 Dice points ahead of models that actually beat them on the
# shared dataset (UNET-MFI: 84.3 on BraTS25-pre vs. 78.4 as a 4-group mean).
GROUPS = pbud.GROUPS

# Raw FLOP counts -> TFLOPs, the unit generate_performance_table.fmt_tflops
# reports and the tables use.
TFLOPS = 1e12


def main() -> None:
    os.makedirs(gpp.OUTPUT_DIR, exist_ok=True)
    points, families = gpp.collect_points("flops", x_scale=TFLOPS, expected_folds=FOLDS, groups=GROUPS)
    if not points:
        print("No models with both fold1 Dice and FLOPs data", file=sys.stderr)
        return
    gpp.render_pareto_plot(
        points,
        families,
        x_label="Whole-volume FLOPs (T)",
        out_path=os.path.join(gpp.OUTPUT_DIR, "pareto_dice_vs_flops_fold1.png"),
    )


if __name__ == "__main__":
    main()
