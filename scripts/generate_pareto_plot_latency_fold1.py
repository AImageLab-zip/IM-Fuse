#!/usr/bin/env python3
"""BraTS25-pre (= BraTS23), fold1-only 2D Pareto plot: whole-volume GPU
latency vs. average Dice.

Same plot as generate_pareto_plot.py (same families, same markers, same zoom
inset -- it reuses that script's collect_points/render_pareto_plot outright),
but with every model put on exactly one evaluation:

  * one dataset -- BraTS25-pre (dsnum 23), instead of a mean over BraTS18,
    BraTS25-pre and the internal MB-96 cohort under both checkpoints;
  * one fold -- fold1, instead of the fold1/fold3/fold5 mean.

Both restrictions exist for the same reason: the MiMoSe-P supernet budgets
only ever trained and tested fold1 on BraTS25-pre (see pareto_budgets.py), so
on generate_pareto_plot.py they are compared against numbers averaged over
data they were never evaluated on. That inflates them badly -- BraTS18 and
MB-96 score several points lower than BraTS25-pre, so a 4-group mean drags
every other model down: UNET-MFI reads 78.4 there but scores 84.3 on
BraTS25-pre alone, ahead of every budget. This plot removes that artifact.

Side effect of dropping the completeness bar to a single fold: any model that
ran fold1 but not all three now qualifies, so this plot can show slightly more
points than the 3-fold one. Conversely, restricting to one dataset drops any
model with no BraTS25-pre results.

Writes pareto_dice_vs_latency_fold1.png into generate_pareto_table.py's
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


def main() -> None:
    os.makedirs(gpp.OUTPUT_DIR, exist_ok=True)
    points, families = gpp.collect_points("lat_gpu", expected_folds=FOLDS, groups=GROUPS)
    if not points:
        print("No models with both fold1 Dice and GPU latency data", file=sys.stderr)
        return
    gpp.render_pareto_plot(
        points,
        families,
        x_label="Whole-volume GPU latency (s)",
        out_path=os.path.join(gpp.OUTPUT_DIR, "pareto_dice_vs_latency_fold1.png"),
    )


if __name__ == "__main__":
    main()
