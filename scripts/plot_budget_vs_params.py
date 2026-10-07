"""Milestone figure: requested parameter budget vs. actually extracted parameters.

This is the gate on the whole Pareto-supernet effort.  If the curve is monotonic
and hugs the diagonal, the budget -> architecture map works and multi-budget
BraTS training can start; if it does not, no amount of segmentation training
will fix it.

Runs on CPU in seconds and touches no data: the budget -> architecture map is
pure arithmetic over the gate thresholds.

    python scripts/plot_budget_vs_params.py
    python scripts/plot_budget_vs_params.py --points 300 --group-size 8
"""

from __future__ import annotations

import argparse
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mimose.nas.budget import budget_to_z  # noqa: E402
from mimose.nas.calibration import (  # noqa: E402
    budget_param_curve,
    calibrate_thresholds_analytic,
)
from mimose.nas.gates import WidthGates  # noqa: E402
from mimose.nas.param_count import exact_parameters  # noqa: E402

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "pareto_supernet"
)

DEFAULT_MAX_FEATURES = (64, 128, 256, 512, 1024)

# The hand-written sweep points that live inside this supernet's range, for
# orientation: the learned family should pass close to them.
REFERENCE_MODELS = {
    "tiny": (16, 32, 64, 128),
    "small": (16, 32, 64, 128, 256),
    "medium": (24, 48, 96, 192, 384),
    "base": (32, 64, 128, 256, 512),
    "large": (40, 80, 160, 320, 640),
    "huge": (48, 96, 192, 384, 768),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument(
        "--max-features",
        type=int,
        nargs="+",
        default=list(DEFAULT_MAX_FEATURES),
        help="Widths of the widest architecture in the space.",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    max_features = tuple(args.max_features)

    # The expressible range: uniform 1/8 width up to full width.
    budget_min = float(
        exact_parameters(tuple(max(c // 8, args.group_size) for c in max_features))
    )
    budget_max = float(exact_parameters(max_features))

    gates = WidthGates(
        max_features,
        budget_min=budget_min,
        budget_max=budget_max,
        group_size=args.group_size,
        tau=args.tau,
    )
    calibrate_thresholds_analytic(gates)

    rows = budget_param_curve(gates, n=args.points)
    requested = [b for b, _, _ in rows]
    extracted = [hard for _, hard, _ in rows]

    errors = [abs(h - b) / b for b, h, _ in rows]
    monotonic = all(a <= b for a, b in zip(extracted, extracted[1:]))

    print(f"supernet widths : {max_features}")
    print(f"budget range    : {budget_min:,.0f} .. {budget_max:,.0f}")
    print(f"gate parameters : {sum(p.numel() for p in gates.parameters())}")
    print(f"monotonic       : {monotonic}")
    print(f"mean rel. error : {statistics.mean(errors):.2%}")
    print(f"max  rel. error : {max(errors):.2%}")
    print()
    print(f"{'requested':>14s} {'extracted':>14s} {'err':>8s}  widths")
    for index in range(0, len(rows), max(len(rows) // 12, 1)):
        budget, hard, _ = rows[index]
        z = budget_to_z(budget, budget_min, budget_max)
        widths = tuple(int(w) for w in gates.hard_widths(z).reshape(-1))
        print(f"{budget:14,.0f} {hard:14,d} {(hard - budget) / budget:+7.2%}  {widths}")

    # ---- figure -------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    fig, (ax, ax_err) = plt.subplots(
        2, 1, figsize=(7.2, 7.4), height_ratios=[3, 1], sharex=True
    )

    diagonal = [budget_min, budget_max]
    ax.plot(diagonal, diagonal, "--", color="0.6", lw=1.2, label="ideal ($y=x$)")
    ax.plot(requested, extracted, "-", color="#1f77b4", lw=1.8, label="extracted A(b)")

    for name, widths in REFERENCE_MODELS.items():
        params = exact_parameters(widths)
        if budget_min <= params <= budget_max:
            ax.plot([params], [params], "o", color="#d62728", ms=5, zorder=5)
            ax.annotate(
                name,
                (params, params),
                textcoords="offset points",
                xytext=(6, -10),
                fontsize=8,
                color="#d62728",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("extracted parameters")
    ax.set_title(
        f"Budget-conditioned architecture family  "
        f"(S={len(max_features)}, group={args.group_size})"
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", frameon=False)

    ax_err.axhline(0.0, color="0.6", ls="--", lw=1.0)
    ax_err.plot(
        requested,
        [100.0 * (h - b) / b for b, h, _ in rows],
        color="#1f77b4",
        lw=1.2,
    )
    ax_err.set_xscale("log")
    ax_err.set_xlabel("requested parameter budget")
    ax_err.set_ylabel("error [%]")
    ax_err.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    figure_path = os.path.join(args.output_dir, "budget_vs_params.png")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    csv_path = os.path.join(args.output_dir, "budget_vs_params.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("requested_budget,extracted_params,soft_params,widths\n")
        for budget, hard, soft in rows:
            z = budget_to_z(budget, budget_min, budget_max)
            widths = " ".join(str(int(w)) for w in gates.hard_widths(z).reshape(-1))
            handle.write(f"{budget:.6f},{hard},{soft:.6f},{widths}\n")

    print()
    print(f"figure -> {figure_path}")
    print(f"csv    -> {csv_path}")

    if not monotonic:
        print("\nFAIL: extracted parameter count is not monotone in the budget")
        return 1
    if statistics.mean(errors) > 0.05:
        print("\nFAIL: mean relative error exceeds 5%")
        return 1
    print("\nPASS: monotone and on the diagonal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
