#!/usr/bin/env python3
"""Print the parameter count of each of ManyMimosas' 15 per-modality-pattern
TinyMimosa subnets (one per non-empty modality-presence combination)."""

from __future__ import annotations

from mimose.models.many_mimosas import ManyMimosas

MODALITY_NAMES = ("t1c", "t1n", "t2f", "t2w")


def pattern_label(key: str) -> str:
    present = [name for name, bit in zip(MODALITY_NAMES, key) if bit == "1"]
    return "+".join(present)


def count_parameters(module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    model = ManyMimosas()

    rows = sorted(
        model.submodels.items(),
        key=lambda kv: (kv[0].count("1"), kv[0]),
    )
    total = 0
    for key, submodel in rows:
        params = count_parameters(submodel)
        total += params
        print(f"{pattern_label(key):15s} ({key})  {params / 1e6:8.3f}M params")

    print(f"\n{'TOTAL':15s} {'':7s}  {total / 1e6:8.3f}M params")


if __name__ == "__main__":
    main()
