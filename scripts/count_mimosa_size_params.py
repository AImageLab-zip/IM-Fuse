"""Print param counts for each mimosa size variant of SimpleUnet.

Sizes are features_per_stage tuples for the shared TinyMimosa/SimpleUnet
backbone (see mimose.models.simple_unet, mimose.models.tiny_mimosa) --
same variants as scripts/generate_mimosa_size_sweep.py.
"""
from __future__ import annotations

from mimose.models.simple_unet import SimpleUnet

SIZES = {
    "mimosa_micro":  (16, 32, 48, 64),
    "mimosa_tiny":  (16, 32, 64, 128),
    "mimosa_small":  (16, 32, 64, 128, 256),
    "mimosa_medium": (24, 48, 96, 192, 384),
    "mimosa_base":   (32, 64, 128, 256, 512),
    "mimosa_large":  (40, 80, 160, 320, 640),
    "mimosa_huge":   (48, 96, 192, 384, 768),
    "mimosa_gargantuan": (64, 128, 256, 512, 1024),
}


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for name, features in SIZES.items():
        model = SimpleUnet(features_per_stage=features)
        n_params = count_params(model)
        print(f"{name:15s} features_per_stage={features!s:30s} params={n_params:,}")
