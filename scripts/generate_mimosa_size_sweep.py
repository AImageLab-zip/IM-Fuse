"""Generate configs + sbatch trees for the mimosa size sweep.

Sweep base is SimpleUnet (see mimose.models.simple_unet): TinyMimosa's
identical backbone, trained on 128x128x128 patches with IM-Fuse-style
preprocessing/cropping. Only features_per_stage (channel width) varies
across the sweep; model/trainer/loss/transforms are exactly the simpleunet
ones. Unlike simpleunet, all mimosa_[size] configs (every size, every
dataset) share one preprocessed folder, mimosa_preprocessed, distinct from
simpleunet's own simpleunet-preprocessed -- see MIMOSA_PREPROCESSED_DIR.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # repo root; hardcode if this is wrong

SIZES = {
    "mimosa_small":  (8, 16, 32, 64, 128),
    "mimosa_medium": (16, 32, 64, 128, 256),
    "mimosa_large":  (32, 64, 128, 256, 512),
    "mimosa_huge":   (64, 128, 256, 512, 512),
}

# All mimosa_[size] configs (every size, every dataset) point at this one
# preprocessed folder rather than a per-size one -- same IM-Fuse-style
# preprocessing (crop/clamp/norm) and 128x128x128 patches as simpleunet's
# own simpleunet-preprocessed, just named/shared separately.
MIMOSA_PREPROCESSED_DIR = "mimosa_preprocessed"

# Substrings that must survive the rename untouched (see §2).
PRESERVE = [
    "simpleunet-preprocessed",
    "simpleunet_internal",
    "trainer: imfuse",
    "model: simpleunet",
    "loss: tinymimosa",
]


def rename(text: str, name: str) -> str:
    holes = {}
    for i, token in enumerate(PRESERVE):
        key = f"\x00P{i}\x00"
        holes[key] = token
        text = text.replace(token, key)

    text = text.replace("SIMPLEUNET", name.upper())
    text = text.replace("SimpleUnet", "".join(p.capitalize() for p in name.split("_")))
    text = text.replace("simpleunet", name)

    for key, token in holes.items():
        text = text.replace(key, token)
    return text


def build_config(src: Path, name: str, features: tuple[int, ...]) -> str:
    text = rename(src.read_text(), name)
    text = text.replace("simpleunet-preprocessed", MIMOSA_PREPROCESSED_DIR)

    # Drop any inherited features_per_stage (incl. the broken string form) first.
    text = re.sub(r"^ +features_per_stage:.*\n", "", text, flags=re.MULTILINE)

    line = f"  features_per_stage: [{', '.join(map(str, features))}]\n"
    text = re.sub(
        r"(^custom_model_kwargs:\n(?:^ +\S.*\n)*)",
        lambda m: m.group(1) + line,
        text, count=1, flags=re.MULTILINE,
    )

    header = (
        f"# {name} = SimpleUnet with features_per_stage={tuple(features)}\n"
        f"# (128x128x128 patches, IM-Fuse-style preprocessing -- see\n"
        f"# mimose.models.simple_unet). Model/trainer/loss/transforms match the\n"
        f"# simpleunet ones; only the per-stage width changes, so the sweep is\n"
        f"# directly comparable. All mimosa_[size] configs (every size, every\n"
        f"# dataset) share one preprocessed folder, {MIMOSA_PREPROCESSED_DIR}.\n"
    )
    lines = text.splitlines(keepends=True)
    body = next(i for i, ln in enumerate(lines) if not ln.startswith("#"))
    return header + "".join(lines[body:])


def gen_configs() -> int:
    n = 0
    for cfg_dir in (ROOT / "src/mimose/data/configs",
                    ROOT / "src/mimose/data/config_templates"):
        for dataset in ("18", "23", "25"):
            src = cfg_dir / f"simpleunet_{dataset}.yaml"
            for name, feats in SIZES.items():
                (cfg_dir / f"{name}_{dataset}.yaml").write_text(build_config(src, name, feats))
                n += 1
    return n


def gen_sbatch() -> int:
    n = 0
    for tree in (ROOT / "sbatch_files", ROOT / "sbatch_cineca"):
        src_tree = tree / "simpleunet"
        if not src_tree.is_dir():
            continue
        for name in SIZES:
            dst = tree / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src_tree, dst)
            for path in sorted(dst.rglob("*.sh")):
                path.write_text(rename(path.read_text(), name))
                new = path.with_name(path.name.replace("simpleunet", name))
                if new != path:
                    path.rename(new)
                    path = new
                path.chmod(0o755)
                n += 1
    return n


if __name__ == "__main__":
    print(f"{gen_configs()} configs, {gen_sbatch()} sbatch scripts")
