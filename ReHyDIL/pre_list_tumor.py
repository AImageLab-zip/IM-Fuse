# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List, Tuple
import random
import numpy as np


MASK_DIR: str    = "/path/to/data_root"
RECURSIVE: bool  = True
STRIP_SUFFIX: bool = True
OUT_DIR: str     = "/path/to/data_root/lists"
PREFIX: str      = ""
SEED: int        = 1111
TRAIN_NAME: str  = "train_tumor.list"
VAL_NAME: str    = "val_tumor.list"
TEST_NAME: str   = "test_tumor.list"
SKIP_EMPTY: bool = True               



def is_mask_nonempty(npy_path: Path) -> bool:
    try:
        arr = np.load(npy_path, mmap_mode="r")
    except Exception as e:
        print(f"[WARN] Failed to load {npy_path}: {e}")
        return False


    if arr.dtype == np.bool_:
        return bool(np.any(arr))

    try:

        maxv = np.nanmax(arr) if np.issubdtype(arr.dtype, np.floating) else np.max(arr)
    except ValueError:
        return False

    return bool(maxv > 0)


def collect_nonempty_basenames(mask_dir: Path, recursive: bool = True, strip_suffix: bool = True) -> Tuple[List[str], int, int]:
    pattern = "**/*.npy" if recursive else "*.npy"
    names: List[str] = []
    seen = set()
    total = 0
    skipped_empty = 0

    for p in sorted(mask_dir.glob(pattern)):
        if not p.is_file():
            continue
        total += 1

        if SKIP_EMPTY and not is_mask_nonempty(p):
            skipped_empty += 1
            continue

        name = p.stem if strip_suffix else p.name
        if name not in seen:
            seen.add(name)
            names.append(name)

    return names, total, skipped_empty


def split_8_1_1(names: List[str], seed: int = 1111) -> Tuple[List[str], List[str], List[str]]:
    """Split into 80% train, 10% val, 10% test (deterministic)."""
    rng = random.Random(seed)
    base = sorted(names)
    rng.shuffle(base)

    n = len(base)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val

    train = base[:n_train]
    val = base[n_train:n_train + n_val]
    test = base[n_train + n_val:]
    return train, val, test


def write_list(path: Path, items: List[str]) -> None:
    """Write items to a .list file (one per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(items), encoding="utf-8")


def main():
    mask_dir = Path(MASK_DIR).expanduser().resolve()
    assert mask_dir.exists(), f"Directory does not exist: {mask_dir}"

    names, total, skipped = collect_nonempty_basenames(mask_dir, recursive=RECURSIVE, strip_suffix=STRIP_SUFFIX)
    assert len(names) > 0, "No valid .npy masks found after filtering."

    train, val, test = split_8_1_1(names, seed=SEED)

    out_dir = Path(OUT_DIR).expanduser().resolve()
    write_list(out_dir / f"{PREFIX}{TRAIN_NAME}", train)
    write_list(out_dir / f"{PREFIX}{VAL_NAME}",   val)
    write_list(out_dir / f"{PREFIX}{TEST_NAME}",  test)

    print(f"[OK] scanned={total}  skipped_empty={skipped}  kept={len(names)}")
    print(f"      train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"Saved to: {out_dir}")
    print(f"Files:")
    print(f"  {out_dir / (PREFIX + TRAIN_NAME)}")
    print(f"  {out_dir / (PREFIX + VAL_NAME)}")
    print(f"  {out_dir / (PREFIX + TEST_NAME)}")


if __name__ == "__main__":
    main()
