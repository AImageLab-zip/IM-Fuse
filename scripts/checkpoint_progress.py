#!/usr/bin/env python3
"""Report training progress for every run by comparing checkpoints to configs.

For each packaged config matching the requested dataset (`dataset_type` field
== "brats18" or "brats23"), finds every run directory under that config's
`art_dir` (the base dir itself plus any `<art_dir>_<run_suffix>` variants
created by `mimose train --run-suffix`, e.g. per-fold runs), loads its
checkpoint, and compares the checkpoint's saved epoch against the config's
`num_epochs` to report percent-complete.

Run suffixes that `sbatch_files/` shows are expected (`fold1`, `fold3`,
`fold5` -- the `seed67` std-computation runs are ignored) but have no run
directory on disk at all are reported as a separate "not started" row, so a
run that was never launched is distinguished from one that simply has no
checkpoint yet.

Each row also reports whether that run's test results have been written,
both for the brats split (`<results_dir>/results[_<run_suffix>].txt`) and
the internal split (`<results_dir>/results_internal[_<run_suffix>].txt`).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import torch
import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONFIGS_DIR = os.path.join(REPO_ROOT, "src", "mimose", "data", "configs")
SBATCH_DIR = os.path.join(REPO_ROOT, "sbatch_files")

# Runs to leave out of the report entirely. Each entry is either a bare
# config name to drop every run of that config (e.g. "rfnet_23"), or
# "config_name/run_suffix" to drop just one run (e.g. "uhved_23/fold3").
IGNORED_RUNS: set[str] = set('ciao'
)

TRAIN_INVOCATION_RE = re.compile(
    r"mimose train --config (?P<config>[\w.]+\.yaml).*?--run-suffix (?P<suffix>\w+)"
)


def expected_run_suffixes() -> dict[str, set[str]]:
    """Map each config filename to the run suffixes launched for it in sbatch_files/."""
    expected: dict[str, set[str]] = {}
    for path in glob.glob(os.path.join(SBATCH_DIR, "**", "*.sh"), recursive=True):
        with open(path, errors="ignore") as fh:
            for line in fh:
                match = TRAIN_INVOCATION_RE.search(line)
                if match and match["suffix"] != "seed67":
                    expected.setdefault(match["config"], set()).add(match["suffix"])
    return expected


def load_configs(dataset: str) -> list[dict]:
    configs = []
    for path in sorted(glob.glob(os.path.join(CONFIGS_DIR, "*.yaml"))):
        with open(path) as fh:
            cfg = yaml.safe_load(fh)
        if cfg.get("dataset_type") == dataset:
            cfg["_config_path"] = path
            cfg["_name"] = os.path.basename(path).removesuffix(".yaml")
            configs.append(cfg)
    return configs


def find_run_dirs(art_dir: str) -> list[str]:
    parent = os.path.dirname(art_dir)
    base = os.path.basename(art_dir)
    if not os.path.isdir(parent):
        return []
    candidates = [art_dir] + glob.glob(os.path.join(parent, f"{base}_*"))
    return sorted(d for d in candidates if os.path.isdir(d))


def checkpoint_epoch(run_dir: str, num_epochs: int) -> tuple[int, str]:
    """Return (completed_epochs, source) for a run directory, or (-1, "missing")."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    final_export = os.path.join(ckpt_dir, "final_weights_only.safetensors")
    if os.path.isfile(final_export):
        return num_epochs, "final_weights_only.safetensors"

    for fname in ("model_last.pth", "best.pth"):
        path = os.path.join(ckpt_dir, fname)
        if os.path.isfile(path):
            try:
                # mmap=True avoids materializing the (100+MB) state_dict/optimizer
                # tensors just to read the `epoch` int -- storages are paged in
                # lazily and we never touch them, which matters a lot on
                # leonardo_work's network filesystem.
                ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            except Exception:
                print(f"Failed to load checkpoint: {path}", file=sys.stderr)
                raise
            epoch = int(ckpt.get("epoch", -1)) + 1
            return epoch, fname

    return -1, "missing"


def results_files_status(results_dir: str | None, run_suffix: str | None) -> tuple[bool, bool]:
    """Return (has_brats_results, has_internal_results) for a run.

    Mirrors the naming `apply_run_suffix()` gives `output_path`: the base
    `results.txt` / `results_internal.txt` filenames get `_<run_suffix>`
    spliced in before the extension when a run suffix (e.g. "fold1") is set.
    """
    if not results_dir:
        return False, False

    def path_for(base: str) -> str:
        name = f"{base}_{run_suffix}.txt" if run_suffix else f"{base}.txt"
        return os.path.join(results_dir, name)

    return os.path.isfile(path_for("results")), os.path.isfile(path_for("results_internal"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="brats23", choices=["brats18", "brats23"])
    args = parser.parse_args()

    console = Console()
    configs = load_configs(args.dataset)
    expected = expected_run_suffixes()

    console.rule(f"[bold cyan]Checkpoint progress — {args.dataset}")

    rows: list[list] = []
    pending: dict[int, tuple[str, int]] = {}  # row index -> (run_dir, num_epochs)

    for cfg in configs:
        if cfg["_name"] in IGNORED_RUNS:
            continue

        num_epochs = cfg["num_epochs"]
        config_fname = os.path.basename(cfg["_config_path"])
        results_dir = cfg.get("results_dir")
        run_dirs = find_run_dirs(cfg["art_dir"])
        found_suffixes = {
            os.path.basename(d).removeprefix(f"{os.path.basename(cfg['art_dir'])}_")
            for d in run_dirs
            if d != cfg["art_dir"]
        }

        def ignored(run_suffix: str | None) -> bool:
            return run_suffix is not None and f"{cfg['_name']}/{run_suffix}" in IGNORED_RUNS

        if not run_dirs:
            rows.append([cfg["_name"], None, -1, num_epochs, "no run dir", *results_files_status(results_dir, None)])
        for run_dir in run_dirs:
            run_label = os.path.basename(run_dir)
            run_suffix = None if run_dir == cfg["art_dir"] else run_label.removeprefix(
                f"{os.path.basename(cfg['art_dir'])}_"
            )
            if ignored(run_suffix):
                continue
            has_brats, has_internal = results_files_status(results_dir, run_suffix)
            rows.append([cfg["_name"], run_label, None, num_epochs, None, has_brats, has_internal])
            pending[len(rows) - 1] = (run_dir, num_epochs)

        for missing_suffix in sorted(expected.get(config_fname, set()) - found_suffixes):
            if ignored(missing_suffix):
                continue
            missing_label = f"{os.path.basename(cfg['art_dir'])}_{missing_suffix}"
            has_brats, has_internal = results_files_status(results_dir, missing_suffix)
            rows.append(
                [cfg["_name"], missing_label, -1, num_epochs, "not started", has_brats, has_internal]
            )

    # checkpoint_epoch() is I/O-bound (reads .pth files off leonardo_work over
    # the network); running the loads for every run dir in parallel is what
    # actually makes this fast, on top of the mmap=True in checkpoint_epoch.
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            idx: executor.submit(checkpoint_epoch, run_dir, num_epochs)
            for idx, (run_dir, num_epochs) in pending.items()
        }
        for idx, future in futures.items():
            epoch, source = future.result()
            rows[idx][2] = epoch
            rows[idx][4] = source

    progress = Progress(
        TextColumn("[bold]{task.fields[label]}", justify="left"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TextColumn("[dim]{task.fields[detail]}"),
        console=console,
        expand=True,
    )

    def results_suffix(has_brats: bool, has_internal: bool) -> str:
        brats_mark = "[green]brats✓[/green]" if has_brats else "[red]brats✗[/red]"
        internal_mark = "[green]internal✓[/green]" if has_internal else "[red]internal✗[/red]"
        return f"{brats_mark} {internal_mark}"

    with progress:
        for config_name, run_label, epoch, num_epochs, source, has_brats, has_internal in rows:
            label = config_name if run_label is None else f"{config_name} ({run_label})"
            results = results_suffix(has_brats, has_internal)
            if epoch < 0:
                color = "yellow" if source == "not started" else "red"
                progress.add_task(
                    "",
                    total=1,
                    completed=0,
                    label=label,
                    detail=f"[{color}]{source}[/{color}] · {results}",
                )
                continue
            pct_epoch = min(epoch, num_epochs)
            detail = f"[green]{pct_epoch}/{num_epochs} epochs[/green] · {source} · {results}"
            if pct_epoch >= num_epochs:
                detail = f"[bold green]done[/bold green] · {source} · {results}"
            progress.add_task(
                "", total=num_epochs, completed=pct_epoch, label=label, detail=detail
            )

    total = len(rows)
    done = sum(1 for _c, _r, epoch, num_epochs, _s, _hb, _hi in rows if epoch >= num_epochs)
    not_started = sum(1 for r in rows if r[4] == "not started")
    missing = sum(1 for r in rows if r[2] < 0 and r[4] != "not started")
    missing_brats_results = sum(1 for r in rows if not r[5])
    missing_internal_results = sum(1 for r in rows if not r[6])
    summary = f"[bold]{done}/{total} runs complete[/bold]"
    if missing:
        summary += f" · [red]{missing} missing checkpoints[/red]"
    if not_started:
        summary += f" · [yellow]{not_started} not started[/yellow]"
    summary += f" · [red]{missing_brats_results} missing brats results[/red]"
    summary += f" · [red]{missing_internal_results} missing internal results[/red]"
    console.rule(summary)


if __name__ == "__main__":
    main()
