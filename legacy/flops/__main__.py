from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

from rich.console import Console

from .core import (
    build_flops_table,
    get_method_spec,
    method_names,
    report_to_dict,
    run_flops_for_method,
    run_flops_subprocess,
)


console = Console()
progress_console = Console(stderr=True)


def _parse_shape(values: list[str] | None) -> tuple[int, int, int] | None:
    if values is None:
        return None
    if len(values) != 3:
        raise argparse.ArgumentTypeError("shape values must be exactly 3 integers")
    return tuple(int(v) for v in values)


def _write_json(target: str, payload: Any) -> None:
    text = json.dumps(payload, indent=2)
    if target == "-":
        sys.stdout.write(text)
        sys.stdout.write("\n")
    else:
        Path(target).write_text(text + "\n")


def _write_csv(target: str, reports: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for report in reports:
        for measurement in report["measurements"]:
            rows.append(measurement)
    if not rows:
        return
    if target == "-":
        handle = sys.stdout
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        return
    with Path(target).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _measurement_status_summary(report: dict[str, Any]) -> str:
    measurements = report.get("measurements", [])
    if not measurements:
        return "no measurements"
    parts = []
    for measurement in measurements:
        name = measurement.get("measurement", "unknown")
        status = measurement.get("status", "unknown")
        parts.append(f"{name}={status}")
    return ", ".join(parts)


def _failure_summary(report: dict[str, Any]) -> str | None:
    for measurement in report.get("measurements", []):
        if measurement.get("status") == "ok":
            continue
        name = measurement.get("measurement", "unknown")
        note = str(measurement.get("note", "")).strip()
        if note:
            return f"{name} failed: {note}"
        return f"{name} failed"
    return None


def _run_all_methods(
    *,
    batch_size: int,
    measure: str,
    shape: tuple[int, int, int] | None,
    full_volume_shape: tuple[int, int, int] | None,
    verbose: bool,
) -> list[dict[str, Any]]:
    names = method_names()
    reports: list[dict[str, Any]] = []

    for index, name in enumerate(names, start=1):
        progress_console.print(f"[{index}/{len(names)}] Running {name}...")
        report = run_flops_subprocess(
            name,
            batch_size=batch_size,
            measure=measure,
            shape=shape,
            full_volume_shape=full_volume_shape,
        )
        reports.append(report)

        summary = _measurement_status_summary(report)
        if verbose:
            progress_console.print(f"[{index}/{len(names)}] Finished {name}: {summary}")
        else:
            failure = _failure_summary(report)
            outcome = failure or "ok"
            progress_console.print(f"[{index}/{len(names)}] Finished {name}: {outcome}")

    return reports


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m legacy.flops")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--json", nargs="?", const="-")

    run_parser = subparsers.add_parser("run")
    group = run_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--method")
    group.add_argument("--all", action="store_true")
    run_parser.add_argument("--batch-size", type=int, default=1)
    run_parser.add_argument(
        "--measure",
        choices=("forward", "full_volume_forward", "all"),
        default="all",
    )
    run_parser.add_argument("--shape", nargs=3)
    run_parser.add_argument("--full-volume-shape", nargs=3)
    run_parser.add_argument("--json", nargs="?", const="-")
    run_parser.add_argument("--csv", nargs="?", const="-")
    run_parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.command == "list":
        methods = []
        for name in method_names():
            spec = get_method_spec(name)
            methods.append({
                "method": name,
                "source_entrypoint": spec.source_entrypoint,
                "model_target": spec.model_target,
                "forward_shape": spec.forward_shape,
                "full_volume_shape": spec.full_volume_shape,
                "full_volume_strategy": spec.full_volume_strategy.kind,
            })
        if args.json:
            _write_json(args.json, methods)
        else:
            for method in methods:
                console.print(
                    f"{method['method']}: {method['model_target']}  "
                    f"forward={method['forward_shape']} full={method['full_volume_shape']}"
                )
        return

    shape = _parse_shape(args.shape)
    full_volume_shape = _parse_shape(args.full_volume_shape)
    reports: list[dict[str, Any]] = []

    if args.all:
        reports = _run_all_methods(
            batch_size=args.batch_size,
            measure=args.measure,
            shape=shape,
            full_volume_shape=full_volume_shape,
            verbose=args.verbose,
        )
    else:
        report = run_flops_for_method(
            args.method,
            batch_size=args.batch_size,
            measure=args.measure,
            shape=shape,
            full_volume_shape=full_volume_shape,
        )
        reports.append(report_to_dict(report))

    if args.json:
        payload: Any = reports if args.all else reports[0]
        _write_json(args.json, payload)
        return
    if args.csv:
        _write_csv(args.csv, reports)
        return

    for payload in reports:
        from .core import FlopsMeasurement, FlopsReport

        measurements = tuple(FlopsMeasurement(**measurement) for measurement in payload["measurements"])
        display_report = FlopsReport(
            method=payload["method"],
            source_entrypoint=payload["source_entrypoint"],
            model_target=payload["model_target"],
            measurements=measurements,
        )
        console.print(build_flops_table(display_report))


if __name__ == "__main__":
    main()
