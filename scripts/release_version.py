#!/usr/bin/env python3
"""Bump the BrainchMark package version with a small, repeatable workflow."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
UV_LOCK_PATH = PROJECT_ROOT / "uv.lock"

SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
PROJECT_VERSION_RE = re.compile(
    r"(?ms)^(\[project\]\n.*?^version = )\"([^\"]+)\""
)
UV_LOCK_VERSION_RE = re.compile(
    r'(?ms)^(\[\[package\]\]\nname = "brainchmark"\nversion = )"([^"]+)"'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bump the BrainchMark version. Use major/minor/patch or an explicit "
            "semantic version like 1.4.0."
        )
    )
    parser.add_argument("target", help="major, minor, patch, or an explicit X.Y.Z")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the computed version without changing files.",
    )
    return parser.parse_args()


def read_project_version(pyproject_text: str) -> str:
    match = PROJECT_VERSION_RE.search(pyproject_text)
    if match is None:
        raise SystemExit("Could not find [project].version in pyproject.toml.")
    return match.group(2)


def bump_version(current: str, target: str) -> str:
    if SEMVER_RE.fullmatch(target):
        return target

    match = SEMVER_RE.fullmatch(current)
    if match is None:
        raise SystemExit(f"Current version is not semantic: {current}")

    major, minor, patch = (int(part) for part in match.groups())
    if target == "major":
        return f"{major + 1}.0.0"
    if target == "minor":
        return f"{major}.{minor + 1}.0"
    if target == "patch":
        return f"{major}.{minor}.{patch + 1}"

    raise SystemExit(
        f"Invalid target '{target}'. Use major, minor, patch, or an explicit X.Y.Z."
    )


def replace_once(pattern: re.Pattern[str], text: str, new_version: str) -> str:
    match = pattern.search(text)
    if match is None:
        return text
    return pattern.sub(rf'\1"{new_version}"', text, count=1)


def main() -> int:
    args = parse_args()
    pyproject_text = PYPROJECT_PATH.read_text(encoding="utf-8")
    current_version = read_project_version(pyproject_text)
    new_version = bump_version(current_version, args.target)

    if current_version == new_version:
        print(f"Version unchanged: {current_version}")
        return 0

    print(f"BrainchMark version: {current_version} -> {new_version}")
    print(f"Suggested git tag: v{new_version}")

    if args.dry_run:
        return 0

    updated_pyproject = replace_once(PROJECT_VERSION_RE, pyproject_text, new_version)
    PYPROJECT_PATH.write_text(updated_pyproject, encoding="utf-8")
    print(f"Updated {PYPROJECT_PATH.relative_to(PROJECT_ROOT)}")

    if UV_LOCK_PATH.exists():
        uv_lock_text = UV_LOCK_PATH.read_text(encoding="utf-8")
        updated_uv_lock = replace_once(UV_LOCK_VERSION_RE, uv_lock_text, new_version)
        if updated_uv_lock != uv_lock_text:
            UV_LOCK_PATH.write_text(updated_uv_lock, encoding="utf-8")
            print(f"Updated {UV_LOCK_PATH.relative_to(PROJECT_ROOT)}")
        else:
            print("Skipped uv.lock version update; package entry not found.")

    print("Next steps:")
    print("  1. Run: uv lock")
    print("  2. Commit the version bump")
    print(f"  3. Create a tag: git tag v{new_version}")
    print("  4. Push branch and tags")
    return 0


if __name__ == "__main__":
    sys.exit(main())
