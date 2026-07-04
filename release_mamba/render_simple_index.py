#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from pathlib import Path


def normalize_project_name(filename: str) -> str:
    base = filename.split("-", 1)[0]
    return base.replace("_", "-").lower()


def render_links(repo: str, tag: str, wheel_names: list[str]) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for wheel_name in wheel_names:
        grouped.setdefault(normalize_project_name(wheel_name), []).append(wheel_name)

    pages: dict[str, str] = {}
    for project_name, project_wheels in grouped.items():
        links = []
        for wheel_name in sorted(project_wheels):
            url = (
                f"https://github.com/{repo}/releases/download/"
                f"{tag}/{wheel_name}"
            )
            links.append(
                f'<a href="{html.escape(url)}">{html.escape(wheel_name)}</a>'
            )
        body = "<html><body>\n" + "<br/>\n".join(links) + "\n</body></html>\n"
        pages[project_name] = body
    return pages


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a minimal PEP 503-style index for GitHub release wheels."
    )
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--tag", required=True, help="release tag")
    parser.add_argument(
        "--wheelhouse",
        default="wheelhouse",
        help="directory containing the built wheels",
    )
    parser.add_argument(
        "--output",
        default="simple",
        help="directory to write the simple-index structure into",
    )
    args = parser.parse_args()

    wheelhouse = Path(args.wheelhouse)
    output_dir = Path(args.output)
    wheel_names = sorted(path.name for path in wheelhouse.glob("*.whl"))
    if not wheel_names:
        raise SystemExit(f"no wheels found in {wheelhouse}")

    pages = render_links(args.repo, args.tag, wheel_names)
    for project_name, content in pages.items():
        project_dir = output_dir / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "index.html").write_text(content, encoding="utf-8")

    root_links = [
        f'<a href="{html.escape(name)}/">{html.escape(name)}</a>'
        for name in sorted(pages)
    ]
    root_content = "<html><body>\n" + "<br/>\n".join(root_links) + "\n</body></html>\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(root_content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
