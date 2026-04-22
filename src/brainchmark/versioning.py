from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from packaging.version import InvalidVersion, Version

from brainchmark import __version__

GITHUB_OWNER = "AImageLab-zip"
GITHUB_REPO = "IM-Fuse"
GITHUB_API_BASE = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"
CACHE_TTL_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class UpdateCheckResult:
    current_version: str
    latest_version: str | None
    update_available: bool
    source: str | None
    error: str | None = None


def _cache_path() -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "brainchmark" / "version_check.json"


def _normalize_version(raw_value: str) -> str | None:
    candidate = raw_value.strip()
    if candidate.startswith("refs/tags/"):
        candidate = candidate.removeprefix("refs/tags/")
    if candidate.startswith("v"):
        candidate = candidate[1:]
    try:
        version = Version(candidate)
    except InvalidVersion:
        return None
    if version.is_prerelease:
        return None
    return str(version)


def _parse_json_response(response: Any) -> Any:
    return json.loads(response.read().decode("utf-8"))


def _github_get_json(url: str, timeout: float) -> Any:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"BrainchMark/{__version__}",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return _parse_json_response(response)


def _fetch_latest_release_version(timeout: float) -> tuple[str | None, str | None]:
    try:
        payload = _github_get_json(f"{GITHUB_API_BASE}/releases/latest", timeout)
    except HTTPError as exc:
        if exc.code == 404:
            return None, None
        raise

    latest_version = _normalize_version(str(payload.get("tag_name", "")))
    if latest_version is None:
        return None, None
    return latest_version, "release"


def _fetch_latest_tag_version(timeout: float) -> tuple[str | None, str | None]:
    payload = _github_get_json(f"{GITHUB_API_BASE}/tags?per_page=30", timeout)
    candidates = [
        normalized
        for item in payload
        if (normalized := _normalize_version(str(item.get("name", "")))) is not None
    ]
    if not candidates:
        return None, None
    latest_version = str(max(Version(candidate) for candidate in candidates))
    return latest_version, "tag"


def _read_cached_result() -> UpdateCheckResult | None:
    path = _cache_path()
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expires_at = int(payload["expires_at"])
        if expires_at <= int(time.time()):
            return None
        return UpdateCheckResult(
            current_version=payload["current_version"],
            latest_version=payload.get("latest_version"),
            update_available=bool(payload["update_available"]),
            source=payload.get("source"),
            error=payload.get("error"),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_cached_result(result: UpdateCheckResult) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "current_version": result.current_version,
        "latest_version": result.latest_version,
        "update_available": result.update_available,
        "source": result.source,
        "error": result.error,
        "expires_at": int(time.time()) + CACHE_TTL_SECONDS,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _compare_versions(current_version: str, latest_version: str | None) -> bool:
    if latest_version is None:
        return False
    return Version(latest_version) > Version(current_version)


def check_for_updates(
    *,
    force_refresh: bool = False,
    timeout: float = 3.0,
) -> UpdateCheckResult:
    current_version = __version__
    if not force_refresh:
        cached_result = _read_cached_result()
        if cached_result is not None and cached_result.current_version == current_version:
            return cached_result

    try:
        latest_version, source = _fetch_latest_release_version(timeout)
        if latest_version is None:
            latest_version, source = _fetch_latest_tag_version(timeout)
        result = UpdateCheckResult(
            current_version=current_version,
            latest_version=latest_version,
            update_available=_compare_versions(current_version, latest_version),
            source=source,
        )
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        result = UpdateCheckResult(
            current_version=current_version,
            latest_version=None,
            update_available=False,
            source=None,
            error=str(exc),
        )

    if result.error is None:
        _write_cached_result(result)
    return result
