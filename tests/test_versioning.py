from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError

from brainchmark.versioning import (
    UpdateCheckResult,
    _normalize_version,
    check_for_updates,
)


def test_normalize_version_accepts_semver_tags() -> None:
    assert _normalize_version("v1.2.3") == "1.2.3"
    assert _normalize_version("refs/tags/v0.4.0") == "0.4.0"


def test_normalize_version_rejects_prerelease_and_invalid_values() -> None:
    assert _normalize_version("v1.2.3rc1") is None
    assert _normalize_version("main") is None


def test_check_for_updates_uses_cached_result(monkeypatch, tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_root))

    cached_result = UpdateCheckResult(
        current_version="0.1.0",
        latest_version="0.2.0",
        update_available=True,
        source="release",
        error=None,
    )
    cache_path = cache_root / "brainchmark" / "version_check.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "current_version": "0.1.0",
                "latest_version": "0.2.0",
                "update_available": True,
                "source": "release",
                "error": None,
                "expires_at": 9999999999,
            }
        ),
        encoding="utf-8",
    )

    def fail_fetch(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network should not be called when cache is valid")

    monkeypatch.setattr("brainchmark.versioning._fetch_latest_release_version", fail_fetch)
    result = check_for_updates()

    assert result == cached_result


def test_check_for_updates_handles_network_failure(
    monkeypatch, tmp_path: Path
) -> None:
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_root))

    def raise_network_error(*_args: object, **_kwargs: object) -> None:
        raise URLError("offline")

    monkeypatch.setattr(
        "brainchmark.versioning._fetch_latest_release_version",
        raise_network_error,
    )

    result = check_for_updates(force_refresh=True)

    assert result.latest_version is None
    assert result.update_available is False
    assert result.error is not None
