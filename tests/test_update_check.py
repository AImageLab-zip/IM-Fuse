from __future__ import annotations

import json

from brainchmark import update_check


def test_extract_version_from_pyproject_reads_project_version() -> None:
    version = update_check._extract_version_from_pyproject(
        """
        [project]
        name = "MiMoSe"
        version = "0.4.0"
        """
    )

    assert version == "0.4.0"


def test_maybe_notify_about_update_skips_when_last_check_is_recent(
    tmp_path,
) -> None:
    state_path = tmp_path / "update_check.json"
    state_path.write_text(
        json.dumps({"last_checked_at": 1_000.0}),
        encoding="utf-8",
    )
    calls = 0

    def fetch_remote_version() -> str | None:
        nonlocal calls
        calls += 1
        return "0.0.3"

    update_check.maybe_notify_about_update(
        now=1_000.0 + update_check.UPDATE_CHECK_INTERVAL_SECONDS - 1,
        state_path=state_path,
        fetch_remote_version=fetch_remote_version,
        echo=lambda message: None,
    )

    assert calls == 0


def test_maybe_notify_about_update_records_check_and_emits_notice(
    tmp_path,
    monkeypatch,
) -> None:
    state_path = tmp_path / "update_check.json"
    messages: list[str] = []
    monkeypatch.setattr(update_check, "__version__", "0.0.2")

    update_check.maybe_notify_about_update(
        now=5_000.0,
        state_path=state_path,
        fetch_remote_version=lambda: "0.0.3",
        echo=messages.append,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["last_checked_at"] == 5_000.0
    assert state["latest_version"] == "0.0.3"
    assert messages == [
        "Update available: MiMoSe 0.0.3 is available on the stable branch "
        "(installed: 0.0.2)."
    ]


def test_maybe_notify_about_update_records_attempt_when_fetch_fails(
    tmp_path,
    monkeypatch,
) -> None:
    state_path = tmp_path / "update_check.json"
    messages: list[str] = []
    monkeypatch.setattr(update_check, "__version__", "0.0.2")

    update_check.maybe_notify_about_update(
        now=9_000.0,
        state_path=state_path,
        fetch_remote_version=lambda: None,
        echo=messages.append,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state == {"last_checked_at": 9_000.0}
    assert messages == []
