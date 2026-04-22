from __future__ import annotations

from pathlib import Path

from brainchmark.updater import UpdateError, UpdateResult, update_checkout


def test_update_checkout_refuses_dirty_worktree(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr("brainchmark.updater._repo_root", lambda: repo_root)
    monkeypatch.setattr("brainchmark.updater._is_git_repository", lambda _repo_root: True)
    monkeypatch.setattr("brainchmark.updater._worktree_dirty", lambda _repo_root: True)

    try:
        update_checkout()
    except UpdateError as exc:
        assert "Working tree is not clean" in str(exc)
    else:
        raise AssertionError("Expected UpdateError for dirty worktree")


def test_update_checkout_switches_branch_and_runs_install(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    commands: list[list[str]] = []

    monkeypatch.setattr("brainchmark.updater._repo_root", lambda: repo_root)
    monkeypatch.setattr("brainchmark.updater._is_git_repository", lambda _repo_root: True)
    monkeypatch.setattr("brainchmark.updater._worktree_dirty", lambda _repo_root: False)
    monkeypatch.setattr(
        "brainchmark.updater._current_branch",
        lambda _repo_root: "main",
    )
    monkeypatch.setattr(
        "brainchmark.updater._local_branch_exists",
        lambda _repo_root, branch: branch == "ocarpentiero",
    )

    commit_map = {"HEAD": ["oldcommit", "newcommit"]}

    def fake_commit_for_ref(_repo_root: Path, ref: str) -> str:
        values = commit_map[ref]
        return values.pop(0)

    def fake_run_command(
        args: list[str],
        *,
        cwd: Path,
        capture_output: bool = True,
    ) -> object:
        commands.append(args)
        return object()

    monkeypatch.setattr("brainchmark.updater._commit_for_ref", fake_commit_for_ref)
    monkeypatch.setattr("brainchmark.updater._run_command", fake_run_command)

    result = update_checkout(branch="ocarpentiero")

    assert result == UpdateResult(
        repo_root=repo_root,
        remote="origin",
        branch="ocarpentiero",
        previous_commit="oldcommit",
        current_commit="newcommit",
        changed=True,
        switched_branch=True,
        install_ran=True,
    )
    assert commands == [
        ["git", "fetch", "origin", "ocarpentiero"],
        ["git", "switch", "ocarpentiero"],
        ["git", "pull", "--ff-only", "origin", "ocarpentiero"],
        ["bash", "install.sh"],
    ]


def test_update_checkout_no_change_skips_install(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    commands: list[list[str]] = []

    monkeypatch.setattr("brainchmark.updater._repo_root", lambda: repo_root)
    monkeypatch.setattr("brainchmark.updater._is_git_repository", lambda _repo_root: True)
    monkeypatch.setattr("brainchmark.updater._worktree_dirty", lambda _repo_root: False)
    monkeypatch.setattr(
        "brainchmark.updater._current_branch",
        lambda _repo_root: "ocarpentiero",
    )
    monkeypatch.setattr(
        "brainchmark.updater._commit_for_ref",
        lambda _repo_root, _ref: "samecommit",
    )

    def fake_run_command(
        args: list[str],
        *,
        cwd: Path,
        capture_output: bool = True,
    ) -> object:
        commands.append(args)
        return object()

    monkeypatch.setattr("brainchmark.updater._run_command", fake_run_command)

    result = update_checkout(run_install=True)

    assert result.changed is False
    assert result.install_ran is False
    assert commands == [
        ["git", "fetch", "origin", "ocarpentiero"],
        ["git", "pull", "--ff-only", "origin", "ocarpentiero"],
    ]
