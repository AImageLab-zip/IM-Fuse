from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


class UpdateError(RuntimeError):
    """Raised when the local BrainchMark checkout cannot be updated safely."""


@dataclass(frozen=True)
class UpdateResult:
    repo_root: Path
    remote: str
    branch: str
    previous_commit: str
    current_commit: str
    changed: bool
    switched_branch: bool
    install_ran: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_command(
    args: list[str],
    *,
    cwd: Path,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            check=True,
            text=True,
            capture_output=capture_output,
        )
    except FileNotFoundError as exc:
        raise UpdateError(f"Command not found: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or "unknown command failure"
        raise UpdateError(message) from exc


def _git_output(repo_root: Path, *args: str) -> str:
    completed = _run_command(["git", *args], cwd=repo_root)
    return completed.stdout.strip()


def _is_git_repository(repo_root: Path) -> bool:
    try:
        result = _git_output(repo_root, "rev-parse", "--is-inside-work-tree")
    except UpdateError:
        return False
    return result == "true"


def _worktree_dirty(repo_root: Path) -> bool:
    return bool(_git_output(repo_root, "status", "--porcelain"))


def _current_branch(repo_root: Path) -> str:
    branch = _git_output(repo_root, "branch", "--show-current")
    if not branch:
        raise UpdateError("Detached HEAD is not supported for brainchmark update.")
    return branch


def _local_branch_exists(repo_root: Path, branch: str) -> bool:
    try:
        _git_output(repo_root, "show-ref", "--verify", f"refs/heads/{branch}")
    except UpdateError:
        return False
    return True


def _switch_branch(repo_root: Path, *, remote: str, branch: str) -> None:
    if _local_branch_exists(repo_root, branch):
        _run_command(["git", "switch", branch], cwd=repo_root, capture_output=False)
        return
    _run_command(
        ["git", "switch", "--track", "-c", branch, f"{remote}/{branch}"],
        cwd=repo_root,
        capture_output=False,
    )


def _commit_for_ref(repo_root: Path, ref: str) -> str:
    return _git_output(repo_root, "rev-parse", ref)


def update_checkout(
    *,
    branch: str | None = None,
    remote: str = "origin",
    run_install: bool = True,
) -> UpdateResult:
    repo_root = _repo_root()
    if not _is_git_repository(repo_root):
        raise UpdateError(f"{repo_root} is not a Git repository.")
    if _worktree_dirty(repo_root):
        raise UpdateError(
            "Working tree is not clean. Commit, stash, or discard changes before updating."
        )

    current_branch = _current_branch(repo_root)
    target_branch = branch or current_branch

    _run_command(["git", "fetch", remote, target_branch], cwd=repo_root, capture_output=False)
    switched_branch = current_branch != target_branch
    if switched_branch:
        _switch_branch(repo_root, remote=remote, branch=target_branch)

    previous_commit = _commit_for_ref(repo_root, "HEAD")
    _run_command(
        ["git", "pull", "--ff-only", remote, target_branch],
        cwd=repo_root,
        capture_output=False,
    )
    current_commit = _commit_for_ref(repo_root, "HEAD")

    install_ran = False
    changed = previous_commit != current_commit
    if changed and run_install:
        _run_command(["bash", "install.sh"], cwd=repo_root, capture_output=False)
        install_ran = True

    return UpdateResult(
        repo_root=repo_root,
        remote=remote,
        branch=target_branch,
        previous_commit=previous_commit,
        current_commit=current_commit,
        changed=changed,
        switched_branch=switched_branch,
        install_ran=install_ran,
    )
