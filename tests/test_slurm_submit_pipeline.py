from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker" / "slurm"))

from submit_pipeline import (  # noqa: E402
    PipelineResult,
    _job_states,
    _parse_job_id,
    submit_preprocess_job,
    submit_test_job,
    wait_for_completion,
)


class FakeSSHRunner:
    """Records every command it's asked to run and returns scripted responses,
    standing in for a real SSHRunner talking to a login node."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self.host = "fake-host"
        self.commands: list[str] = []
        self.responses = responses or {}
        self.sacct_states: list[dict[str, str]] = []
        self._sacct_call = 0

    def run(self, command: str, *, timeout: float = 60.0) -> str:
        self.commands.append(command)
        if command.startswith("sacct"):
            states = self.sacct_states[min(self._sacct_call, len(self.sacct_states) - 1)]
            self._sacct_call += 1
            return "\n".join(f"{job_id}|{state}" for job_id, state in states.items())
        for prefix, response in self.responses.items():
            if command.startswith(prefix):
                return response
        raise AssertionError(f"unscripted command: {command}")


def test_parse_job_id_handles_plain_and_cluster_qualified_output() -> None:
    assert _parse_job_id("12345\n") == "12345"
    assert _parse_job_id("12345;aimagelab\n") == "12345"


def test_submit_preprocess_job_renders_script_and_returns_job_id(tmp_path: Path) -> None:
    runner = FakeSSHRunner(responses={"sbatch --parsable": "111\n"})

    job_id = submit_preprocess_job(
        runner,
        run_id="run1",
        package_dir=tmp_path / "pkg",
        package_name="model.mimosepkg",
        raw_data_dir=tmp_path / "raw",
        shared_dir=tmp_path / "shared",
        log_dir=tmp_path / "logs",
        script_dir=tmp_path / "scripts",
        preprocess_image="/work/grana_neuro/mimose-preprocess.sqsh",
        account="grana_neuro",
        partition="all_usr_prod",
    )

    assert job_id == "111"
    script_path = tmp_path / "scripts" / "preprocess-run1.sbatch.sh"
    assert script_path.is_file()
    script = script_path.read_text()
    assert "#SBATCH --partition=all_usr_prod" in script
    assert "#SBATCH --gres=gpu:0" in script
    assert "#SBATCH --account=grana_neuro" in script
    assert "--container-image=/work/grana_neuro/mimose-preprocess.sqsh" in script
    assert "harness.preprocess_run" in script
    assert "--package /submission/model.mimosepkg" in script
    assert len(runner.commands) == 1
    assert str(script_path) in runner.commands[0]


def test_submit_test_job_chains_dependency_on_preprocess_job(tmp_path: Path) -> None:
    runner = FakeSSHRunner(responses={"sbatch --parsable": "222\n"})

    job_id = submit_test_job(
        runner,
        run_id="run1",
        preprocess_job_id="111",
        package_dir=tmp_path / "pkg",
        package_name="model.mimosepkg",
        shared_dir=tmp_path / "shared",
        results_dir=tmp_path / "results",
        log_dir=tmp_path / "logs",
        script_dir=tmp_path / "scripts",
        test_image="/work/grana_neuro/mimose-test.sqsh",
        dataset_type="brats18",
        account="grana_neuro",
        partition="all_usr_prod",
        gpu_count=1,
    )

    assert job_id == "222"
    script = (tmp_path / "scripts" / "test-run1.sbatch.sh").read_text()
    assert "#SBATCH --dependency=afterok:111" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "--dataset-type brats18" in script
    assert "--data-dir /input/preprocessed" in script
    assert "--split-file /input/split.json" in script


def test_job_states_parses_sacct_output() -> None:
    runner = FakeSSHRunner()
    runner.sacct_states = [{"111": "COMPLETED", "222": "RUNNING"}]

    states = _job_states(runner, ["111", "222"])

    assert states == {"111": "COMPLETED", "222": "RUNNING"}


def test_wait_for_completion_polls_until_test_job_terminal() -> None:
    runner = FakeSSHRunner()
    runner.sacct_states = [
        {"111": "RUNNING", "222": "PENDING"},
        {"111": "COMPLETED", "222": "RUNNING"},
        {"111": "COMPLETED", "222": "COMPLETED"},
    ]
    polls: list[tuple[str, str]] = []

    result = wait_for_completion(
        runner, "111", "222", poll_interval=0, on_poll=lambda p, t: polls.append((p, t))
    )

    assert isinstance(result, PipelineResult)
    assert result.preprocess_state == "COMPLETED"
    assert result.test_state == "COMPLETED"
    assert result.succeeded is True
    assert polls == [
        ("RUNNING", "PENDING"),
        ("COMPLETED", "RUNNING"),
        ("COMPLETED", "COMPLETED"),
    ]


def test_wait_for_completion_stops_early_when_preprocess_fails() -> None:
    """Don't wait out SLURM's own dependency-cancellation cycle once the
    preprocessing job has already failed -- the test job can never run."""
    runner = FakeSSHRunner()
    runner.sacct_states = [
        {"111": "RUNNING", "222": "PENDING"},
        {"111": "FAILED", "222": "PENDING"},
    ]

    result = wait_for_completion(runner, "111", "222", poll_interval=0)

    assert result.preprocess_state == "FAILED"
    assert result.test_state == "PENDING"
    assert result.succeeded is False
    assert len(runner.commands) == 2


def test_wait_for_completion_reports_test_job_failure() -> None:
    runner = FakeSSHRunner()
    runner.sacct_states = [{"111": "COMPLETED", "222": "OUT_OF_MEMORY"}]

    result = wait_for_completion(runner, "111", "222", poll_interval=0)

    assert result.succeeded is False
    assert result.test_state == "OUT_OF_MEMORY"


def test_wait_for_completion_respects_max_wait(monkeypatch) -> None:
    runner = FakeSSHRunner()
    runner.sacct_states = [{"111": "RUNNING", "222": "PENDING"}]

    times = iter([0.0, 0.0, 100.0])
    monkeypatch.setattr("submit_pipeline.time.monotonic", lambda: next(times))
    sleeps: list[float] = []
    monkeypatch.setattr("submit_pipeline.time.sleep", lambda s: sleeps.append(s))

    result = wait_for_completion(runner, "111", "222", poll_interval=5, max_wait=10)

    assert result.preprocess_state == "RUNNING"
    assert result.test_state == "PENDING"
    assert result.succeeded is False
