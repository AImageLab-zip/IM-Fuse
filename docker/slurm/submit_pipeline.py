"""Submit the two-container `.mimosepkg` evaluation pipeline (CPU-only
preprocessing, then GPU testing) to a remote SLURM login node over SSH, chain
them with a SLURM job dependency so ordering is guaranteed even if this
process dies between submissions, and poll until the pipeline finishes.

This assumes (typical of this kind of cluster, and true of this one):
- the account this runs as already has SSH key-based access to `--host`
  (no password/host-key prompts -- see ssh_runner.SSHRunner)
- `--host` and wherever this script writes the rendered sbatch scripts share
  a filesystem (e.g. both mount /work), so the remote `sbatch` invocation can
  read the same script path this process wrote locally
- pyxis/enroot are configured on `--host`'s cluster (`srun
  --container-image=...`), and `--preprocess-image`/`--test-image` are
  already built and reachable there (a `docker://...` reference, or a local
  `.sqsh` path -- this script does not build or publish images itself)

Usage (run directly as a script -- not `python -m docker.slurm...`, since this
package intentionally isn't nested under `mimose` and uses plain sibling
imports resolved via the script's own directory on sys.path):

    python docker/slurm/submit_pipeline.py \\
        --host ailb-login-01 \\
        --package /work/phd_mimose/runs/ims2trans18_fold1/export/ims2trans18_fold1.mimosepkg \\
        --raw-data-dir /work/phd_mimose/internal_raw/internal \\
        --run-dir /work/phd_mimose/runs/ims2trans18_fold1/pipeline_runs/internal \\
        --preprocess-image /work/grana_neuro/mimose-preprocess.sqsh \\
        --test-image /work/grana_neuro/mimose-test.sqsh \\
        --dataset-type brats18 \\
        --account grana_neuro \\
        --partition all_usr_prod
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

from ssh_runner import SSHRunner
from templates import PREPROCESS_SBATCH_TEMPLATE, TEST_SBATCH_TEMPLATE

TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "DEADLINE",
    "BOOT_FAIL",
    "PREEMPTED",
}
SUCCESS_STATES = {"COMPLETED"}


@dataclasses.dataclass
class PipelineResult:
    preprocess_job_id: str
    test_job_id: str
    preprocess_state: str
    test_state: str

    @property
    def succeeded(self) -> bool:
        return self.preprocess_state in SUCCESS_STATES and self.test_state in SUCCESS_STATES


def _parse_job_id(sbatch_output: str) -> str:
    # `sbatch --parsable` prints "<jobid>" or "<jobid>;<cluster>" (multi-cluster setups).
    first_line = sbatch_output.strip().splitlines()[0]
    return first_line.split(";")[0].strip()


def submit_preprocess_job(
    runner: SSHRunner,
    *,
    run_id: str,
    package_dir: Path,
    package_name: str,
    raw_data_dir: Path,
    shared_dir: Path,
    log_dir: Path,
    script_dir: Path,
    preprocess_image: str,
    account: str,
    partition: str,
    cpus: int = 8,
    mem: str = "32G",
    time_limit: str = "02:00:00",
) -> str:
    log_dir.mkdir(parents=True, exist_ok=True)
    script_dir.mkdir(parents=True, exist_ok=True)

    script = PREPROCESS_SBATCH_TEMPLATE.format(
        run_id=run_id,
        partition=partition,
        preprocess_cpus=cpus,
        preprocess_mem=mem,
        preprocess_time=time_limit,
        log_dir=log_dir,
        account=account,
        preprocess_image=preprocess_image,
        package_dir=package_dir,
        raw_data_dir=raw_data_dir,
        shared_dir=shared_dir,
        package_name=package_name,
    )
    script_path = script_dir / f"preprocess-{run_id}.sbatch.sh"
    script_path.write_text(script)

    output = runner.run(f"sbatch --parsable {script_path}")
    return _parse_job_id(output)


def submit_test_job(
    runner: SSHRunner,
    *,
    run_id: str,
    preprocess_job_id: str,
    package_dir: Path,
    package_name: str,
    shared_dir: Path,
    results_dir: Path,
    log_dir: Path,
    script_dir: Path,
    test_image: str,
    dataset_type: str,
    account: str,
    partition: str,
    gpu_count: int = 1,
    cpus: int = 8,
    mem: str = "64G",
    time_limit: str = "04:00:00",
) -> str:
    log_dir.mkdir(parents=True, exist_ok=True)
    script_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    script = TEST_SBATCH_TEMPLATE.format(
        run_id=run_id,
        partition=partition,
        test_cpus=cpus,
        test_mem=mem,
        test_time=time_limit,
        log_dir=log_dir,
        account=account,
        gpu_count=gpu_count,
        preprocess_job_id=preprocess_job_id,
        test_image=test_image,
        package_dir=package_dir,
        shared_dir=shared_dir,
        results_dir=results_dir,
        package_name=package_name,
        dataset_type=dataset_type,
    )
    script_path = script_dir / f"test-{run_id}.sbatch.sh"
    script_path.write_text(script)

    output = runner.run(f"sbatch --parsable {script_path}")
    return _parse_job_id(output)


def _job_states(runner: SSHRunner, job_ids: list[str]) -> dict[str, str]:
    ids = ",".join(job_ids)
    output = runner.run(f"sacct -j {ids} --format=JobID,State --noheader --parsable2 -X")
    states: dict[str, str] = {}
    for line in output.strip().splitlines():
        if not line.strip():
            continue
        job_id, state = line.split("|", 1)
        # Strip qualifiers like "CANCELLED by 12345" down to the bare state.
        states[job_id.strip()] = state.strip().split(" ")[0]
    return states


def wait_for_completion(
    runner: SSHRunner,
    preprocess_job_id: str,
    test_job_id: str,
    *,
    poll_interval: float = 30.0,
    max_wait: float | None = None,
    on_poll=None,
) -> PipelineResult:
    """Poll both jobs until the pipeline reaches a terminal state.

    Returns as soon as the preprocessing job fails (no point waiting out
    SLURM's own dependency-cancellation cycle for the test job), or once the
    test job itself reaches a terminal state.
    """
    started = time.monotonic()
    preprocess_state = "PENDING"
    test_state = "PENDING"

    while True:
        states = _job_states(runner, [preprocess_job_id, test_job_id])
        preprocess_state = states.get(preprocess_job_id, preprocess_state)
        test_state = states.get(test_job_id, test_state)

        if on_poll is not None:
            on_poll(preprocess_state, test_state)

        if preprocess_state in TERMINAL_STATES and preprocess_state not in SUCCESS_STATES:
            # Test job can never run now; don't wait for SLURM to notice.
            break
        if test_state in TERMINAL_STATES:
            break
        if max_wait is not None and (time.monotonic() - started) > max_wait:
            break

        time.sleep(poll_interval)

    return PipelineResult(
        preprocess_job_id=preprocess_job_id,
        test_job_id=test_job_id,
        preprocess_state=preprocess_state,
        test_state=test_state,
    )


def submit_pipeline(
    *,
    host: str,
    package: Path,
    raw_data_dir: Path,
    run_dir: Path,
    preprocess_image: str,
    test_image: str,
    dataset_type: str,
    account: str,
    partition: str = "all_usr_prod",
    gpu_count: int = 1,
    poll_interval: float = 30.0,
    max_wait: float | None = None,
    on_poll=None,
) -> PipelineResult:
    run_id = str(int(time.time()))
    runner = SSHRunner(host)

    shared_dir = run_dir / "shared"
    results_dir = run_dir / "results"
    log_dir = run_dir / "logs"
    script_dir = run_dir / "scripts"

    preprocess_job_id = submit_preprocess_job(
        runner,
        run_id=run_id,
        package_dir=package.parent,
        package_name=package.name,
        raw_data_dir=raw_data_dir,
        shared_dir=shared_dir,
        log_dir=log_dir,
        script_dir=script_dir,
        preprocess_image=preprocess_image,
        account=account,
        partition=partition,
    )
    test_job_id = submit_test_job(
        runner,
        run_id=run_id,
        preprocess_job_id=preprocess_job_id,
        package_dir=package.parent,
        package_name=package.name,
        shared_dir=shared_dir,
        results_dir=results_dir,
        log_dir=log_dir,
        script_dir=script_dir,
        test_image=test_image,
        dataset_type=dataset_type,
        account=account,
        partition=partition,
        gpu_count=gpu_count,
    )

    return wait_for_completion(
        runner,
        preprocess_job_id,
        test_job_id,
        poll_interval=poll_interval,
        max_wait=max_wait,
        on_poll=on_poll,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True, help="SLURM login node to SSH into and run sbatch on.")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--raw-data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="Shared-filesystem directory for this run's shared/results/logs/scripts.")
    parser.add_argument("--preprocess-image", required=True, help="docker://... ref or .sqsh path, already reachable from --host.")
    parser.add_argument("--test-image", required=True)
    parser.add_argument("--dataset-type", required=True, choices=("brats18", "brats23", "brats25"))
    parser.add_argument("--account", default="grana_neuro")
    parser.add_argument("--partition", default="all_usr_prod")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--max-wait", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    def on_poll(preprocess_state: str, test_state: str) -> None:
        print(f"preprocess={preprocess_state} test={test_state}", file=sys.stderr)

    result = submit_pipeline(
        host=args.host,
        package=args.package,
        raw_data_dir=args.raw_data_dir,
        run_dir=args.run_dir,
        preprocess_image=args.preprocess_image,
        test_image=args.test_image,
        dataset_type=args.dataset_type,
        account=args.account,
        partition=args.partition,
        gpu_count=args.gpu_count,
        poll_interval=args.poll_interval,
        max_wait=args.max_wait,
        on_poll=on_poll,
    )

    print(f"preprocess_job_id={result.preprocess_job_id} state={result.preprocess_state}")
    print(f"test_job_id={result.test_job_id} state={result.test_state}")
    print(f"results -> {args.run_dir / 'results'}")
    if not result.succeeded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
