"""Build/run the `docker/preprocess` and `docker/test` images that make up the
inference harness (see `.claude/docker-inference-pipeline-guide.md`).

Uses the `docker` CLI via `subprocess` rather than the Docker SDK, since the
two Dockerfiles already document their exact `docker build`/`docker run`
invocations as the source of truth -- no extra dependency needed to mirror
them here.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

from runner.config import REPO_ROOT

PREPROCESS_DOCKERFILE = REPO_ROOT / "docker" / "preprocess" / "Dockerfile"
TEST_DOCKERFILE = REPO_ROOT / "docker" / "test" / "Dockerfile"

# Matches harness/run.py's/preprocess_run.py's own --extract-dir default.
EXTRACT_DIR_IN_CONTAINER = "/tmp/mimosepkg"


class DockerOrchestrationError(RuntimeError):
    """Raised when a `docker build`/`docker run` invocation fails."""


def _preprocess_image_tag(mimose_version: str) -> str:
    return f"mimose-preprocess:{mimose_version}"


def _test_image_tag(mimose_version: str) -> str:
    return f"mimose-test:{mimose_version}"


def _image_exists(tag: str) -> bool:
    result = subprocess.run(["docker", "images", "-q", tag], capture_output=True, text=True, check=True)
    return bool(result.stdout.strip())


def _docker_build(dockerfile: Path, tag: str) -> None:
    result = subprocess.run(
        ["docker", "build", "-f", str(dockerfile), "-t", tag, str(REPO_ROOT)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise DockerOrchestrationError(f"docker build -t {tag} failed:\n{result.stdout}\n{result.stderr}")


def ensure_images_built(mimose_version: str) -> tuple[str, str]:
    """Build the preprocess/test images for `mimose_version` if they don't
    already exist locally. Images are tagged per version and reused across
    every upload at that version -- never rebuilt per upload.
    """
    preprocess_tag = _preprocess_image_tag(mimose_version)
    test_tag = _test_image_tag(mimose_version)
    if not _image_exists(preprocess_tag):
        _docker_build(PREPROCESS_DOCKERFILE, preprocess_tag)
    if not _image_exists(test_tag):
        _docker_build(TEST_DOCKERFILE, test_tag)
    return preprocess_tag, test_tag


def read_manifest(package_path: Path) -> dict:
    with zipfile.ZipFile(package_path) as zf:
        return json.loads(zf.read("manifest.json"))


def new_job_dir() -> Path:
    """Per-upload work directory mounted into both containers -- mirrors the
    harness's own `/tmp/mimosepkg` extract-dir convention, so concurrent
    uploads don't collide.
    """
    return Path(tempfile.mkdtemp(prefix="mimose_job_"))


def run_preprocess_container(
    *,
    package_path: Path,
    job_dir: Path,
    raw_brats_dir: Path,
    image_tag: str,
    num_workers: int = 8,
) -> None:
    """Writes `job_dir/preprocessed/*.npz` and `job_dir/split.json`."""
    result = subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{package_path}:/workspace/upload.mimosepkg:ro",
            "-v", f"{raw_brats_dir}:/input:ro",
            "-v", f"{job_dir}:/workspace",
            image_tag,
            "--package", "/workspace/upload.mimosepkg",
            "--raw-data-dir", "/input",
            "--output-dir", "/workspace",
            "--extract-dir", EXTRACT_DIR_IN_CONTAINER,
            "--num-workers", str(num_workers),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise DockerOrchestrationError(f"preprocess container failed:\n{result.stdout}\n{result.stderr}")


def run_test_container(
    *,
    package_path: Path,
    job_dir: Path,
    dataset_type: str,
    image_tag: str,
    gpu_device: str = "0",
    num_workers: int = 8,
) -> Path:
    """Writes the test report into `job_dir/output/` and returns that directory."""
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "docker", "run", "--rm",
            "--gpus", f"device={gpu_device}",
            "-v", f"{package_path}:/workspace/upload.mimosepkg:ro",
            "-v", f"{job_dir}:/workspace",
            image_tag,
            "--package", "/workspace/upload.mimosepkg",
            "--data-dir", "/workspace/preprocessed",
            "--output-dir", "/workspace/output",
            "--dataset-type", dataset_type,
            "--split-file", "/workspace/split.json",
            "--extract-dir", EXTRACT_DIR_IN_CONTAINER,
            "--num-workers", str(num_workers),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise DockerOrchestrationError(f"test container failed:\n{result.stdout}\n{result.stderr}")
    return output_dir
