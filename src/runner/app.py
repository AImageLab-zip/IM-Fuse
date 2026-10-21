"""FastAPI website: upload a `.mimosepkg`, run it through the Docker inference
harness (`docker/preprocess` + `docker/test`), and download the report.

Run with `mimose-runner` (console script) or `uvicorn runner.app:app`.
"""
from __future__ import annotations

import threading
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

from runner.config import ConfigError, get_port, get_raw_brats_dir
from runner.docker_orchestration import (
    DockerOrchestrationError,
    ensure_images_built,
    new_job_dir,
    read_manifest,
    run_preprocess_container,
    run_test_container,
)

app = FastAPI(title="MiMoSe runner")

# One job at a time: the test container needs the GPU, and this isn't meant
# to be a multi-tenant queue yet -- a deliberate scoping choice, easy to
# revisit later if concurrent uploads become a real need.
_JOB_LOCK = threading.Lock()
_JOBS: dict[str, Path] = {}

_UPLOAD_FORM = """<!doctype html>
<html>
<head><title>MiMoSe runner</title></head>
<body>
<h1>MiMoSe: run a .mimosepkg</h1>
<form action="/upload" method="post" enctype="multipart/form-data">
  <input type="file" name="package" accept=".mimosepkg" required>
  <button type="submit">Run</button>
</form>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _UPLOAD_FORM


@app.post("/upload", response_class=HTMLResponse)
async def upload(package: UploadFile) -> str:
    job_dir = new_job_dir()
    package_path = job_dir / "upload.mimosepkg"
    contents = await package.read()
    if not contents:
        raise HTTPException(400, f"{package.filename} was uploaded as an empty file (0 bytes).")
    package_path.write_bytes(contents)

    try:
        manifest = read_manifest(package_path)
    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise HTTPException(400, f"{package.filename} is not a valid .mimosepkg: {exc}") from exc

    preprocessing_config = manifest.get("preprocessing")
    if preprocessing_config is None:
        raise HTTPException(
            400,
            f"{package.filename} has no embedded preprocessing config -- it was "
            "exported before this was tracked, or from a minimal export-only "
            "YAML. Re-export with the full training config.",
        )

    mimose_version = manifest["mimose_version"]
    dataset_type = str(preprocessing_config["dataset_type"])
    raw_brats_dir = get_raw_brats_dir()

    with _JOB_LOCK:
        try:
            preprocess_tag, test_tag = ensure_images_built(mimose_version)
            run_preprocess_container(
                package_path=package_path,
                job_dir=job_dir,
                raw_brats_dir=raw_brats_dir,
                image_tag=preprocess_tag,
            )
            output_dir = run_test_container(
                package_path=package_path,
                job_dir=job_dir,
                dataset_type=dataset_type,
                image_tag=test_tag,
            )
        except DockerOrchestrationError as exc:
            raise HTTPException(500, str(exc)) from exc

    job_id = uuid.uuid4().hex
    _JOBS[job_id] = output_dir

    report_files = sorted(p.name for p in output_dir.iterdir())
    links = "".join(f'<li><a href="/jobs/{job_id}/{name}">{name}</a></li>' for name in report_files)
    return f"<html><body><h1>Done: {package.filename}</h1><ul>{links}</ul></body></html>"


@app.get("/jobs/{job_id}/{filename}")
def download(job_id: str, filename: str) -> FileResponse:
    output_dir = _JOBS.get(job_id)
    if output_dir is None:
        raise HTTPException(404, "unknown job")
    file_path = (output_dir / filename).resolve()
    if output_dir.resolve() not in file_path.parents or not file_path.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(file_path)


def main() -> None:
    try:
        get_raw_brats_dir()
    except ConfigError as exc:
        raise SystemExit(str(exc)) from None
    uvicorn.run(app, host="0.0.0.0", port=get_port())


if __name__ == "__main__":
    main()
