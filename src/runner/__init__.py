"""Local website that drives the `docker/harness/` inference pipeline: upload a
`.mimosepkg`, preprocess raw BRaTS cases per its manifest, run the mask-sweep
test loop, and download the resulting report. See `src/runner/app.py`."""
