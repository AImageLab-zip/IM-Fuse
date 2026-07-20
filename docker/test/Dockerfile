# GPU testing image: runs mask-sweep test inference for a `mimose export`-produced
# `.mimosepkg`, given an already-preprocessed --data-dir (see docker/preprocess/
# for the CPU-only companion image that produces it). Build once per
# mimose_version, reuse for every package produced at that version -- do not
# rebuild per package. See .claude/docker-inference-pipeline-guide.md.
#
# Build from the repo root so both `pyproject.toml`/`uv.lock` and `docker/harness/`
# are in the build context:
#   docker build -f docker/test/Dockerfile -t mimose-test:<mimose_version> .
#
# Pin this to an exact digest before using in production
# (`docker pull nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 && docker inspect --format='{{index .RepoDigests 0}}' ...`),
# matching the CUDA build torch==2.9.1 expects.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 10001 evaluator
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Reuse the exact resolved dependency set mimose export/train/test ran with,
# instead of hand-rolling a pip install list that could silently drift.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --python 3.12

# The harness never imports the outer `mimose` package -- it only ever
# imports the `mimose.pyz` bundled inside the specific `.mimosepkg` it's
# asked to load (see harness/load.py). Only third-party dependencies need to
# come from this image.
COPY docker/harness/ /app/harness/

RUN mkdir -p /tmp/mimosepkg /output && chown -R evaluator:evaluator /app /tmp/mimosepkg /output
USER evaluator

ENV PATH="/app/.venv/bin:${PATH}"
ENTRYPOINT ["python", "-m", "harness.run"]
