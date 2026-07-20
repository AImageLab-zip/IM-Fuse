# CPU-only preprocessing image: reads a `.mimosepkg`'s embedded preprocessing
# config and turns raw case directories into the `.npz` + split.json the
# docker/test/ GPU image's --data-dir/--split-file expect. Never imports the
# model class (mimose.models.*), so it never needs torch/mamba_ssm/CUDA --
# it's the harness/extract_package() code path, not resolve_package(). See
# docker/harness/preprocess_run.py and .claude/docker-inference-pipeline-guide.md.
#
# Build from the repo root so both `pyproject.toml`/`uv.lock` and
# `docker/harness/` are in the build context:
#   docker build -f docker/preprocess/Dockerfile -t mimose-preprocess:<mimose_version> .
#
# Reuses the same pyproject.toml/uv.lock as the GPU image for a guaranteed-
# consistent dependency set (including torch, which importlib still has to
# resolve as a name even though it's never exercised on this path); trim to a
# preprocessing-only dependency subset later if image size becomes a concern.
FROM python:3.12.13-slim

RUN useradd -m -u 10001 evaluator
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --python 3.12

COPY docker/harness/ /app/harness/

RUN mkdir -p /tmp/mimosepkg /shared && chown -R evaluator:evaluator /app /tmp/mimosepkg /shared
USER evaluator

ENV PATH="/app/.venv/bin:${PATH}"
ENTRYPOINT ["python", "-m", "harness.preprocess_run"]
