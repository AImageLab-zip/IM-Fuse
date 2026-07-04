# Publishing Rebuilt Mamba Wheels For `uv`

This folder covers the `uv`-native release flow for rebuilt
`causal-conv1d` and `mamba-ssm` wheels and publishing them to GitHub Releases.

## 1. Build the wheels

Build on a machine that matches the target environment:

- Linux
- same Python version as the project
- compatible CUDA toolkit
- compatible glibc / CPU architecture

First, download the source archives from PyPI into `release_mamba/sources/`:

```text
release_mamba/sources/causal_conv1d-1.6.2.post1.tar.gz
release_mamba/sources/mamba_ssm-2.3.2.post1.tar.gz
```

Then run:

```bash
cd release_mamba
bash build_stuff.sh
```

The script writes wheels to `release_mamba/wheelhouse/`.

Environment knobs:

```bash
CUDA_MODULE=cuda/12.8 PYTHON_BIN=3.12.13 \
WHEELHOUSE=wheelhouse SOURCES_DIR=sources bash build_stuff.sh
```

## 2. Check what is inside the wheels

Unpack and inspect the built artifacts before publishing them:

```bash
mkdir -p /tmp/wheelcheck
python -m zipfile -e wheelhouse/causal_conv1d-*.whl /tmp/wheelcheck/causal
python -m zipfile -e wheelhouse/mamba_ssm-*.whl /tmp/wheelcheck/mamba

find /tmp/wheelcheck -type f | grep -E '\.so|cuda|cudnn|cublas|nvrtc|nvidia'
find /tmp/wheelcheck -name '*.so' -print
```

For each `.so`, run:

```bash
ldd /tmp/wheelcheck/.../your_extension.so
```

The normal case is that the wheel contains the package extension itself, but
not NVIDIA runtime libraries such as `libcudart.so` or `libcublas.so`.

## 3. Create a GitHub release

Create a tag that encodes the stack:

```bash
git tag wheels-cu128-py312-v1
git push origin wheels-cu128-py312-v1
```

Then create a release and upload the wheel files from `wheelhouse/` as assets.

If you use the GitHub CLI:

```bash
gh release create wheels-cu128-py312-v1 wheelhouse/*.whl \
  --title "Unofficial wheels: cu128 py312" \
  --notes "Rebuilt wheels for causal-conv1d and mamba-ssm for Python 3.12 / CUDA 12.8."
```

## 4. Point `uv` directly at the release assets

If you already know the exact GitHub Release asset URLs for the wheel files,
you do not need to publish a package index or host the generated `simple/`
directory anywhere.

In `pyproject.toml`, set direct URL sources:

```toml
[tool.uv.sources]
causal-conv1d = { url = "https://github.com/AImageLab-zip/MiMoSe/releases/download/wheels-cu128-py312-v1/causal_conv1d-...whl" }
mamba-ssm = { url = "https://github.com/AImageLab-zip/MiMoSe/releases/download/wheels-cu128-py312-v1/mamba_ssm-...whl" }
```

Then refresh and install:

```bash
uv lock
uv sync
```

## 5. Optional: generate a small index instead of using direct URLs

If you prefer to point `uv` at an index instead of hardcoding exact wheel
URLs, generate a tiny static HTML tree that links to the GitHub Release assets:

```bash
python render_simple_index.py \
  --repo AImageLab-zip/MiMoSe \
  --tag wheels-cu128-py312-v1 \
  --wheelhouse wheelhouse \
  --output simple
```

This creates:

```text
simple/
  index.html
  causal-conv1d/
    index.html
  mamba-ssm/
    index.html
```

If that `simple/` directory is published somewhere static over HTTPS, then in
`pyproject.toml` pin only these two packages to the rebuilt wheel index:

```toml
[tool.uv.sources]
causal-conv1d = { index = "mimose-wheels" }
mamba-ssm = { index = "mimose-wheels" }

[[tool.uv.index]]
name = "mimose-wheels"
url = "https://YOUR_USER.github.io/YOUR_REPO/simple"
explicit = true
```

`explicit = true` keeps all unrelated packages on their normal indexes.

## 6. Sync normally

After either configuration is in place:

```bash
uv lock
uv sync
```

If the lockfile was created before the new source configuration existed,
refresh it after adding the `tool.uv.sources` entry and, if applicable, the
`tool.uv.index` entry.
