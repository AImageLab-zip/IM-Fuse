#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PARENT_DIR"

uv sync
ip install causal-conv1d>=1.4.0 --no-build-isolation
pip install mamba-ssm --no-build-isolation