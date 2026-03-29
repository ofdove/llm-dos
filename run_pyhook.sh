#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Install pyhook agent into the active Python environment
cp "$REPO_ROOT"/pyhook/pyhook_agent.py "$REPO_ROOT"/pyhook/sitecustomize.py \
   "$(python -c "import site; print(site.getsitepackages()[0])")"

# Start the collector (requires root for BPF map access)
sudo PYHOOK_LIBBPF="$REPO_ROOT/ebpf/cuda-tracing/libbpf.so" \
    python "$REPO_ROOT/pyhook/pyhook_collector.py"
