#!/usr/bin/env bash
set -euo pipefail

# Enable pyhook tracing for vLLM
export PYHOOK_ENABLE=1
export PYHOOK_INCLUDE="vllm.entrypoints.openai.api_server"
export PYHOOK_SOCK=/tmp/pyhook.sock

# Launch vLLM (adjust model name as needed)
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000
