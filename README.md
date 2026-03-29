# LLM-DoS: eBPF-Based Denial-of-Service Detection for LLM Serving Systems

This repository contains the source code for the paper:

> **Detecting Denial-of-Service Attacks on LLM Serving Systems via eBPF-Based GPU and CPU Tracing**

## Overview

LLM-DoS is a kernel-level detection system for denial-of-service (DoS) attacks targeting large language model (LLM) inference services. It uses eBPF to instrument both GPU operations (via uprobes on the CUDA driver API) and CPU scheduling events (via kernel tracepoints), enabling per-request trace reconstruction and real-time classification without modifying the model or serving framework.

### Key Features

- **Dual-layer eBPF tracing**: GPU events (kernel launches, memory transfers, synchronization) + CPU events (scheduling, futex, mmap)
- **Per-request correlation**: A lightweight Python hook (`sys.setprofile`) propagates request IDs to eBPF via BPF maps
- **30-dimensional feature extraction**: 20 GPU features + 10 CPU features per request
- **Real-time classification**: Random Forest, Gradient Boosting, and Logistic Regression classifiers
- **Early detection**: Classify requests within seconds of arrival

## Repository Structure

```
.
├── ebpf/
│   └── cuda-tracing/          # eBPF programs for GPU and CPU tracing
│       ├── cuda_trace.bpf.c   # BPF C program (uprobes + tracepoints)
│       ├── cuda_trace.c        # User-space loader and event consumer
│       ├── cuda_evt.h          # Event type definitions
│       ├── helper.h            # Helper macros
│       └── Makefile            # Build instructions
├── pyhook/                     # Python request-tracing hooks
│       ├── pyhook_agent.py     # sys.setprofile hook for vLLM
│       ├── pyhook_collector.py # Collector daemon (writes to BPF map)
│       ├── sitecustomize.py    # Auto-load bootstrap
│       └── app1.py             # Demo workload
├── scripts/                    # Analysis and training pipeline
│   ├── build_llm_dos_dataset.py      # Build ML dataset from raw traces
│   ├── train_classifiers.py          # Train RF/GB/LR classifiers
│   ├── train_time_windowed.py        # Time-windowed training variant
│   ├── run_early_detection.py        # Early detection experiments
│   ├── run_paper_experiments.py      # Full paper experiment pipeline
│   ├── analyze_trace_completeness.py # Trace quality analysis
│   └── figures/
│       ├── plot_cdf_duration.py      # Figure: duration distributions
│       └── plot_early_detection.py   # Figure: early detection performance
├── experiments/                # Prompt-sending scripts for data collection
│   ├── send_prompts_simple.py        # Send normal prompts to vLLM
│   ├── send_dos_prompts_simple.py    # Send DoS prompts to vLLM
│   ├── send_normal_prompts.py        # Batch normal prompt sender
│   ├── send_dos_prompts.py           # Batch DoS prompt sender
│   └── send_all_prompts*.sh          # Orchestration scripts
├── monitor/                    # Prometheus + Grafana monitoring
│   ├── docker-compose.yaml
│   ├── prometheus.yaml
│   └── grafana.json
├── models/                     # Trained model metadata
│   └── metadata.json
├── run_vllm.sh                 # Launch vLLM with pyhook tracing
├── run_pyhook.sh               # Launch the pyhook collector
├── pyproject.toml
└── LICENSE
```

## Prerequisites

- **Linux kernel** >= 5.15 with eBPF support
- **NVIDIA GPU** with CUDA driver installed (`libcuda.so`)
- **Python** >= 3.12
- **libbpf** >= 1.0 (for building eBPF programs)
- **clang/llvm** (for compiling BPF C code)

## Quick Start

### 1. Install Python dependencies

```bash
pip install -e .
# Or with uv:
uv sync
```

### 2. Build the eBPF tracer

```bash
cd ebpf/cuda-tracing

# Generate vmlinux.h from your running kernel
bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h

# Build
make
```

### 3. Launch vLLM with tracing

Terminal 1 — Start vLLM with the pyhook agent:
```bash
./run_vllm.sh
```

Terminal 2 — Start the pyhook collector (requires root):
```bash
./run_pyhook.sh
```

Terminal 3 — Start the eBPF tracer (requires root):
```bash
sudo ./ebpf/cuda-tracing/cuda_trace
```

### 4. Collect traces

```bash
# Send normal requests
python experiments/send_prompts_simple.py

# Send DoS requests
python experiments/send_dos_prompts_simple.py
```

### 5. Build dataset and train classifiers

```bash
python scripts/build_llm_dos_dataset.py --max-files 1000 --split 0.2
python scripts/train_classifiers.py
```

### 6. Run early detection experiments

```bash
python scripts/run_early_detection.py
```

## How It Works

1. **Tracing**: eBPF uprobes on `libcuda.so` capture GPU API calls (kernel launches, memory operations, synchronization). Kernel tracepoints capture CPU scheduling, futex, and mmap events. A Python `sys.setprofile` hook in vLLM assigns a unique trace ID to each incoming request and writes the TID-to-trace-ID mapping into a pinned BPF map.

2. **Trace reconstruction**: The user-space consumer reads events from the BPF ring buffer and groups them by trace ID to produce one trace per request.

3. **Feature extraction**: Each trace is summarized into a 30-dimensional feature vector (20 GPU features + 10 CPU features), including operation counts, latency statistics, memory sizes, and scheduling patterns.

4. **Classification**: A lightweight classifier (Random Forest by default) distinguishes normal requests from DoS attacks in real time. Early detection is possible within the first few seconds of a request.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{llmdos2025,
  title={Detecting Denial-of-Service Attacks on LLM Serving Systems via eBPF-Based GPU and CPU Tracing},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The eBPF programs (`ebpf/cuda-tracing/*.bpf.c`) are dual-licensed under GPL-2.0 (required for BPF programs loaded into the Linux kernel) and MIT.
