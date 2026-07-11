# Testing Guide

## Prerequisites

- Running stack via Docker Compose **or** manual sidecar + backend (see [README](../README.md#quick-start))
- For full LLM paths: NVIDIA GPU, Ollama with `qwen2.5:1.5b`, and host access from the backend container (`host.docker.internal`)

## End-to-end system test (gRPC sidecar)

```bash
cd python_backend
uv run test.py --concurrent 100 --rounds 5 --expected-reader-mode nvml
```

Skip LLM calls (connectivity, metrics, token guard only):

```bash
uv run test.py --skip-llm
```

## NVML vs nvidia-smi A/B (same load)

```bash
# NVML
VRAM_READER_MODE=nvml docker compose -p distribute up -d --build --force-recreate
cd python_backend
uv run test.py --concurrent 100 --rounds 5 --expected-reader-mode nvml

# SMI
VRAM_READER_MODE=smi docker compose -p distribute up -d --build --force-recreate
cd python_backend
uv run test.py --concurrent 100 --rounds 5 --expected-reader-mode nvidia-smi
```

Screenshots: [`assets/benchmarks/nvml.png`](../assets/benchmarks/nvml.png), [`assets/benchmarks/smi.png`](../assets/benchmarks/smi.png)

## Batching throughput benchmark

```bash
# No batching
MAX_BATCH_SIZE=1 MAX_WAIT_MS=0 VRAM_READER_MODE=nvml BACKEND_TIMEOUT_MS=120000 \
  docker compose -p distribute up -d --force-recreate sidecar
cd python_backend
uv run benchmark/batching_bench.py --concurrent 100 --rounds 5 --timeout 180 --json

# Cool down / recreate before the second scenario (recommended)

# Default batching
MAX_BATCH_SIZE=8 MAX_WAIT_MS=50 VRAM_READER_MODE=nvml BACKEND_TIMEOUT_MS=120000 \
  docker compose -p distribute up -d --force-recreate sidecar
uv run benchmark/batching_bench.py --concurrent 100 --rounds 5 --timeout 180 --json
```

See [benchmarks.md](benchmarks.md#batching-throughput-with-vs-without-micro-batching) for latest numbers.

## Go unit tests

```bash
cd rust_ops && cargo build --release && cd ..
LD_LIBRARY_PATH=$PWD/rust_ops/target/release go test ./...
LD_LIBRARY_PATH=$PWD/rust_ops/target/release go test -race ./...
```

Packages with tests: `internal/batcher`, `internal/vramguard`, `internal/grpcserver`.

## Python lint

```bash
cd python_backend
uv run ruff check .
```
