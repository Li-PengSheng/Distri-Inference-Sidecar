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

Screenshots: `docs/nvml_v2.png`, `docs/smi_v2.png`

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
