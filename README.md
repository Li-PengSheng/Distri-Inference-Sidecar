# Distri-Inference-Sidecar

[简体中文文档](README.zh-CN.md)

A lightweight production-style **gRPC inference sidecar** that provides:

- dynamic micro-batching for throughput
- token-limit guard backed by a Rust tokenizer
- VRAM-aware circuit breaker
- Prometheus/Grafana observability
- **NVML-first** GPU polling with `nvidia-smi` fallback

---

## Features

- **Dynamic batching** (`internal/batcher`)
  - queues single requests and flushes micro-batches to backend
  - adaptive wait window based on observed QPS
- **Token limit guard** (`internal/tokenizer`, `rust_ops`)
  - rejects oversized prompts before expensive backend calls
- **VRAM guard** (`internal/vramguard`)
  - opens circuit when VRAM utilization crosses threshold
  - supports `VRAM_READER_MODE=auto|nvml|smi`
- **Observability** (`internal/metrics`)
  - request/batch/success/error metrics
  - VRAM reader mode and polling quality metrics

---

## Architecture

```text
gRPC client (:50051)
    -> grpcserver.Infer()
       -> tokenizer.Validate()
       -> batcher.Submit()
          -> flushBatch() -> HTTP /infer (python_backend:8000)

vramguard.Start()
    -> reader: NVML (preferred) or nvidia-smi fallback
    -> circuit open/close state

metrics exposed at :9090/metrics
```

---

## Prerequisites

- Go 1.25+
- Rust 1.85+
- Python 3.12+ with `uv`
- NVIDIA driver + NVML available
- Docker + Docker Compose (recommended)

---

## Quick Start

### Docker Compose (recommended)

```bash
docker compose -p distribute up -d --build
```

Services:

- backend: `:8000`
- sidecar gRPC: `:50051`
- sidecar metrics: `:9091` (container `:9090`)
- prometheus: `:9090`
- grafana: `:3000`

### Local run (manual)

```bash
# terminal 1: backend
cd python_backend
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000

# terminal 2: sidecar
cd ..
go build ./cmd/sidecar
BACKEND_URL=http://localhost:8000/infer ./sidecar
```

---

## Configuration

| Key | Default | Description |
|---|---|---|
| `BACKEND_URL` | required | backend `/infer` endpoint |
| `VRAM_READER_MODE` | `auto` | default behavior is **NVML first** in `auto`; fallback to `nvidia-smi` only when NVML is unavailable. Can force `nvml` or `smi` |
| `PollIntervalMs` | `500` | VRAM polling interval |
| `OOMThresholdPct` | `90` | circuit-breaker threshold |
| `MaxBatchSize` | `8` | max requests per flush |
| `MaxWaitMs` | `50` | max wait before partial flush |

---

## gRPC API

Defined in `proto/inference.proto`:

- `Infer(InferRequest) returns (InferResponse)`
- `HealthCheck(HealthRequest) returns (HealthResponse)`

---

## Metrics

Core metrics:

- `infer_latency_ms` (histogram)
- `batch_size` (histogram)
- `rejected_requests_total`
- `circuit_breaker_trips_total`
- `infer_success_total`
- `infer_errors_total`
- `vram_used_mb`
- `vram_poll_duration_ms`
- `vram_poll_errors_total`
- `vram_reader_mode{mode="nvml|nvidia-smi"}`

---

## Tests and Results

### 1) End-to-end system test (gRPC sidecar path)

```bash
cd python_backend
uv run test.py --concurrent 100 --rounds 5 --expected-reader-mode nvml
```

### 2) NVML vs SMI A/B test (same load)

```bash
# NVML run
VRAM_READER_MODE=nvml docker compose -p distribute up -d --build --force-recreate
cd python_backend
uv run test.py --concurrent 100 --rounds 5 --expected-reader-mode nvml

# SMI run
VRAM_READER_MODE=smi docker compose -p distribute up -d --build --force-recreate
cd python_backend
uv run test.py --concurrent 100 --rounds 5 --expected-reader-mode nvidia-smi
```

Screenshots:

- SMI mode: ![](docs/smi.png)
- NVML mode: ![](docs/nvml.png)

Observed outcome:

- reader mode switches correctly (`nvml=1` vs `nvidia-smi=1`)
- VRAM poll p95 drops from tens of ms (SMI) to sub-ms (NVML)
- no obvious request-outcome regression under the same load

### 3) Python vs Rust tokenizer benchmark

```bash
cd python_backend/benchmark
uv run tokenizer_bench.py
```

Benchmark screenshot:

![](docs/rustvspy.png)

Notes:

- whitespace counting: FFI overhead is amortized and Rust/PyO3 shows ~2x+ speedup over pure Python
- BPE encode: the dominant bottleneck is algorithm complexity in the current BPE implementation, not the binding layer
- takeaway: FFI acceleration is most effective when compute dominates boundary overhead; for high-frequency short calls, batch APIs are needed to amortize crossing cost
- output explicitly marks whether batch path is true FFI (`[ffi batch]`) or fallback

---

## Project Structure

```text
cmd/sidecar/            # entrypoint
internal/batcher/       # dynamic micro-batching
internal/grpcserver/    # gRPC API implementation
internal/metrics/       # Prometheus metrics
internal/tokenizer/     # Go <-> Rust tokenizer bridge
internal/vramguard/     # NVML/smi VRAM circuit breaker
python_backend/         # FastAPI backend and tests
rust_ops/               # Rust tokenizer + C ABI + PyO3 module
docs/                   # result screenshots
```

---

## Development

```bash
# regenerate protobuf stubs
buf generate

# go sanity
go test ./...

# python lint
cd python_backend
uv run ruff check .
```

---

## License

This project is for educational and experimental use.
