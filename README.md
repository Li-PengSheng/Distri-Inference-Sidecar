# Distri-Inference-Sidecar

[简体中文文档](README.zh-CN.md)

A **gRPC inference sidecar** that batches concurrent requests, enforces token limits at ingress, protects GPU memory with a circuit breaker, and exports Prometheus metrics — without changing your model runtime.

---

## Architecture

![](assets/architecture/sidecar-architecture.png)

```text
gRPC client (:50051)
  → tokenizer.Validate()     # admission (authoritative)
  → batcher.Submit()         # micro-batch
  → HTTP POST /infer         # python_backend :8000 → Ollama

vramguard                    # NVML-first VRAM polling, hysteresis circuit breaker
metrics                      # :9090/metrics (+ /health for probes)
```

The Python backend is **execution-only**; it does not perform token admission. See [configuration](docs/configuration.md) for API and env reference.

---

## Key engineering decisions

1. **Admission at the sidecar, not the backend** — BPE token counting runs in Go via `rust_ops` before any batching or backend call. Rejections surface as gRPC `InvalidArgument`; backend errors stay in `InferResponse.error`.

2. **Micro-batching for HTTP fan-out, not GPU parallelism** — Requests are grouped by count and a short wait window. With Ollama’s serial inference, success throughput stays GPU-bound; the win is fewer sidecar→backend round-trips (**501 → 66 flushes**, avg batch **7.6**, ~**87%** fewer calls). Details in [benchmarks](docs/benchmarks.md).

3. **NVML-first VRAM guard with hysteresis** — Poll via NVML (< 1 ms p95) and fall back to `nvidia-smi`. Circuit opens at 90% VRAM, closes at 85%, avoiding flapping. Overload maps to gRPC `ResourceExhausted`.

---

## Quick Start

**Docker Compose (recommended)**

```bash
docker compose -p distribute up -d --build
```

| Service | Port |
|---------|------|
| backend | `:8000` |
| sidecar gRPC | `:50051` |
| sidecar metrics | `:9091` (container `:9090`) |
| Prometheus | `:9090` |
| Grafana | `:3000` (dashboard: **Distri-Inference-Sidecar**) |

**Local (two terminals)**

```bash
# backend
cd python_backend && uv sync && uv run uvicorn main:app --host 0.0.0.0 --port 8000

# sidecar (from repo root; build rust_ops first — see limitations)
cd rust_ops && cargo build --release && cd ..
LD_LIBRARY_PATH=$PWD/rust_ops/target/release \
  BACKEND_URL=http://localhost:8000/infer go run ./cmd/sidecar
```

Prerequisites: Go 1.25+, Rust 1.85+, Python 3.12+ (`uv`), NVIDIA driver (for VRAM guard in production paths).

---

## Key experimental conclusions

| Experiment | Headline result |
|------------|-----------------|
| [NVML vs SMI](docs/benchmarks.md#nvml-vs-smi-vram-polling-100-concurrent-5-rounds) | NVML poll **sub-ms** vs SMI **~30–55 ms** (**> 97%** lower); both hit 120 s timeout walls under 100-concurrent Ollama load |
| [Batching on/off](docs/benchmarks.md#batching-throughput-with-vs-without-micro-batching) | GPU-bound; HTTP flushes **501 → 66** (~87% fewer); success rate limited by 120 s timeouts at 100 concurrent |
| Token guard | Oversized prompts rejected at gRPC layer before backend contact |

Full reproduction: [docs/testing.md](docs/testing.md) · [docs/benchmarks.md](docs/benchmarks.md)

---

## Known limitations

Cancel/dequeue behaviour, Compose health ordering, `rust_ops` build deps, mixed-model batches, and Ollama throughput / timeout caveats are documented in **[docs/limitations.md](docs/limitations.md)**.

---

## License

Educational and experimental use.
