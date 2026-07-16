# Configuration & API Reference

## Environment variables

| Key | Default | Description |
|---|---|---|
| `BACKEND_URL` | required | Backend `/infer` endpoint |
| `VRAM_READER_MODE` | `auto` | `auto` and `nvml` prefer NVML, then fall back to `nvidia-smi` if NVML cannot initialise. `smi` and `nvidia-smi` force the CLI reader. On RTX 4060: NVML poll p95 **< 1 ms** vs **~30 ms** for `nvidia-smi` (see [benchmarks](benchmarks.md#nvml-vs-smi-vram-polling-100-concurrent-5-rounds)) |
| `MAX_BATCH_SIZE` | `8` | Max requests per micro-batch before immediate flush. At 100 concurrent, avg flush size **7.6** (p95 **8.0**) |
| `MAX_WAIT_MS` | `50` | Max wait window (ms) to collect a partial batch. On GPU-bound Ollama loads the dominant latency is inference queue time, not this window |
| `BACKEND_TIMEOUT_MS` | `120000` | HTTP timeout (ms) per backend batch call. At 100 concurrent on Ollama, p99 often approaches this budget; raise it if you need higher completion rates |
| `BATCHER_DEBUG_TOKENIZE` | off | When `true`, logs per-request BPE token counts in the batcher (not for production) |
| `POLL_INTERVAL_MS` | `500` | VRAM guard polling interval (ms) |
| `OOM_THRESHOLD_PCT` | `90` | VRAM utilisation (%) at which the circuit-breaker opens |
| `CLOSE_THRESHOLD_PCT` | `85` | VRAM utilisation (%) at which an open circuit closes (hysteresis band between the two) |
| `TOKENIZER_CORPUS_PATH` | unset | Path to a BPE training corpus file. Unset falls back to a small built-in English corpus (token counts poorly approximate real prompts; a startup warning is logged) |

## Python backend environment variables (`python_backend/main.py`)

| Key | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://host.docker.internal:11434/api/generate` | Ollama generate endpoint |
| `DEFAULT_MODEL_NAME` | `qwen2.5:1.5b` | Model used when a request omits `model_name` |
| `OLLAMA_TIMEOUT_S` | `120` | Per-request Ollama timeout (s). It is independent of the sidecar's whole-batch HTTP timeout, so equal values do not guarantee that the backend responds before the sidecar deadline. |
| `OLLAMA_MAX_CONCURRENCY` | `8` | Max in-flight Ollama requests (semaphore) |

## gRPC API (`proto/inference.proto`)

- `Infer(InferRequest) returns (InferResponse)`
- `HealthCheck(HealthRequest) returns (HealthResponse)`

**Status codes (sidecar ingress / transport)**

| Condition | gRPC code |
|-----------|-----------|
| Token limit exceeded | `InvalidArgument` |
| VRAM circuit open, queue full | `ResourceExhausted` |
| Backend unreachable / non-2xx HTTP | `Unavailable` |
| Invalid backend response body | `Internal` |
| Client cancel / deadline | `Canceled` / `DeadlineExceeded` |

**`InferResponse.error`** — backend execution errors only (e.g. Ollama failure with HTTP 200).

## Metrics (`:9090/metrics`, host-mapped `:9091` in Compose)

| Metric | Type | Meaning |
|--------|------|---------|
| `infer_latency_ms` | histogram | Batch flush duration: JSON preparation plus the backend HTTP round trip, excluding queue wait time |
| `batch_size` | histogram | Requests per flush |
| `rejected_requests_total` | counter | Token-limit rejections |
| `circuit_breaker_trips_total` | counter | VRAM guard rejections |
| `infer_success_total` / `infer_errors_total` | counter | Per-request outcomes |
| `queue_rejects_total` | counter | Batch queue full rejections |
| `vram_used_mb` | gauge | Last VRAM reading in MiB (the historical metric name retains an MB suffix) |
| `vram_poll_duration_ms` | histogram | VRAM poll latency |
| `vram_poll_errors_total` | counter | VRAM poll failures |
| `vram_reader_mode{mode=...}` | gauge | Active reader (1=active) |

**Grafana dashboard convention**

- Accepted = `batch_size_sum`
- Rejected = `rejected_requests_total + circuit_breaker_trips_total + queue_rejects_total`
- Input Total = Accepted + Rejected

Observed under 100 concurrent × 5 rounds (`qwen2.5:1.5b`, RTX 4060): NVML poll stays sub-ms while SMI spikes to ~30–55 ms; default batching yields **66 flushes** for ~500 admitted requests (avg batch **7.6**). Screenshots: [`assets/benchmarks/nvml.png`](../assets/benchmarks/nvml.png), [`assets/benchmarks/smi.png`](../assets/benchmarks/smi.png). Full tables in [benchmarks.md](benchmarks.md).

Dashboard is auto-loaded via Docker Compose Grafana provisioning from `grafana/provisioning/dashboards/json/distri-sidecar.json`.

## Project layout

```text
assets/
  architecture/         README diagrams
  benchmarks/           Grafana / bench screenshots
cmd/sidecar/            entrypoint
internal/batcher/       dynamic micro-batching
internal/grpcserver/    gRPC API
internal/metrics/       Prometheus
internal/tokenizer/     Go ↔ Rust tokenizer
internal/vramguard/     VRAM circuit breaker
python_backend/         FastAPI execution backend
rust_ops/               Rust tokenizer + C ABI
proto/                  gRPC contract
gen/                    generated Go stubs (do not hand-edit)
python_backend/gen/     generated Python stubs (do not hand-edit)
docs/                   markdown guides (no images)
```

## Development commands

```bash
buf generate
./scripts/fix_python_gen_imports.sh
cd rust_ops && cargo build --release && cd ..
LD_LIBRARY_PATH=$PWD/rust_ops/target/release go test -race ./...
cd python_backend && uv run ruff check .
```
