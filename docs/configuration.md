# Configuration & API Reference

## Environment variables

| Key | Default | Description |
|---|---|---|
| `BACKEND_URL` | required | Backend `/infer` endpoint |
| `VRAM_READER_MODE` | `auto` | NVML-first in `auto`; falls back to `nvidia-smi` when NVML is unavailable. Force `nvml`, `smi`, or `nvidia-smi` (alias of `smi`) |
| `MAX_BATCH_SIZE` | `8` | Max requests per micro-batch before immediate flush |
| `MAX_WAIT_MS` | `50` | Max wait window (ms) to collect a partial batch |
| `BACKEND_TIMEOUT_MS` | `120000` | HTTP timeout (ms) per backend batch call |
| `BATCHER_DEBUG_TOKENIZE` | off | When `true`, logs per-request BPE token counts in the batcher (not for production) |

## Hard-coded sidecar defaults (`cmd/sidecar/main.go`)

| Setting | Value |
|---------|-------|
| `PollIntervalMs` | 500 |
| `OOMThresholdPct` | 90 (open circuit) |
| `CloseThresholdPct` | 85 (close circuit; hysteresis band 85–90%) |

See [limitations.md](limitations.md) for why these are not env-configurable yet.

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
| `infer_latency_ms` | histogram | Backend batch latency |
| `batch_size` | histogram | Requests per flush |
| `rejected_requests_total` | counter | Token-limit rejections |
| `circuit_breaker_trips_total` | counter | VRAM guard rejections |
| `infer_success_total` / `infer_errors_total` | counter | Per-request outcomes |
| `queue_rejects_total` | counter | Batch queue full rejections |
| `vram_used_mb` | gauge | Last VRAM reading |
| `vram_poll_duration_ms` | histogram | VRAM poll latency |
| `vram_poll_errors_total` | counter | VRAM poll failures |
| `vram_reader_mode{mode=...}` | gauge | Active reader (1=active) |

**Grafana dashboard convention**

- Accepted = `batch_size_sum`
- Rejected = `rejected_requests_total + circuit_breaker_trips_total + queue_rejects_total`
- Input Total = Accepted + Rejected

Import the dashboard from `grafana-dashboard.json` or use Docker Compose Grafana provisioning (auto-loaded from `grafana/provisioning/`).

## Project layout

```text
cmd/sidecar/            entrypoint
internal/batcher/       dynamic micro-batching
internal/grpcserver/    gRPC API
internal/metrics/       Prometheus
internal/tokenizer/     Go ↔ Rust tokenizer
internal/vramguard/     VRAM circuit breaker
python_backend/         FastAPI execution backend
rust_ops/               Rust tokenizer + C ABI
proto/                  gRPC contract
gen/                    generated Go stubs
```

## Development commands

```bash
buf generate
./scripts/fix_python_gen_imports.sh
cd rust_ops && cargo build --release && cd ..
LD_LIBRARY_PATH=$PWD/rust_ops/target/release go test -race ./...
cd python_backend && uv run ruff check .
```
