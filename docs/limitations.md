# Known Limitations

This document records intentional constraints and operational caveats for Distri-Inference-Sidecar as of the current release.

## VRAM threshold constants are not configurable via environment variables

`OOMThresholdPct` (default **90**) and `CloseThresholdPct` (default **85**) are hard-coded in `cmd/sidecar/main.go`. To change the hysteresis band you must edit the source and rebuild the sidecar binary. There is no `OOM_THRESHOLD_PCT` or `CLOSE_THRESHOLD_PCT` env var yet.

## gRPC client cancellation does not dequeue early

When a caller cancels or times out, the request may already be sitting in the batcher queue. The sidecar does **not** remove it from the queue at cancel time. Cancellation is honoured only at `flushBatch`, immediately before the HTTP call to the backend: cancelled requests are filtered out and receive a terminal error on their `ResultCh`.

Practical effect: a cancelled request still occupies a queue slot (capacity 1 000) until its batch is flushed or the process stops. Under heavy cancel churn this can delay other work even though no backend compute is wasted.

## Prometheus startup depends on sidecar health

In `docker-compose.yaml`, `prometheus` uses `depends_on: sidecar: condition: service_healthy`. Prometheus will not start scraping until the sidecar passes its HTTP probe at `:9090/health`.

The sidecar healthcheck is configured with **`start_period: 20s`** (plus `interval: 10s`, `timeout: 5s`, `retries: 5`). On machines **without a GPU**, NVML may be unavailable and the guard falls back to `nvidia-smi`; first boot can take longer while the stack warms up. If Prometheus stays in `starting` state, increase the sidecar `start_period` to **40–60s** in `docker-compose.yaml`:

```yaml
sidecar:
  healthcheck:
    start_period: 60s
```

## Build and tests depend on `rust_ops`

The Go tokenizer bridge (`internal/tokenizer`) links against `librust_ops.so` via CGO. You must build the Rust library before compiling the sidecar or running tests that call `tokenizer.Init` (including `internal/grpcserver` integration tests).

```bash
cd rust_ops && cargo build --release
cd ..

# Option A: build sidecar with the library on the loader path
LD_LIBRARY_PATH=$PWD/rust_ops/target/release go build ./cmd/sidecar

# Option B: run Go tests (including -race)
LD_LIBRARY_PATH=$PWD/rust_ops/target/release go test -race ./...
```

Without this step you will see link errors (`librust_ops.so: cannot open shared object file`) or test failures in packages that initialise the tokenizer.

## Micro-batching vs GPU throughput with Ollama

Benchmarks on `qwen2.5:1.5b` via Ollama on RTX 4060 (100 concurrent, 5 rounds, NVML mode) show **identical end-to-end throughput** with and without batching (**1.66 req/s**) because Ollama serialises GPU inference. The main win from micro-batching is **fewer sidecar→backend HTTP calls** (500 requests → 66 flushes, average batch size 7.6, ~87% reduction), plus slightly more stable tail latency (p95 60.1 s vs 60.2 s). Do not expect batching alone to multiply GPU token throughput until the runtime supports true parallel or continuous batching.

## Mixed `model_name` values in one batch

The batcher groups requests by time and count, **not** by `model_name`. Each request carries its own `model_name`; the Python backend invokes the model per item inside the batch. This is correct for the current execution-only backend.

If you later adopt a runtime that requires **homogeneous batches** (e.g. continuous batching kernels that assume one model per batch), the collection logic in `internal/batcher` must be extended to partition by `model_name` (or another key) before flush.
