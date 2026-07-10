# Benchmarks

Detailed reproduction steps and notes for performance experiments referenced in the README.

## Batching throughput (with vs without micro-batching)

Requires Docker Compose + Ollama + `python_backend/benchmark/batching_bench.py`.

```bash
# No batching
MAX_BATCH_SIZE=1 MAX_WAIT_MS=0 VRAM_READER_MODE=nvml \
  docker compose -p distribute up -d --build --force-recreate sidecar
cd python_backend
uv run benchmark/batching_bench.py --concurrent 100 --rounds 5 --json

# Default batching
MAX_BATCH_SIZE=8 MAX_WAIT_MS=50 VRAM_READER_MODE=nvml \
  docker compose -p distribute up -d --build --force-recreate sidecar
uv run benchmark/batching_bench.py --concurrent 100 --rounds 5 --json
```

### Results (`qwen2.5:1.5b`, Ollama, 100 concurrent, 5 rounds, NVML, `BACKEND_TIMEOUT_MS=120000`)

| Scenario | Concurrency | MaxBatch | MaxWait | Avg latency | p95 / TTFT | Throughput (req/s) |
|---|---:|---:|---:|---:|---:|---:|
| no batching | 100 | 1 | 0 ms | 47,811 ms | 60,513 ms | 1.65 |
| batching | 100 | 8 | 50 ms | 59,924 ms | 60,148 ms | 1.66 |

**Takeaways**

- Workload is GPU-bound (Ollama serial inference); end-to-end throughput is similar.
- Batching cuts sidecar→backend HTTP fan-out (~500 calls → ~65 flushes, avg batch ~7.6).
- Batching slightly stabilises tail latency (p95 60.1 s vs 60.5 s) at the cost of higher average latency from the wait window.
- With a 30 s backend timeout at 100 concurrent, no-batching completes 20/100 while batching completes 0/100; default timeout is now 120 s — lower only when testing timeout behaviour.

## NVML vs SMI VRAM polling (100 concurrent, 5 rounds)

| Metric | NVML | nvidia-smi |
|--------|------|------------|
| VRAM poll p95 | < 1 ms (~0.7 ms peak) | 30–90 ms |
| Accepted requests | 501 | 501 |
| Rejected requests | 1 | 1 |
| VRAM used | ~2.47 GB | ~2.31 GB |

NVML reduces poll p95 jitter by **> 97%** with no request-outcome regression.

## Tokenizer FFI benchmark (non-production path)

```bash
cd python_backend/benchmark
uv run tokenizer_bench.py
```

Screenshot: `docs/rustvspy_v2.png`

This measures ctypes/FFI boundary overhead only — not production admission policy. BPE encode time is dominated by algorithm cost, not the binding layer. Compare like workloads only (e.g. Python whitespace split vs Rust whitespace split).
