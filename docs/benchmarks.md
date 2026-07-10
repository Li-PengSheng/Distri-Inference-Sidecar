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

### Results (`qwen2.5:1.5b`, Ollama, RTX 4060, 100 concurrent, 5 rounds, NVML, `BACKEND_TIMEOUT_MS=120000`)

| Scenario | Concurrency | MaxBatch | MaxWait | Success | Avg latency | p95 / TTFT | Throughput (req/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| no batching | 100 | 1 | 0 ms | 500/500 | 47,798 ms | 60,162 ms | 1.66 |
| batching | 100 | 8 | 50 ms | 500/500 | 59,860 ms | 60,127 ms | 1.66 |

**Takeaways**

- Workload is GPU-bound (Ollama serial inference); end-to-end throughput is identical at **1.66 req/s**.
- Batching cuts sidecar→backend HTTP fan-out (**500 calls → 66 flushes**, avg batch **7.6**, ~**87%** fewer round-trips).
- Batching slightly stabilises tail latency (p95 60.1 s vs 60.2 s) at the cost of higher average latency from the 50 ms wait window.
- With `BACKEND_TIMEOUT_MS=120000`, both modes complete **500/500** requests at 100 concurrent.

## NVML vs SMI VRAM polling (100 concurrent, 5 rounds)

Reproduce with `python_backend/test.py --concurrent 100 --rounds 5`. Grafana screenshots: [`assets/benchmarks/nvml.png`](../assets/benchmarks/nvml.png), [`assets/benchmarks/smi.png`](../assets/benchmarks/smi.png).

| Metric | NVML | nvidia-smi |
|--------|------|------------|
| VRAM poll p95 | < 1 ms (~0.7 ms peak) | ~30–50 ms (avg ~30 ms) |
| Accepted requests | 501 | 401 |
| Rejected requests | 1 (token limit) | 1 (token limit) |
| Infer success / errors | 501 / 0 | 401 / 0 |
| Peak VRAM | ~2.2 GB | ~2.6 GB |
| Throughput peak | ~1.75 req/s | ~1.60 req/s |
| Micro-batch avg / p95 | 7.6 / 8.0 | 6–8 |

NVML reduces VRAM poll latency by **> 97%** (sub-ms vs ~30 ms per `nvidia-smi` subprocess). Request admission is equivalent aside from the shared token-limit rejection; the NVML run completed the full 500-request load (501 accepted incl. warmup), while the SMI screenshot reflects 401 accepted (4 completed rounds).

## Tokenizer FFI benchmark (non-production path)

```bash
cd python_backend/benchmark
uv run tokenizer_bench.py
```

Screenshot: [`assets/benchmarks/rustvspy_v2.png`](../assets/benchmarks/rustvspy_v2.png)

This measures ctypes/FFI boundary overhead only — not production admission policy. BPE encode time is dominated by algorithm cost, not the binding layer. Compare like workloads only (e.g. Python whitespace split vs Rust whitespace split).
