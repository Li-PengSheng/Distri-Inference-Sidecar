# Benchmarks

Detailed reproduction steps and notes for performance experiments referenced in the README.

**Hardware / stack for all numbers below:** `qwen2.5:1.5b` via Ollama, RTX 4060, Docker Compose sidecar + python_backend, `BACKEND_TIMEOUT_MS=120000`, `OLLAMA_TIMEOUT_S=120`. Captured on the current codebase (post FFI / vramguard / batcher hardening).

> **Note on success rates:** At 100 concurrent × 5 rounds, Ollama’s serial GPU queue routinely pushes p99 near the 120 s timeout budget. Full **500/500** completion is **not** stably reproducible on this setup; treat success counts as load-sensitive, and use flush / poll metrics for apples-to-apples engineering comparisons.

---

## Batching throughput (with vs without micro-batching)

Requires Docker Compose + Ollama + `python_backend/benchmark/batching_bench.py`.

```bash
# No batching
MAX_BATCH_SIZE=1 MAX_WAIT_MS=0 VRAM_READER_MODE=nvml BACKEND_TIMEOUT_MS=120000 \
  docker compose -p distribute up -d --force-recreate sidecar
cd python_backend
uv run benchmark/batching_bench.py --concurrent 100 --rounds 5 --timeout 180 --json

# Cool down (recommended: recreate sidecar again after ≥30s idle)

# Default batching
MAX_BATCH_SIZE=8 MAX_WAIT_MS=50 VRAM_READER_MODE=nvml BACKEND_TIMEOUT_MS=120000 \
  docker compose -p distribute up -d --force-recreate sidecar
uv run benchmark/batching_bench.py --concurrent 100 --rounds 5 --timeout 180 --json
```

### Results (latest run)

Client: `batching_bench.py --concurrent 100 --rounds 5 --timeout 180`. Sidecar metrics reset by `--force-recreate`.

| Scenario | MaxBatch | MaxWait | Client success | Avg latency | p95 latency | Throughput (success/s) | HTTP flushes (`batch_size_count`) | Avg batch (`sum/count`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no batching | 1 | 0 ms | **243/500** | 65,896 ms | 113,602 ms | 0.40 | **501** | 1.0 |
| batching | 8 | 50 ms | **64/500** | 69,039 ms | 98,952 ms | 0.11 | **66** | **7.6** |

Errors in both runs were overwhelmingly `Unavailable: backend ... context deadline exceeded` (sidecar→backend 120 s HTTP timeout under Ollama backlog). The batching scenario was run ~30 s after the no-batching scenario on the same Ollama process, so its success rate is **pessimistic** relative to a cold start.

**Takeaways**

- Workload remains **GPU-bound** (Ollama serialises inference). End-to-end success throughput does **not** improve with micro-batching under this runtime.
- The durable win is **sidecar→backend HTTP fan-out**: **501 → 66 flushes** for the same ~500 admitted requests (**~87%** fewer round-trips; avg batch **7.6**).
- Tail latency among *successful* requests is in the same order of magnitude (~100 s p95); both modes sit against the 120 s timeout wall.
- Do not expect batching alone to multiply GPU token throughput until the runtime supports true parallel / continuous batching.

---

## NVML vs SMI VRAM polling (100 concurrent, 5 rounds)

Reproduce with `python_backend/test.py`. Grafana screenshots from the same load windows: [`assets/benchmarks/nvml.png`](../assets/benchmarks/nvml.png), [`assets/benchmarks/smi.png`](../assets/benchmarks/smi.png).

```bash
# NVML
VRAM_READER_MODE=nvml docker compose -p distribute up -d --force-recreate sidecar
cd python_backend
uv run test.py --concurrent 100 --rounds 5 --expected-reader-mode nvml

# SMI (prefer a cool Ollama / recreated stack between A/B runs)
VRAM_READER_MODE=smi docker compose -p distribute up -d --force-recreate sidecar
uv run test.py --concurrent 100 --rounds 5 --expected-reader-mode nvidia-smi
```

### Results (latest `test.py` runs + Grafana)

| Metric | NVML ([nvml.png](../assets/benchmarks/nvml.png)) | nvidia-smi ([smi.png](../assets/benchmarks/smi.png)) |
|--------|------|------------|
| Reader mode metric | `nvml=1` | `nvidia-smi=1` |
| VRAM poll duration | **sub-ms** (~0.15–0.5 ms) | **~30–55 ms** spikes |
| VRAM poll errors | 0 | 0 |
| Concurrent round success (100×5) | 100 / 96 / 96 / 88 / 88 ≈ **468/500** | 80 / 72 / 62 / 16 / ~0 ≈ **~230/500** |
| Grafana Accepted / Rejected | ~465 / **1** (token limit) | **501** / **1** (VRAM circuit in window) |
| Grafana Infer success / errors | ~445 / ~20 | **231** / **270** |
| Peak VRAM (dashboard) | ~2.7 GB | ~2.4–2.5 GB |
| Micro-batch size | mostly 5.5–8.0 | mostly 6.0–8.0 |

**Takeaways**

- NVML cuts VRAM poll latency by **> 97%** versus shelling out to `nvidia-smi` (sub-ms vs tens of ms). That is the primary A/B result.
- Both modes exercise the same admission / batching path; late-round `DEADLINE_EXCEEDED` / backend HTTP deadline failures are from **Ollama queue + 120 s budgets**, not from poll read failures (`vram_poll_errors_total` stayed 0).
- The SMI run’s lower concurrent success rate is consistent with timeout sensitivity under serial GPU load (and may also reflect residual heat/backlog if A/B runs are chained). Prefer NVML in production for cheap, stable polling — not because SMI “breaks” inference.

---

## Tokenizer FFI benchmark (non-production path)

```bash
cd python_backend/benchmark
uv run tokenizer_bench.py
```

Screenshot: [`assets/benchmarks/rustvspy_v2.png`](../assets/benchmarks/rustvspy_v2.png)

This measures ctypes/FFI boundary overhead only — not production admission policy. BPE encode time is dominated by algorithm cost, not the binding layer. Compare like workloads only (e.g. Python whitespace split vs Rust whitespace split).
