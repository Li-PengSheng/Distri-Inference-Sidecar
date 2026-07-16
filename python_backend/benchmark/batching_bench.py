"""
Batching throughput/latency benchmark for Distri-Inference-Sidecar.

Sends concurrent gRPC Infer requests through the sidecar and reports:
  - average latency (ms)
  - p95 latency / TTFT proxy (ms)
  - throughput (req/s)
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import grpc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen import inference_pb2, inference_pb2_grpc

GRPC_ADDR = "localhost:50051"
MODEL_NAME = "qwen2.5:1.5b"


def percentile(values: list[int], p: float) -> int:
    """Return the benchmark's nearest-rank-style percentile for integer samples.

    Empty input returns 0 so an all-failure run still produces a machine-readable
    result. The index uses ``int(len(values) * p)`` capped at the last sample;
    it is intentionally a lightweight reporting statistic rather than an
    interpolated percentile implementation.
    """
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(len(ordered) * p))
    return ordered[idx]


def send_request(i: int, timeout_s: int) -> dict:
    """Issue one independently timed gRPC inference request for benchmark item ``i``.

    A channel is created and closed per request so worker threads do not share
    gRPC client state. Transport failures and backend-reported errors are both
    returned as data, allowing the benchmark to report partial success instead
    of aborting the entire concurrent scenario.
    """
    prompt = f"Question {i}: What is {i} + {i}?"
    t0 = time.perf_counter()
    channel = grpc.insecure_channel(GRPC_ADDR)
    stub = inference_pb2_grpc.InferenceServiceStub(channel)
    try:
        resp = stub.Infer(
            inference_pb2.InferRequest(
                request_id=f"bench-{i}",
                input_data=prompt.encode(),
                model_name=MODEL_NAME,
            ),
            timeout=timeout_s,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {"id": i, "latency_ms": latency_ms, "error": resp.error}
    except Exception as e:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {"id": i, "latency_ms": latency_ms, "error": str(e)}
    finally:
        channel.close()


def run_benchmark(concurrent: int, rounds: int, timeout_s: int, warmup: bool = True) -> dict:
    """Run ``rounds`` concurrent request waves and return aggregate measurements.

    Args:
        concurrent: Requests submitted simultaneously in each round.
        rounds: Number of independently timed waves.
        timeout_s: Per-RPC client deadline in seconds.
        warmup: Whether to send one excluded request before measurement.

    Returns:
        A JSON-serialisable summary containing successful-request latency,
        throughput, error count, and up to three representative errors.

    The warmup intentionally does not contribute to totals: it reduces startup
    effects without claiming that the first cold request represents steady-state
    batching behaviour.
    """
    all_latencies: list[int] = []
    errors: list[str] = []
    total_elapsed_s = 0.0

    if warmup:
        send_request(0, timeout_s)

    for _ in range(rounds):
        round_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = [executor.submit(send_request, i, timeout_s) for i in range(concurrent)]
            for f in as_completed(futures):
                result = f.result()
                if result["error"]:
                    errors.append(result["error"])
                else:
                    all_latencies.append(result["latency_ms"])
        total_elapsed_s += time.perf_counter() - round_start

    success = len(all_latencies)
    total_requests = concurrent * rounds
    throughput = success / total_elapsed_s if total_elapsed_s > 0 else 0.0

    return {
        "concurrent": concurrent,
        "rounds": rounds,
        "total_requests": total_requests,
        "success": success,
        "errors": len(errors),
        "avg_latency_ms": int(sum(all_latencies) / success) if success else 0,
        "p95_latency_ms": percentile(all_latencies, 0.95),
        "throughput_rps": round(throughput, 2),
        "sample_errors": errors[:3],
    }


def main():
    """Parse benchmark options, run the selected scenario, and set the exit code.

    JSON mode is intended for automation; both output modes exit non-zero when
    at least one measured request fails.
    """
    parser = argparse.ArgumentParser(description="Sidecar batching benchmark")
    parser.add_argument("--concurrent", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--json", action="store_true", help="print machine-readable result")
    args = parser.parse_args()

    result = run_benchmark(args.concurrent, args.rounds, args.timeout, warmup=not args.no_warmup)

    if args.json:
        print(json.dumps(result))
        sys.exit(0 if result["errors"] == 0 else 1)

    print(f"concurrent={result['concurrent']} rounds={result['rounds']}")
    print(f"success={result['success']}/{result['total_requests']} errors={result['errors']}")
    print(f"avg_latency_ms={result['avg_latency_ms']}")
    print(f"p95_latency_ms={result['p95_latency_ms']}")
    print(f"throughput_rps={result['throughput_rps']}")
    if result["sample_errors"]:
        print("sample_errors:")
        for err in result["sample_errors"]:
            print(f"  - {err}")
    sys.exit(0 if result["errors"] == 0 else 1)


if __name__ == "__main__":
    main()
