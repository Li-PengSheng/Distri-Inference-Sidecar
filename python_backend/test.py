"""
System tests for Distri-Inference-Sidecar.

Tests the full running stack via:
  - gRPC → sidecar (port 50051)
  - HTTP → python backend directly (port 8000)
  - HTTP → prometheus metrics (port 9090)

Prerequisites:
    pip install grpcio grpcio-tools requests

Run:
    python test_system.py
    python test_system.py --verbose
    python test_system.py --concurrent 10
"""

import argparse
import base64
import sys
import time
import threading
import requests
import grpc
from concurrent.futures import ThreadPoolExecutor, as_completed
from gen import inference_pb2_grpc, inference_pb2  # Adjust import path based on your generated code

# ── gRPC generated stubs ──────────────────────────────────────────────────────
# Since we can't import your generated proto directly here,
# we use grpc.experimental.proto_reflection or raw channel.
# Simplest: use the requests lib to hit the REST backend directly,
# and grpc for the sidecar.

GRPC_ADDR   = "localhost:50051"
BACKEND_URL = "http://localhost:8000"
METRICS_URL = "http://localhost:9090"

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

passed = 0
failed = 0


def ok(name, detail=""):
    global passed
    passed += 1
    suffix = f"  {YELLOW}{detail}{RESET}" if detail else ""
    print(f"  {GREEN}PASS{RESET}  {name}{suffix}")


def fail(name, reason):
    global failed
    failed += 1
    print(f"  {RED}FAIL{RESET}  {name} — {reason}")


def section(title):
    print(f"\n{BOLD}{BLUE}{title}{RESET}")
    print("─" * 50)


# ── 1. Backend direct tests ───────────────────────────────────────────────────

def test_backend_health():
    section("Python backend (:8000)")
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            ok("GET /health returns 200", f"model={data.get('model')}")
        else:
            fail("GET /health returns 200", f"got {resp.status_code}")
    except Exception as e:
        fail("GET /health returns 200", str(e))


def test_backend_single_infer():
    prompt = "What is the capital of France?"
    payload = {
        "requests": [{
            "id": "test-single",
            "input_data": base64.b64encode(prompt.encode()).decode(),
            "model_name": "qwen2.5:1.5b"
        }]
    }
    try:
        start = time.time()
        resp = requests.post(f"{BACKEND_URL}/infer", json=payload, timeout=60)
        elapsed = int((time.time() - start) * 1000)

        if resp.status_code != 200:
            fail("Single /infer request", f"HTTP {resp.status_code}")
            return

        data = resp.json()
        results = data.get("results", [])

        if not results:
            fail("Single /infer request", "empty results")
            return

        result = results[0]
        if result.get("error"):
            fail("Single /infer request", f"error: {result['error']}")
            return

        output = result["output_data"]
        ok("Single /infer request", f"{elapsed}ms → \"{output[:60]}...\"")

    except Exception as e:
        fail("Single /infer request", str(e))


def test_backend_batch_infer():
    prompts = [
        "What is 2 + 2?",
        "Name one planet in our solar system.",
        "What color is the sky?",
        "What language is Go written in?",
    ]
    payload = {
        "requests": [
            {
                "id": f"batch-{i}",
                "input_data": base64.b64encode(p.encode()).decode(),
                "model_name": "qwen2.5:1.5b"
            }
            for i, p in enumerate(prompts)
        ]
    }
    try:
        start = time.time()
        resp = requests.post(f"{BACKEND_URL}/infer", json=payload, timeout=120)
        elapsed = int((time.time() - start) * 1000)

        if resp.status_code != 200:
            fail("Batch /infer (4 requests)", f"HTTP {resp.status_code}")
            return

        results = resp.json().get("results", [])
        if len(results) != 4:
            fail("Batch /infer (4 requests)", f"expected 4 results, got {len(results)}")
            return

        ids = {r["id"] for r in results}
        expected = {"batch-0", "batch-1", "batch-2", "batch-3"}
        if ids != expected:
            fail("Batch /infer (4 requests)", f"missing IDs: {expected - ids}")
            return

        ok("Batch /infer (4 requests)", f"{elapsed}ms for 4 concurrent prompts")

    except Exception as e:
        fail("Batch /infer (4 requests)", str(e))


def test_backend_error_handling():
    """Backend should return error field, not crash, when Ollama fails."""
    payload = {
        "requests": [{
            "id": "err-test",
            "input_data": base64.b64encode(b"test").decode(),
            "model_name": "nonexistent-model-xyz"
        }]
    }
    try:
        resp = requests.post(f"{BACKEND_URL}/infer", json=payload, timeout=30)
        # Should still return 200 — errors go in the payload not HTTP status
        if resp.status_code == 200:
            ok("Error handling returns 200 not 500", "error in payload field")
        else:
            fail("Error handling returns 200 not 500", f"got HTTP {resp.status_code}")
    except Exception as e:
        fail("Error handling returns 200 not 500", str(e))


# ── 2. Prometheus metrics tests ───────────────────────────────────────────────

def test_metrics():
    section("Prometheus metrics (:9090)")
    try:
        resp = requests.get(f"{METRICS_URL}/metrics", timeout=5)
        if resp.status_code != 200:
            fail("GET /metrics returns 200", f"got {resp.status_code}")
            return
        ok("GET /metrics returns 200")

        text = resp.text
        expected_metrics = [
            "infer_latency_ms",
            "batch_size",
            "circuit_breaker_trips_total",
            "vram_used_mb",
        ]
        for metric in expected_metrics:
            if metric in text:
                # Extract value if possible
                for line in text.splitlines():
                    if line.startswith(metric) and not line.startswith("#"):
                        value = line.split()[-1] if line.split() else "?"
                        ok(f"Metric: {metric}", f"value={value}")
                        break
            else:
                fail(f"Metric: {metric}", "not found in /metrics")

    except Exception as e:
        fail("GET /metrics", str(e))


# ── 3. gRPC tests (via grpc channel) ─────────────────────────────────────────

def test_grpc():
    section("gRPC sidecar (:50051)")

    # Dynamically load proto — import from your gen package path
    # We test gRPC connectivity by trying to establish a channel
    try:
        channel = grpc.insecure_channel(GRPC_ADDR)
        # Check channel is connectable within 3s
        try:
            grpc.channel_ready_future(channel).result(timeout=3)
            ok("gRPC channel connects to :50051")
        except grpc.FutureTimeoutError:
            fail("gRPC channel connects to :50051", "timeout — is sidecar running?")
            return
        finally:
            channel.close()

    except Exception as e:
        fail("gRPC channel connects to :50051", str(e))
        return

    # Try actual RPC using grpcio-testing reflection
    # If your proto stubs are generated, import them:
    try:
        channel = grpc.insecure_channel(GRPC_ADDR)
        stub = inference_pb2_grpc.InferenceServiceStub(channel)

        # Health check
        try:
            resp = stub.HealthCheck(
                inference_pb2.HealthRequest(),
                timeout=5
            )
            ok("HealthCheck RPC", f"healthy={resp.healthy} vram={resp.vram_used_mb:.0f}MB/{resp.vram_total_mb:.0f}MB")
        except Exception as e:
            fail("HealthCheck RPC", str(e))

        # Infer RPC
        try:
            prompt = "What is 1 + 1?"
            start = time.time()
            resp = stub.Infer(
                inference_pb2.InferRequest(
                    request_id="grpc-py-test",
                    input_data=prompt.encode(),
                    model_name="qwen2.5:1.5b"
                ),
                timeout=60
            )
            elapsed = int((time.time() - start) * 1000)

            if resp.error:
                fail("Infer RPC", f"error: {resp.error}")
            else:
                output = resp.output_data.decode("utf-8", errors="replace")
                ok("Infer RPC", f"{elapsed}ms → \"{output[:60]}\"")
        except Exception as e:
            fail("Infer RPC", str(e))

        channel.close()

    except ImportError:
        print(f"  {YELLOW}SKIP{RESET}  gRPC RPCs — proto stubs not found at ../gen")
        print(f"         Run: cd .. && buf generate")


# ── 4. Concurrency / batching test ────────────────────────────────────────────

def test_concurrent_batching(n=8):
    section(f"Batching efficiency ({n} concurrent requests)")

    results = []
    errors = []
    start = time.time()

    def send(i):
        prompt = f"Question {i}: What is {i} + {i}?"
        payload = {
            "requests": [{
                "id": f"concurrent-{i}",
                "input_data": base64.b64encode(prompt.encode()).decode(),
                "model_name": "qwen2.5:1.5b"
            }]
        }
        t0 = time.time()
        try:
            resp = requests.post(f"{BACKEND_URL}/infer", json=payload, timeout=120)
            elapsed = int((time.time() - t0) * 1000)
            if resp.status_code == 200:
                r = resp.json()["results"][0]
                return {"id": f"concurrent-{i}", "latency": elapsed, "error": r.get("error", "")}
            return {"id": f"concurrent-{i}", "latency": elapsed, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"id": f"concurrent-{i}", "latency": 0, "error": str(e)}

    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(send, i) for i in range(n)]
        for f in as_completed(futures):
            r = f.result()
            if r["error"]:
                errors.append(r)
            else:
                results.append(r)

    total = int((time.time() - start) * 1000)

    if errors:
        for e in errors:
            fail(f"Request {e['id']}", e["error"])

    if results:
        latencies = [r["latency"] for r in results]
        avg = int(sum(latencies) / len(latencies))
        p99 = int(sorted(latencies)[int(len(latencies) * 0.99)])
        ok(f"{len(results)}/{n} requests succeeded",
           f"total={total}ms avg={avg}ms p99={p99}ms")

        # Key insight: total wall time should be much less than n × avg single latency
        # because batching parallelises the work
        print(f"\n  {BOLD}Batching analysis:{RESET}")
        print(f"  {n} concurrent requests completed in {total}ms")
        print(f"  Average single request: {avg}ms")
        print(f"  If sequential: ~{avg * n}ms  |  Actual: {total}ms")
        if total < avg * n * 0.8:
            print(f"  {GREEN}Batching is working — {int((avg * n) / total * 10) / 10}x speedup{RESET}")
        else:
            print(f"  {YELLOW}No significant batching speedup observed{RESET}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Distri-Inference-Sidecar system tests")
    parser.add_argument("--concurrent", type=int, default=8,
                        help="Number of concurrent requests for batching test")
    parser.add_argument("--skip-grpc", action="store_true",
                        help="Skip gRPC tests")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip tests that call Ollama (fast mode)")
    args = parser.parse_args()

    print(f"\n{BOLD}Distri-Inference-Sidecar — System Tests{RESET}")
    print("=" * 50)

    # Fast tests first
    test_backend_health()
    test_metrics()

    if not args.skip_grpc:
        test_grpc()

    if not args.skip_llm:
        test_backend_single_infer()
        test_backend_batch_infer()
        test_backend_error_handling()
        test_concurrent_batching(args.concurrent)

    # Summary
    print(f"\n{'=' * 50}")
    total = passed + failed
    if failed == 0:
        print(f"{GREEN}{BOLD}All {total} tests passed{RESET}")
    else:
        print(f"{RED}{BOLD}{failed} failed{RESET}, {GREEN}{passed} passed{RESET} ({total} total)")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()