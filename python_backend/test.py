"""
System tests for Distri-Inference-Sidecar.

Tests the full running stack via:
  - gRPC -> sidecar (port 50051)
  - HTTP -> python backend directly (port 8000)
  - HTTP -> prometheus metrics (port 9091)
"""

import argparse
import base64
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import grpc
import requests
from gen import inference_pb2, inference_pb2_grpc

GRPC_ADDR = "localhost:50051"
BACKEND_URL = "http://localhost:8000"
METRICS_URL = "http://localhost:9091"

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

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
    payload = {
        "requests": [{
            "id": "test-single",
            "input_data": base64.b64encode(b"What is the capital of France?").decode(),
            "model_name": "qwen2.5:1.5b",
        }]
    }
    try:
        start = time.time()
        resp = requests.post(f"{BACKEND_URL}/infer", json=payload, timeout=60)
        elapsed = int((time.time() - start) * 1000)
        if resp.status_code != 200:
            fail("Single /infer request", f"HTTP {resp.status_code}")
            return
        results = resp.json().get("results", [])
        if not results:
            fail("Single /infer request", "empty results")
            return
        result = results[0]
        if result.get("error"):
            fail("Single /infer request", f"error: {result['error']}")
            return
        ok("Single /infer request", f"{elapsed}ms")
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
                "model_name": "qwen2.5:1.5b",
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
        ok("Batch /infer (4 requests)", f"{elapsed}ms")
    except Exception as e:
        fail("Batch /infer (4 requests)", str(e))


def test_backend_error_handling():
    payload = {
        "requests": [{
            "id": "err-test",
            "input_data": base64.b64encode(b"test").decode(),
            "model_name": "nonexistent-model-xyz",
        }]
    }
    try:
        resp = requests.post(f"{BACKEND_URL}/infer", json=payload, timeout=30)
        if resp.status_code == 200:
            ok("Error handling returns 200 not 500", "error in payload field")
        else:
            fail("Error handling returns 200 not 500", f"got HTTP {resp.status_code}")
    except Exception as e:
        fail("Error handling returns 200 not 500", str(e))


def test_grpc():
    section("gRPC sidecar (:50051)")
    try:
        channel = grpc.insecure_channel(GRPC_ADDR)
        grpc.channel_ready_future(channel).result(timeout=3)
        ok("gRPC channel connects to :50051")
        channel.close()
    except Exception as e:
        fail("gRPC channel connects to :50051", str(e))
        return

    try:
        channel = grpc.insecure_channel(GRPC_ADDR)
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        health = stub.HealthCheck(inference_pb2.HealthRequest(), timeout=5)
        ok("HealthCheck RPC", f"healthy={health.healthy}")

        start = time.time()
        resp = stub.Infer(
            inference_pb2.InferRequest(
                request_id="grpc-py-test",
                input_data=b"Hi?",
                model_name="qwen2.5:1.5b",
            ),
            timeout=120,
        )
        elapsed = int((time.time() - start) * 1000)
        if resp.error:
            fail("Infer RPC", resp.error)
        else:
            ok("Infer RPC", f"{elapsed}ms")
        channel.close()
    except Exception as e:
        fail("gRPC RPC", str(e))


def test_token_limit_rejection():
    section("Token Limit Guard (gRPC layer)")
    try:
        channel = grpc.insecure_channel(GRPC_ADDR)
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        long_input = ("hello world " * 300).encode()
        resp = stub.Infer(
            inference_pb2.InferRequest(
                request_id="test-token-limit",
                input_data=long_input,
                model_name="qwen2.5:1.5b",
            ),
            timeout=10,
        )
        if resp.error and "too long" in resp.error:
            ok("Long input rejected at gRPC layer", resp.error)
        elif resp.error:
            ok("Long input rejected", resp.error)
        else:
            fail("Long input should be rejected", "got successful response instead")
        channel.close()
    except Exception as e:
        fail("Token limit rejection test", str(e))


def test_metrics(expected_reader_mode: str | None = None):
    section("Prometheus metrics (:9091)")
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
            "rejected_requests_total",
            "vram_poll_duration_ms",
            "vram_poll_errors_total",
            "vram_reader_mode",
        ]
        for metric in expected_metrics:
            if metric in text:
                ok(f"Metric: {metric}")
            else:
                fail(f"Metric: {metric}", "not found")

        if expected_reader_mode:
            key = f'vram_reader_mode{{mode="{expected_reader_mode}"}} 1'
            if key in text:
                ok("VRAM reader mode metric", expected_reader_mode)
            else:
                fail("VRAM reader mode metric", f"expected {key}")
    except Exception as e:
        fail("Metrics test", str(e))


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
                "model_name": "qwen2.5:1.5b",
            }]
        }
        t0 = time.time()
        try:
            resp = requests.post(f"{BACKEND_URL}/infer", json=payload, timeout=120)
            elapsed = int((time.time() - t0) * 1000)
            if resp.status_code == 200:
                r = resp.json()["results"][0]
                return {"id": i, "latency": elapsed, "error": r.get("error", "")}
            return {"id": i, "latency": elapsed, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"id": i, "latency": 0, "error": str(e)}

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
        for e in errors[:5]:
            fail(f"Request {e['id']}", e["error"])
    if results:
        latencies = [r["latency"] for r in results]
        avg = int(sum(latencies) / len(latencies))
        p99 = int(sorted(latencies)[int(len(latencies) * 0.99)])
        ok(f"{len(results)}/{n} requests succeeded", f"total={total}ms avg={avg}ms p99={p99}ms")


def test_concurrent_batching_rounds(n=8, rounds=1):
    section(f"Batching rounds (concurrent={n}, rounds={rounds})")
    for i in range(rounds):
        print(f"\n  Round {i + 1}/{rounds}")
        test_concurrent_batching(n)


def main():
    parser = argparse.ArgumentParser(description="Distri-Inference-Sidecar system tests")
    parser.add_argument("--concurrent", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--expected-reader-mode", choices=["nvml", "nvidia-smi"])
    args = parser.parse_args()

    print(f"\n{BOLD}Distri-Inference-Sidecar — System Tests{RESET}")
    print("=" * 50)

    test_backend_health()
    test_grpc()
    test_token_limit_rejection()
    time.sleep(3)
    test_metrics(args.expected_reader_mode)

    if not args.skip_llm:
        test_backend_single_infer()
        test_backend_batch_infer()
        test_backend_error_handling()
        test_concurrent_batching_rounds(args.concurrent, args.rounds)

    print(f"\n{'=' * 50}")
    total = passed + failed
    if failed == 0:
        print(f"{GREEN}{BOLD}All {total} tests passed{RESET}")
    else:
        print(f"{RED}{BOLD}{failed} failed{RESET}, {GREEN}{passed} passed{RESET} ({total} total)")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
