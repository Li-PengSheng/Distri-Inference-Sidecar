# Python Execution Backend

This FastAPI service is the execution-only HTTP backend for
`Distri-Inference-Sidecar`. It receives batches from the Go sidecar, invokes
Ollama once per item, and returns one result per input item. Token admission,
queue capacity, VRAM protection, and gRPC status mapping remain the sidecar's
responsibility.

## Run locally

```bash
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

The service exposes `POST /infer` and `GET /health`. The health endpoint is a
process-liveness probe only: it does not contact Ollama.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `BACKEND_PORT` | `8000` | Port used when starting `main.py` directly. |
| `OLLAMA_URL` | `http://host.docker.internal:11434/api/generate` | Ollama generate endpoint. |
| `DEFAULT_MODEL_NAME` | `qwen2.5:1.5b` | Model used when an item has an empty `model_name`. |
| `OLLAMA_TIMEOUT_S` | `120` | Timeout for each Ollama request. It is independent of the sidecar's whole-batch HTTP timeout, which can expire first after semaphore wait and response overhead. |
| `OLLAMA_MAX_CONCURRENCY` | `8` | Process-wide semaphore limit for simultaneous Ollama calls. |

## `POST /infer` contract

The Go sidecar sends a JSON object with a `requests` array. Each item contains
`id`, `input_data`, and `model_name`.

```json
{
  "requests": [
    {
      "id": "request-1",
      "input_data": "SGVsbG8=",
      "model_name": "qwen2.5:1.5b"
    }
  ]
}
```

`input_data` is the base64 JSON representation emitted by Go when it marshals
a `[]byte`. Pydantic represents that JSON string as `bytes`; the current
backend then decodes those received bytes as UTF-8 before supplying the prompt
to Ollama. It does not base64-decode arbitrary binary payloads, so callers
should use the sidecar rather than treating this endpoint as a general binary
inference API.

The response preserves input order and has one result per item:

```json
{
  "results": [
    {
      "id": "request-1",
      "output_data": "...",
      "error": ""
    }
  ]
}
```

Failures from an individual Ollama call are returned in that item's `error`
field while the HTTP request remains successful, allowing unrelated items in
the same batch to complete. Request-body validation failures are returned by
FastAPI before execution begins.

## Concurrency and timeouts

`POST /infer` creates one coroutine per item, but all Ollama calls acquire the
shared semaphore. This bounds load on runtimes that serialize or poorly handle
many concurrent GPU requests. A slow item can therefore wait both for the
semaphore and for Ollama; configure `OLLAMA_TIMEOUT_S` with the expected queue
time in mind.
