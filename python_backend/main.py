"""Execution-only FastAPI inference backend for the Distri-Inference-Sidecar.

Receives batched POST /infer payloads from the Go sidecar, forwards each
prompt to Ollama under a concurrency semaphore, and returns per-request
results. Token admission and VRAM policy live in the sidecar, not here.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager

sys.path.append(os.path.join(os.path.dirname(__file__), "gen"))

from typing import List

import httpx
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434/api/generate")
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "qwen2.5:1.5b")
# This is the timeout for one Ollama call, not the sidecar's full batch HTTP
# request. Equal defaults keep configuration simple, but the outer sidecar
# timeout can still expire first because a batch also includes semaphore wait
# time and response processing.
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "120"))
# Limit in-flight Ollama calls because some runtimes serialize GPU work or
# degrade under bursts. Queueing here makes the concurrency bound explicit
# instead of allowing unbounded work to accumulate inside the runtime.
OLLAMA_MAX_CONCURRENCY = int(os.getenv("OLLAMA_MAX_CONCURRENCY", "8"))


class SingleReq(BaseModel):
    """One inference item from the sidecar JSON batch.

    ``input_data`` is received in the JSON representation produced by Go's
    ``[]byte`` encoder (a base64 string represented here as bytes). The current
    execution path decodes those received bytes as UTF-8 before sending them to
    Ollama; it does not accept an arbitrary binary model payload.
    """

    id: str
    input_data: bytes
    model_name: str


class BatchPayload(BaseModel):
    """Request body for POST /infer: a list of SingleReq items."""

    requests: List[SingleReq]


async def call_ollama(
    client: httpx.AsyncClient, semaphore: asyncio.Semaphore, req: SingleReq
) -> dict:
    """Call Ollama for one request and convert its outcome to a result object.

    Args:
        client: Shared asynchronous HTTP client owned by the FastAPI lifespan.
        semaphore: Process-wide limit on simultaneous Ollama calls.
        req: Request whose prompt bytes and optional model name are forwarded.

    Returns:
        An ``id``, ``output_data``, and ``error`` mapping. ``error`` is empty
        only when Ollama returns a successful response.

    Side effects:
        Sends an HTTP request to Ollama after acquiring the shared semaphore.

    Errors:
        Network, timeout, HTTP-status, and response-processing exceptions are
        caught and returned as a per-item error. This containment is deliberate:
        one model failure must not turn an otherwise valid sidecar batch into a
        single failed HTTP response for every sibling request.
    """

    prompt = req.input_data.decode("utf-8", errors="replace")

    try:
        async with semaphore:
            resp = await client.post(
                OLLAMA_URL,
                json={
                    "model": req.model_name or DEFAULT_MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=OLLAMA_TIMEOUT_S,
            )
        resp.raise_for_status()
        output = resp.json().get("response", "")
        return {"id": req.id, "output_data": output, "error": ""}
    except Exception as e:
        return {"id": req.id, "output_data": "", "error": f"{type(e).__name__}: {e}"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create a shared httpx client and Ollama concurrency semaphore."""

    client = httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_S)
    app.state.http_client = client
    app.state.ollama_semaphore = asyncio.Semaphore(OLLAMA_MAX_CONCURRENCY)
    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(title="Inference Backend", version="0.1.0", lifespan=lifespan)


@app.post("/infer")
async def infer(payload: BatchPayload, request: Request):
    """Execute every request in a sidecar batch and return per-item results.

    Args:
        payload: Requests collected by the Go sidecar.
        request: FastAPI request providing the shared HTTP client and semaphore.

    Returns:
        A ``results`` envelope in input order, with one id/output_data/error
        object per submitted request.

    Side effects:
        Sends one Ollama HTTP request per item. All coroutines are scheduled
        together, while the shared semaphore bounds actual Ollama concurrency.

    Errors:
        Per-item exceptions are converted by ``call_ollama`` into result errors
        so one failed request does not discard successful siblings in the batch.
        Validation errors are handled by FastAPI before this function runs.
    """
    # This backend only executes requests; admission and VRAM policy belong to
    # the sidecar so all ingress paths apply the same limits before batching.
    client: httpx.AsyncClient = request.app.state.http_client
    semaphore: asyncio.Semaphore = request.app.state.ollama_semaphore
    tasks = [call_ollama(client, semaphore, req) for req in payload.requests]
    results = await asyncio.gather(*tasks)
    return {"results": results}


@app.get("/health")
async def health():
    """Liveness probe used by compose/ops; does not query Ollama."""

    return {
        "status": "ok",
        "model": DEFAULT_MODEL_NAME,
    }


if __name__ == "__main__":
    port = int(os.getenv("BACKEND_PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
