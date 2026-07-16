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
# Default matches the sidecar's BACKEND_TIMEOUT_MS (120 s) so the Python side
# does not time out requests the Go side is still willing to wait for.
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "120"))
# Ollama serialises GPU inference; cap in-flight requests so a burst of
# batches queues here instead of piling up inside Ollama.
OLLAMA_MAX_CONCURRENCY = int(os.getenv("OLLAMA_MAX_CONCURRENCY", "8"))


class SingleReq(BaseModel):
    """One inference item inside a batch (mirrors the sidecar JSON shape)."""

    id: str
    input_data: bytes
    model_name: str


class BatchPayload(BaseModel):
    """Request body for POST /infer: a list of SingleReq items."""

    requests: List[SingleReq]


async def call_ollama(
    client: httpx.AsyncClient, semaphore: asyncio.Semaphore, req: SingleReq
) -> dict:
    """Call Ollama for one request; always returns an id/output_data/error dict."""

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
    # Backend is execution-only: sidecar handles admission and policy checks.
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
