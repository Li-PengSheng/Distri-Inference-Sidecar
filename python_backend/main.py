import asyncio
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "gen"))

from typing import List

import httpx
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


OLLAMA_URL = "http://host.docker.internal:11434/api/generate"
MODEL_NAME = "qwen2.5:1.5b"
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "512"))
TOKENIZER_VOCAB_SIZE = int(os.getenv("TOKENIZER_VOCAB_SIZE", "500"))
TOKENIZER_TRAIN_CORPUS = os.getenv(
    "TOKENIZER_TRAIN_CORPUS",
    "hello world foo bar " * 500,
)

app = FastAPI(title="Inference Backend", version="0.1.0")

try:
    import rust_ops as rust_ops_py
except ImportError:
    rust_ops_py = None

if rust_ops_py is not None:
    rust_ops_py.py_bpe_train(TOKENIZER_TRAIN_CORPUS, TOKENIZER_VOCAB_SIZE)


class SingleReq(BaseModel):
    id: str
    input_data: bytes
    model_name: str


class BatchPayload(BaseModel):
    requests: List[SingleReq]


async def call_ollama(client: httpx.AsyncClient, req: SingleReq) -> dict:
    prompt = req.input_data.decode("utf-8", errors="replace")

    try:
        resp = await client.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        output = resp.json().get("response", "")
        return {"id": req.id, "output_data": output, "error": ""}  # ← string, not bytes
    except Exception as e:
        return {"id": req.id, "output_data": "", "error": str(e)}


@app.post("/infer")
async def infer(payload: BatchPayload):
    prompts = [req.input_data.decode("utf-8", errors="replace") for req in payload.requests]
    if rust_ops_py is not None:
        token_counts = rust_ops_py.py_bpe_encode_lens(prompts)
        if token_counts is None:
            token_counts = [len(p.split()) for p in prompts]
    else:
        token_counts = [len(p.split()) for p in prompts]

    accepted_reqs: list[SingleReq] = []
    accepted_indices: list[int] = []
    ordered_results: list[dict | None] = [None] * len(payload.requests)

    for idx, (req, token_count) in enumerate(zip(payload.requests, token_counts)):
        if token_count > MAX_INPUT_TOKENS:
            ordered_results[idx] = {
                "id": req.id,
                "output_data": "",
                "error": f"input too long: {token_count} tokens (max {MAX_INPUT_TOKENS})",
            }
            continue
        accepted_reqs.append(req)
        accepted_indices.append(idx)

    # Fan valid requests in the batch out to Ollama concurrently.
    async with httpx.AsyncClient() as client:
        tasks = [call_ollama(client, req) for req in accepted_reqs]
        accepted_results = await asyncio.gather(*tasks)

    for idx, result in zip(accepted_indices, accepted_results):
        ordered_results[idx] = result

    return {"results": [result for result in ordered_results if result is not None]}


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}

if __name__ == "__main__":
    port = int(os.getenv("BACKEND_PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)