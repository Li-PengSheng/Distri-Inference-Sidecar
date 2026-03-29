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

app = FastAPI(title="Inference Backend", version="0.1.0")


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
        return {"id": req.id, "output_data": output.encode(), "error": ""}
    except Exception as e:
        return {"id": req.id, "output_data": b"", "error": str(e)}


@app.post("/infer")
async def infer(payload: BatchPayload):
    # Fan all requests in the batch out to Ollama concurrently
    async with httpx.AsyncClient() as client:
        tasks = [call_ollama(client, req) for req in payload.requests]
        results = await asyncio.gather(*tasks)
    return {"results": list(results)}


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}

if __name__ == "__main__":
    port = int(os.getenv("BACKEND_PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)