"""Mock inference backend for the Distri-Inference-Sidecar.

This FastAPI application simulates an ML model inference server. It is
intentionally minimal and is used for local development and integration
testing only. In production, replace this with a real model-serving runtime
(e.g. TorchServe, Triton, vLLM, etc.).

The sidecar forwards batched requests to POST /infer. This module exposes a
simple POST /predict endpoint that trains a throwaway RandomForest on the
Iris dataset and returns a prediction — purely to demonstrate the shape of the
request/response without requiring a GPU.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import time, base64

app = FastAPI(title="Mock Inference Backend", version="0.1.0")

class SingleReq(BaseModel):
    id: str
    input_data: bytes
    model_name: str

class BatchPayload(BaseModel):
    requests: List[SingleReq]

@app.post("/infer")
async def infer(payload: BatchPayload):
    results = []
    for req in payload.requests:
        # Replace this block with real model logic later
        output = f"echo:{base64.b64encode(req.input_data).decode()}"
        results.append({
            "id": req.id,
            "output_data": output.encode(),
            "error": ""
        })
    return {"results": results}