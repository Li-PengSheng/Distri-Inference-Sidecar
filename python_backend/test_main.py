"""
Tests for the Python inference backend.

Run with:
    cd python_backend
    pip install pytest pytest-asyncio httpx
    pytest test_main.py -v
"""

import asyncio
import base64
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

from main import app, call_ollama, SingleReq

client = TestClient(app)


# ── Health check ──────────────────────────────────────────────────────────────

def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model" in data


# ── /infer endpoint ───────────────────────────────────────────────────────────

def test_infer_single_request():
    """Single request is forwarded to Ollama and echoed back."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"response": "Paris"}

    with patch("main.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post.return_value = mock_response

        payload = {
            "requests": [{
                "id": "req-1",
                "input_data": base64.b64encode(b"What is the capital of France?").decode(),
                "model_name": "qwen2.5-1.5b"
            }]
        }
        resp = client.post("/infer", json=payload)

    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 1
    assert results[0]["id"] == "req-1"
    assert results[0]["error"] == ""


def test_infer_batch_request():
    """Multiple requests are fanned out concurrently."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"response": "some answer"}

    with patch("main.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post.return_value = mock_response

        payload = {
            "requests": [
                {
                    "id": f"req-{i}",
                    "input_data": base64.b64encode(f"prompt {i}".encode()).decode(),
                    "model_name": "qwen2.5-1.5b"
                }
                for i in range(4)
            ]
        }
        resp = client.post("/infer", json=payload)

    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 4

    # Verify all IDs are present
    ids = {r["id"] for r in results}
    assert ids == {"req-0", "req-1", "req-2", "req-3"}

    # Verify asyncio.gather was called once (not 4 serial calls)
    assert mock_client.post.call_count == 4


def test_infer_ollama_error_is_handled():
    """If Ollama fails for one request, it returns an error field not a 500."""
    with patch("main.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post.side_effect = Exception("ollama timeout")

        payload = {
            "requests": [{
                "id": "req-fail",
                "input_data": base64.b64encode(b"hello").decode(),
                "model_name": "qwen2.5-1.5b"
            }]
        }
        resp = client.post("/infer", json=payload)

    assert resp.status_code == 200  # NOT 500 — error is in the payload
    results = resp.json()["results"]
    assert results[0]["error"] != ""
    assert results[0]["output_data"] == ""


def test_infer_empty_batch():
    """Empty batch returns empty results."""
    resp = client.post("/infer", json={"requests": []})
    assert resp.status_code == 200
    assert resp.json()["results"] == []


# ── call_ollama unit test ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_call_ollama_decodes_bytes():
    """input_data bytes are decoded to UTF-8 before sending to Ollama."""
    captured = {}

    async def fake_post(url, json=None, timeout=None):
        captured["json"] = json
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"response": "answer"}
        return mock

    mock_client = AsyncMock()
    mock_client.post.side_effect = fake_post

    req = SingleReq(id="r1", input_data=b"What is 2+2?", model_name="qwen2.5-1.5b")
    result = await call_ollama(mock_client, req)

    assert captured["json"]["prompt"] == "What is 2+2?"
    assert result["id"] == "r1"
    assert result["error"] == ""