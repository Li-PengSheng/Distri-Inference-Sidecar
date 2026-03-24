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
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="Mock Inference Backend", version="0.1.0")


@app.post("/predict")
async def predict(data: dict) -> dict:
    """Run a mock inference and return a placeholder prediction.

    Trains a RandomForestClassifier on the Iris dataset on every call (for
    demonstration purposes only) and returns a single prediction for a
    hard-coded sample input.

    Args:
        data: Arbitrary JSON body (ignored in this mock implementation).

    Returns:
        A dict with a ``result`` key containing the stringified prediction.
    """
    iris = load_iris()
    clf = RandomForestClassifier()
    clf.fit(iris.data, iris.target)
    prediction = clf.predict([[5.1, 3.5, 1.4, 0.2]])
    return {"result": f"processed: {prediction}"}


def __main__() -> None:
    """Entry point placeholder (use uvicorn to run the app instead)."""
    pass
