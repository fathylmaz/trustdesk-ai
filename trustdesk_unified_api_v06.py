"""TrustDesk AI v0.6 — local-only model comparison API.

Loads the existing TF-IDF model and a separately fine-tuned DistilBERT.
Neither model is calibrated for production use. This module NEVER executes
banking actions, enables automatic routing, or asserts OOD safety.

Run (from the trustdesk-ai-starter project root):
  python -m uvicorn trustdesk_unified_api_v06:app --host 127.0.0.1 --port 8002
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Reuse the real project's classes; avoid copying inference implementations.
from trustdesk.inference import classify as baseline_classify, load_local_model
from trustdesk_transformer_api_v05 import LocalTransformerPredictor

BASELINE_DEFAULT = Path("artifacts/baseline.joblib")
TRANSFORMER_DEFAULT = Path("artifacts/transformer_benchmark_4epochs/distilbert_model")
MODEL_TYPES = ("tfidf", "distilbert")


class TextRequest(BaseModel):
    text: str = Field(min_length=3, max_length=2000, description="English customer support text")


class SingleModelRequest(TextRequest):
    model: Literal["tfidf", "distilbert"]


class RankedPrediction(BaseModel):
    label: str
    score: float


class PredictionResult(BaseModel):
    model: Literal["tfidf", "distilbert"]
    label: str
    score: float
    top_predictions: list[RankedPrediction]
    score_note: str
    inference_ms: float = Field(ge=0)
    automatic_routing_enabled: Literal[False] = False


class CompareResponse(BaseModel):
    text: str
    predictions: list[PredictionResult]
    warning: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast instead of starting an API whose models were not loaded.
    baseline_path = Path(os.getenv("TRUSTDESK_UNIFIED_BASELINE_PATH", str(BASELINE_DEFAULT)))
    transformer_path = Path(os.getenv("TRUSTDESK_UNIFIED_TRANSFORMER_DIR", str(TRANSFORMER_DEFAULT)))
    device = os.getenv("TRUSTDESK_UNIFIED_DEVICE", "auto").lower()
    app.state.baseline = load_local_model(baseline_path)
    app.state.transformer = LocalTransformerPredictor(transformer_path, device=device)
    # Serializes local GPU inference in this learning-stage API.
    app.state.predict_lock = Lock()
    try:
        yield
    finally:
        app.state.baseline = None
        app.state.transformer = None


app = FastAPI(
    title="TrustDesk AI — Local Model Comparison",
    version="0.6.0-experimental",
    description=("Local educational benchmark service. Scores from different models "
                 "must not be interpreted as comparable, calibrated certainty; "
                 "automatic routing and live banking actions are disabled."),
    lifespan=lifespan,
)

# Only local React development origins; NEVER permissive '*'.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _ready(request: Request):
    baseline = getattr(request.app.state, "baseline", None)
    transformer = getattr(request.app.state, "transformer", None)
    if baseline is None or transformer is None:
        raise HTTPException(status_code=503, detail="Local models not loaded")
    return baseline, transformer


def _predict(model_name: Literal["tfidf", "distilbert"], text: str, baseline, transformer) -> PredictionResult:
    t0 = perf_counter()
    if model_name == "tfidf":
        # The old inference method has a heuristic review flag; deliberately
        # drop that flag: comparison API does not make routing decisions.
        result = baseline_classify(baseline, text, review_threshold=0.0)
        score_note = "Uncalibrated baseline scores; do not interpret as validated certainty."
    else:
        result = transformer.classify(text)
        score_note = result.get("score_note", "Uncalibrated Transformer softmax scores.")
    ms = round((perf_counter() - t0) * 1000.0, 3)
    return PredictionResult(
        model=model_name,
        label=result["label"],
        score=result["score"],
        top_predictions=result["top_predictions"],
        score_note=score_note,
        inference_ms=ms,
        automatic_routing_enabled=False,
    )


@app.get("/health")
def health(request: Request):
    baseline = getattr(request.app.state, "baseline", None)
    transformer = getattr(request.app.state, "transformer", None)
    loaded = baseline is not None and transformer is not None
    return {"status": "ok" if loaded else "not_loaded", "models_loaded": loaded,
            "device": getattr(transformer, "device", None) if loaded else None,
            "automatic_routing_enabled": False}


@app.get("/v1/models")
def models():
    return {"models": [
        {"id": "tfidf", "description": "TF-IDF + Logistic Regression baseline (80% train split)"},
        {"id": "distilbert", "description": "DistilBERT fine-tuned 4 epochs (64% train split)"},
    ], "note": "Production safety, OOD rejection and calibrated confidence have not been established."}


@app.post("/v1/predict", response_model=PredictionResult)
def single_predict(payload: SingleModelRequest, request: Request):
    baseline, transformer = _ready(request)
    with request.app.state.predict_lock:
        return _predict(payload.model, payload.text, baseline, transformer)


@app.post("/v1/compare", response_model=CompareResponse)
def compare(payload: TextRequest, request: Request):
    baseline, transformer = _ready(request)
    with request.app.state.predict_lock:
        results = [_predict(name, payload.text, baseline, transformer) for name in MODEL_TYPES]
    return CompareResponse(
        text=payload.text,
        predictions=results,
        warning=("Experimental local demo only. Both models always return one of the 77 banking intents, "
                 "even for out-of-domain text. Do not compare raw confidence scores or automate banking actions."),
    )
