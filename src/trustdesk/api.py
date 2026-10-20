"""Local-development-only FastAPI service for trained classifier."""
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from trustdesk.inference import classify, load_local_model


class PredictRequest(BaseModel):
    text: str = Field(min_length=3, max_length=2000, description="English customer support question")


class RankedPrediction(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    label: str
    score: float
    needs_human_review: bool
    top_predictions: list[RankedPrediction]
    score_note: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.environ.get("TRUSTDESK_MODEL_PATH", "artifacts/baseline.joblib")
    app.state.model = load_local_model(model_path)
    yield


app = FastAPI(title="TrustDesk AI - Baseline", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request):
    review_threshold = float(os.environ.get("TRUSTDESK_REVIEW_THRESHOLD", "0.4"))
    return classify(request.app.state.model, payload.text, review_threshold=review_threshold)
