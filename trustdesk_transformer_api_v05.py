"""TrustDesk AI v0.5: local-only experimental DistilBERT inference service.

NO live-banking actions, automatic routing, calibration or OOD claims.
Start with:
  python -m uvicorn trustdesk_transformer_api_v05:app --host 127.0.0.1 --port 8001
Or smoke-test with:
  python trustdesk_transformer_api_v05.py --smoke --device cuda
"""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import json
import math
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

DEFAULT_MODEL_DIR = Path("artifacts/transformer_benchmark_4epochs/distilbert_model")
MAX_LENGTH = 96  # Use the same truncation length as training.


class PredictRequest(BaseModel):
    text: str = Field(min_length=3, max_length=2000, description="English banking support request")


class RankedPrediction(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    label: str
    score: float
    top_predictions: list[RankedPrediction]
    automatic_routing_enabled: bool
    score_note: str


def checked_labels(id2label: dict[Any, str], num_labels: int) -> list[str]:
    """Validate model label metadata; NEVER assume class index order."""
    if num_labels <= 0:
        raise ValueError("Model has no output classes")
    labels = []
    for idx in range(num_labels):
        label = id2label.get(idx, id2label.get(str(idx)))
        if not isinstance(label, str) or not label or label.startswith("LABEL_"):
            raise ValueError(f"Missing genuine BANKING77 label mapping for output class {idx}")
        labels.append(label)
    if len(set(labels)) != num_labels:
        raise ValueError("Duplicate labels in saved model config")
    return labels


def rank_probabilities(probabilities: list[float], labels: list[str], top_k: int = 3) -> list[dict[str, Any]]:
    """Turn softmax outputs into labelled rankings (not calibrated certainty)."""
    if not probabilities or len(probabilities) != len(labels):
        raise ValueError("Mismatch between probabilities and labels")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities):
        raise ValueError("Non-finite or invalid softmax probabilities")
    total = sum(probabilities)
    if not math.isclose(total, 1.0, abs_tol=1e-4):
        raise ValueError(f"Softmax probabilities do not sum to 1: {total}")
    order = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:top_k]
    return [{"label": labels[i], "score": round(float(probabilities[i]), 5)} for i in order]


class LocalTransformerPredictor:
    def __init__(self, model_dir: Path, device: str = "auto"):
        """Load ONLY locally created model files. No HF network/model-code downloads."""
        if device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be auto, cuda or cpu")
        if not (model_dir / "config.json").is_file() or not (model_dir / "model.safetensors").is_file():
            raise FileNotFoundError(f"Missing saved model in {model_dir}. Check the 4-epoch output directory.")
        if not (model_dir / "tokenizer.json").is_file() and not (model_dir / "vocab.txt").is_file():
            raise FileNotFoundError(f"Missing local tokenizer in {model_dir}")

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable. Verify GPU and CUDA PyTorch installation.")
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False, use_safetensors=True
        ).to(self.device)
        self.model.eval()
        self.labels = checked_labels(self.model.config.id2label, self.model.config.num_labels)

    def classify(self, text: str) -> dict[str, Any]:
        tokenized = self.tokenizer([text], truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        with self.torch.inference_mode():
            logits = self.model(**tokenized).logits
            probs = self.torch.softmax(logits.float(), dim=-1)[0].cpu().tolist()
        ranked = rank_probabilities(probs, self.labels)
        return {
            "label": ranked[0]["label"],
            "score": ranked[0]["score"],
            "top_predictions": ranked,
            "automatic_routing_enabled": False,
            "score_note": "Experimental uncalibrated softmax scores; not validated certainty, OOD rejection or banking safety.",
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.environ.get("TRUSTDESK_TRANSFORMER_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    device = os.environ.get("TRUSTDESK_TRANSFORMER_DEVICE", "auto").lower()
    app.state.predictor = LocalTransformerPredictor(model_path, device=device)
    yield
    app.state.predictor = None


app = FastAPI(
    title="TrustDesk AI - Experimental DistilBERT",
    description="Local experiment only. Does not authorize real banking actions or automatic routing.",
    version="0.5.0-experimental",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    return {"status": "ok" if predictor else "not_loaded", "model_loaded": predictor is not None,
            "device": predictor.device if predictor else None}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request):
    return request.app.state.predictor.classify(payload.text)


def main():
    parser = argparse.ArgumentParser(description="TrustDesk AI local DistilBERT smoke test")
    parser.add_argument("--smoke", action="store_true", help="Load local checkpoint and classify three examples")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    args = parser.parse_args()
    if not args.smoke:
        parser.error("Use --smoke or start the API with python -m uvicorn trustdesk_transformer_api_v05:app --host 127.0.0.1 --port 8001")
    predictor = LocalTransformerPredictor(Path(args.model_dir), args.device)
    examples = [
        "My card was stolen yesterday.",
        "My card payment was declined at the supermarket.",
        "How humid is the air in the city today?",
    ]
    print("\n=== LOCAL DISTILBERT SMOKE TEST ===")
    print("Device:", predictor.device, "| Labels:", len(predictor.labels))
    for sample in examples:
        print(json.dumps({"text": sample, **predictor.classify(sample)}, ensure_ascii=False, indent=2))
    print("NOTE: OOD examples may still receive a banking label. Auto-routing is DISABLED.")


if __name__ == "__main__":
    main()
