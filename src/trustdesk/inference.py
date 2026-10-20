"""Prediction helper. Scores are not yet calibrated probabilities."""
from __future__ import annotations
from pathlib import Path
import joblib


def load_local_model(path: str | Path):
    # WARNING: joblib loads pickle objects and can execute arbitrary code.
    # Never load untrusted files; only use artifacts generated on this machine.
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model missing: {path}. First run: python -m trustdesk.train")
    return joblib.load(path)


def classify(model, text: str, review_threshold: float = 0.4, top_k: int = 3) -> dict:
    if not (0.0 <= review_threshold <= 1.0):
        raise ValueError("review_threshold must be in [0, 1]")
    scores = model.predict_proba([text])[0]
    order = scores.argsort()[::-1][:top_k]
    top = [{"label": str(model.classes_[i]), "score": round(float(scores[i]), 5)} for i in order]
    return {
        "label": top[0]["label"],
        "score": top[0]["score"],
        "needs_human_review": top[0]["score"] < review_threshold,
        "top_predictions": top,
        "score_note": "uncalibrated model output; not a validated certainty estimate",
    }
