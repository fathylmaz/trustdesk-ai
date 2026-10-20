r"""TrustDesk AI | v0.1 validation-only review-threshold evaluation.

Place alongside pyproject.toml in trustdesk-ai-starter and run:
    .\.venv\Scripts\python.exe .\trustdesk_threshold_evaluation.py

Requires the BANKING77 training CSV and your own trained baseline.joblib.
This script does NOT retrain, change the API threshold, or inspect the official test set.
All model scores are UNCALIBRATED: do not interpret as certainty.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from trustdesk.data import load_csv
from trustdesk.inference import load_local_model

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "raw" / "train.csv"
MODEL = BASE / "artifacts" / "baseline.joblib"
BASELINE_METRICS = BASE / "artifacts" / "metrics.json"
ERROR_SUMMARY = BASE / "artifacts" / "error_analysis" / "summary.json"
OUTPUT_DIR = BASE / "artifacts" / "threshold_evaluation"
THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
              0.65, 0.70, 0.71, 0.75, 0.80, 0.85, 0.90)


def assess_threshold(actual_correct: np.ndarray, top_scores: np.ndarray,
                     threshold: float) -> dict:
    """Summary of a policy that accepts scores >= threshold; otherwise reviews."""
    if actual_correct.shape != top_scores.shape:
        raise ValueError("actual_correct and top_scores must have identical shapes")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be within [0, 1]")
    accepted = top_scores >= threshold
    reviewed = ~accepted
    total = len(accepted)
    accepted_count = int(accepted.sum())
    wrong_accepted = int((accepted & ~actual_correct).sum())
    right_accepted = int((accepted & actual_correct).sum())
    wrong_reviewed = int((reviewed & ~actual_correct).sum())
    right_reviewed = int((reviewed & actual_correct).sum())
    error_count = int((~actual_correct).sum())
    return {
        "threshold": round(float(threshold), 2),
        "validation_samples": total,
        "auto_accepted": accepted_count,
        "sent_for_review": int(reviewed.sum()),
        "review_rate_pct": round(100 * reviewed.sum() / total, 2) if total else None,
        "auto_accepted_correct": right_accepted,
        "auto_accepted_wrong": wrong_accepted,
        "auto_accept_error_rate_pct": round(100 * wrong_accepted / accepted_count, 2)
        if accepted_count else None,
        "errors_caught_by_review": wrong_reviewed,
        "errors_caught_pct": round(100 * wrong_reviewed / error_count, 2)
        if error_count else None,
        "correct_sent_for_review": right_reviewed,
    }


def run() -> None:
    dataset = load_csv(DATA)
    # Exactly mirrors train.py: only the original TRAIN split, NEVER test.csv.
    _, val_texts, _, val_labels = train_test_split(
        dataset["text"].tolist(), dataset["category"].tolist(),
        test_size=0.2, random_state=42,
        stratify=dataset["category"].tolist(),
    )
    model = load_local_model(MODEL)
    predictions = np.asarray(model.predict(val_texts))
    actual = np.asarray(val_labels)
    correct = predictions == actual
    probs = np.asarray(model.predict_proba(val_texts))
    top_scores = probs.max(axis=1)
    accuracy = float(accuracy_score(actual, predictions))

    # Safety checks: fail rather than silently use a different validation split/model.
    if BASELINE_METRICS.is_file():
        baseline = json.loads(BASELINE_METRICS.read_text(encoding="utf-8"))
        expected_accuracy = baseline.get("validation", {}).get("accuracy")
        if expected_accuracy is not None and abs(accuracy - float(expected_accuracy)) > 0.001:
            raise RuntimeError("Model/data/validation split differs from metrics.json. Stop here.")
        expected_n = baseline.get("validation", {}).get("size")
        if expected_n is not None and len(actual) != int(expected_n):
            raise RuntimeError("Validation sample count differs from metrics.json. Stop here.")

    rows = [assess_threshold(correct, top_scores, t) for t in THRESHOLDS]
    report = pd.DataFrame(rows)
    reference = next(row for row in rows if row["threshold"] == 0.4)
    if ERROR_SUMMARY.is_file():
        earlier = json.loads(ERROR_SUMMARY.read_text(encoding="utf-8"))
        expected = earlier.get("incorrect_not_flagged_for_review")
        if expected is not None and reference["auto_accepted_wrong"] != int(expected):
            raise RuntimeError("0.40 error count disagrees with error_analysis/summary.json")
        expected = earlier.get("flagged_for_review")
        if expected is not None and reference["sent_for_review"] != int(expected):
            raise RuntimeError("0.40 review count disagrees with error_analysis/summary.json")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_DIR / "threshold_sweep.csv", index=False, encoding="utf-8-sig")
    summary = {
        "dataset": "BANKING77 original training split, validation subset only",
        "validation_samples": int(len(actual)),
        "validation_accuracy": round(accuracy, 5),
        "validation_errors": int((~correct).sum()),
        "baseline_threshold_0_40": reference,
        "notes": [
            "Scores are not calibrated probabilities; threshold is a demo policy, not a production setting.",
            "Reducing errors by raising a threshold may sharply increase manual review volume.",
            "BANKING77 contains in-domain banking queries: this does not measure out-of-domain detection.",
            "Thresholds were inspected on validation data only; official test data were NOT used.",
            "No API behavior or model artifact was changed by this analysis.",
        ],
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    columns = ["threshold", "auto_accepted", "sent_for_review", "review_rate_pct",
               "auto_accepted_wrong", "auto_accept_error_rate_pct", "errors_caught_pct"]
    print("\n=== VALIDATION THRESHOLD SWEEP ===")
    print(f"Validation: {len(actual)} items, {int((~correct).sum())} wrong; accuracy={accuracy:.5f}")
    print(report[columns].to_string(index=False))
    print("\nATTENTION: uncalibrated scores; values do not establish a production policy.")
    print(f"Files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
