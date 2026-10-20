r"""TrustDesk AI: reproducible validation-set error analysis for the v0.1 baseline.

Place in the root of trustdesk-ai-starter and run from that directory:
    .\.venv\Scripts\python.exe .\trustdesk_error_analysis.py

Does not retrain the model or modify the existing model artifact.
Use validation errors for development decisions; do not tune against the official test set.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from trustdesk.data import load_csv
from trustdesk.inference import load_local_model

TRAIN_CSV = Path("data/raw/train.csv")
MODEL_PATH = Path("artifacts/baseline.joblib")
OUTPUT_DIR = Path("artifacts/error_analysis")
REVIEW_THRESHOLD = 0.40  # Existing demo threshold; uncalibrated. Not a production cutoff.


def run_analysis() -> None:
    data = load_csv(TRAIN_CSV)
    # EXACT same split as the v0.1 training module: critical for unbiased validation.
    _, val_texts, _, val_labels = train_test_split(
        data["text"].tolist(),
        data["category"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=data["category"].tolist(),
    )
    model = load_local_model(MODEL_PATH)
    classes = np.asarray(model.classes_)
    scores = model.predict_proba(val_texts)
    ranked_idx = np.argsort(scores, axis=1)[:, ::-1]
    best_idx = ranked_idx[:, 0]
    second_idx = ranked_idx[:, 1] if len(classes) > 1 else best_idx
    predicted = classes[best_idx]
    top_scores = scores[np.arange(len(scores)), best_idx]
    second_scores = scores[np.arange(len(scores)), second_idx]
    actual = np.asarray(val_labels)
    is_wrong = predicted != actual
    needs_review = top_scores < REVIEW_THRESHOLD

    accuracy = float(accuracy_score(actual, predicted))
    macro_f1 = float(f1_score(actual, predicted, average="macro", zero_division=0))
    # Cross-check that reconstructed validation is the actual trained-model split.
    reported_path = Path("artifacts/metrics.json")
    if reported_path.exists():
        previous = json.loads(reported_path.read_text(encoding="utf-8"))
        target = previous.get("validation", {}).get("accuracy")
        if target is not None and abs(accuracy - float(target)) > 0.001:
            raise RuntimeError(
                "Validation split does not match the trained model/metrics.json. "
                "Stop and check whether the model, data, or train.py has changed."
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    details = pd.DataFrame({
        "text": val_texts,
        "actual_label": actual,
        "predicted_label": predicted,
        "top_score_uncalibrated": np.round(top_scores, 5),
        "second_label": classes[second_idx],
        "second_score_uncalibrated": np.round(second_scores, 5),
        "needs_human_review_at_0_40": needs_review,
        "is_correct": ~is_wrong,
    })
    errors = details.loc[is_wrong].copy()
    errors.sort_values("top_score_uncalibrated", ascending=False, inplace=True)
    errors.to_csv(OUTPUT_DIR / "validation_errors.csv", index=False, encoding="utf-8-sig")
    # These 20 errors are high-score mistakes: especially useful for manual diagnosis.
    review_20 = errors.head(20).copy()
    review_20.insert(len(review_20.columns), "manual_error_reason", "")
    review_20.to_csv(OUTPUT_DIR / "review_20_errors.csv", index=False, encoding="utf-8-sig")

    confusions = (
        errors.groupby(["actual_label", "predicted_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["count", "actual_label", "predicted_label"], ascending=[False, True, True])
    )
    confusions.to_csv(OUTPUT_DIR / "top_confusions.csv", index=False, encoding="utf-8-sig")

    by_class = classification_report(actual, predicted, zero_division=0, output_dict=True)
    class_rows = [
        {"label": label, **by_class[label]}
        for label in sorted(set(actual))
    ]
    class_report = pd.DataFrame(class_rows).sort_values(["f1-score", "label"])
    class_report.to_csv(OUTPUT_DIR / "per_class_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "dataset": "BANKING77 training split: validation subset ONLY",
        "validation_samples": int(len(actual)),
        "correct": int((~is_wrong).sum()),
        "incorrect": int(is_wrong.sum()),
        "accuracy": round(accuracy, 5),
        "macro_f1": round(macro_f1, 5),
        "review_threshold_uncalibrated": REVIEW_THRESHOLD,
        "flagged_for_review": int(needs_review.sum()),
        "incorrect_not_flagged_for_review": int((is_wrong & ~needs_review).sum()),
        "correct_flagged_for_review": int((~is_wrong & needs_review).sum()),
        "warning": "Score is uncalibrated. This analysis does not establish reliable OOD detection or production routing thresholds.",
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== VALIDATION ERROR ANALYSIS ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nTop 10 confused label pairs:")
    print(confusions.head(10).to_string(index=False))
    print(f"\nOutputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    run_analysis()
