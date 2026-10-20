"""Train baseline and export reproducible validation/test evaluations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from trustdesk.data import load_csv


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=30000)),
        ("clf", LogisticRegression(max_iter=1500, random_state=42)),
    ])


def metrics(model: Pipeline, X, y) -> dict:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    classes = model.classes_
    k = min(3, len(classes))
    return {
        "size": len(y),
        "accuracy": round(float(accuracy_score(y, predictions)), 5),
        "macro_f1": round(float(f1_score(y, predictions, average="macro", zero_division=0)), 5),
        "weighted_f1": round(float(f1_score(y, predictions, average="weighted", zero_division=0)), 5),
        "top3_accuracy": round(float(top_k_accuracy_score(y, probabilities, labels=classes, k=k)), 5) if k < len(classes) else 1.0,
    }


def train_and_evaluate(train_csv="data/raw/train.csv", test_csv="data/raw/test.csv", output_dir="artifacts") -> dict:
    train = load_csv(train_csv)
    test = load_csv(test_csv)
    if not set(test.category.unique()).issubset(set(train.category.unique())):
        raise ValueError("Test contains labels not found in train")
    X_train, X_val, y_train, y_val = train_test_split(
        train.text.tolist(), train.category.tolist(),
        test_size=0.2, random_state=42, stratify=train.category.tolist()
    )
    model = build_model()
    model.fit(X_train, y_train)
    summary = {
        "model": "TF-IDF (1,2) + LogisticRegression",
        "seed": 42,
        "train_rows_used": len(X_train),
        "validation_rows": len(X_val),
        "official_test_rows": len(test),
        "class_count": len(model.classes_),
        "validation": metrics(model, X_val, y_val),
        "official_test": metrics(model, test.text.tolist(), test.category.tolist()),
        "caveat": "Scores are dataset-specific. Probability calibration and OOD evaluation have not yet been done.",
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "baseline.joblib")
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report = classification_report(test.category.tolist(), model.predict(test.text.tolist()), zero_division=0)
    (output / "classification_report.txt").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved trained model and reports to: {output.resolve()}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BANKING77 baseline")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    train_and_evaluate(args.train_csv, args.test_csv, args.output_dir)


if __name__ == "__main__":
    main()
