"""TrustDesk AI v0.2 — leakage-conscious, validation-only calibration experiment.

Usage (from project root):
    .\\.venv\\Scripts\\python.exe .\\trustdesk_calibration_experiment.py

NOTE: This experiment creates a NEW model and DOES NOT change the current API.
The validation data has been inspected in previous experiments, so this is an
EXPLORATORY comparison, not an untouched final evaluation. The official
BANKING77 test.csv is not read by this script.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

try:  # scikit-learn 1.6+; works with existing scikit-learn >=1.4 dependency
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover - tested on modern sklearn; legacy fallback
    FrozenEstimator = None

from trustdesk.data import load_csv
from trustdesk.train import build_model

SEED = 42
THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
COVERAGES = (0.10, 0.25, 0.50, 0.75, 1.00)


def split_train_cal_validation(data: pd.DataFrame):
    """Preserve original v0.1 validation, then split ONLY the old fit portion.

    Original BANKING77 train.csv -> 80% development/20% validation.
    Development -> 80% model fit/20% calibration.
    No row indices overlap; stratification protects label coverage.
    """
    all_idx = np.arange(len(data))
    fitcal_idx, val_idx = train_test_split(
        all_idx, test_size=0.20, random_state=SEED,
        stratify=data["category"].to_numpy(),
    )
    fit_idx, cal_idx = train_test_split(
        fitcal_idx, test_size=0.20, random_state=SEED,
        stratify=data.iloc[fitcal_idx]["category"].to_numpy(),
    )
    assert set(fit_idx).isdisjoint(cal_idx)
    assert set(fit_idx).isdisjoint(val_idx)
    assert set(cal_idx).isdisjoint(val_idx)
    assert len(fit_idx) + len(cal_idx) + len(val_idx) == len(data)
    return fit_idx, cal_idx, val_idx


def fit_sigmoid_calibrator(trained_model, X_cal, y_cal):
    """Fit calibration only: do not retrain or refit TF-IDF/LogReg on X_cal."""
    if FrozenEstimator is not None:
        model = CalibratedClassifierCV(FrozenEstimator(trained_model), method="sigmoid", cv=3)
    else:
        warnings.warn("Older scikit-learn: using deprecated cv='prefit'. Upgrade sklearn.", RuntimeWarning)
        model = CalibratedClassifierCV(trained_model, method="sigmoid", cv="prefit")
    model.fit(X_cal, y_cal)
    return model


def top_label_bins(y_true, probabilities, classes, n_bins=10):
    """Top-label reliability bins (NOT the 77-class probability distribution)."""
    y_true = np.asarray(y_true)
    classes = np.asarray(classes)
    predicted_index = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = classes[predicted_index] == y_true
    bin_index = np.minimum(np.floor(confidence * n_bins).astype(int), n_bins - 1)
    rows = []
    total = len(y_true)
    ece = 0.0
    for k in range(n_bins):
        mask = bin_index == k
        count = int(mask.sum())
        observed_accuracy = float(correct[mask].mean()) if count else None
        avg_confidence = float(confidence[mask].mean()) if count else None
        if count:
            ece += (count / total) * abs(observed_accuracy - avg_confidence)
        rows.append({
            "bin_start": round(k / n_bins, 2),
            "bin_end": round((k + 1) / n_bins, 2),
            "count": count,
            "mean_confidence": round(avg_confidence, 6) if count else None,
            "accuracy": round(observed_accuracy, 6) if count else None,
        })
    return rows, float(ece)


def evaluate(y_true, probabilities, classes) -> dict:
    """All metrics calculated on the identical held-out validation rows."""
    y_true = np.asarray(y_true)
    classes = np.asarray(classes)
    if probabilities.shape != (len(y_true), len(classes)):
        raise ValueError("Probabilities/classes have inconsistent shapes")
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid probability distribution")
    prediction = classes[probabilities.argmax(axis=1)]
    _, ece = top_label_bins(y_true, probabilities, classes)
    idx = {str(c): i for i, c in enumerate(classes)}
    true_column = np.array([idx[str(y)] for y in y_true])
    onehot = np.eye(len(classes), dtype=float)[true_column]
    brier = np.mean(np.sum((probabilities - onehot) ** 2, axis=1))
    return {
        "count": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, prediction)), 5),
        "macro_f1": round(float(f1_score(y_true, prediction, average="macro", zero_division=0)), 5),
        "multiclass_log_loss": round(float(log_loss(y_true, probabilities, labels=classes)), 5),
        "multiclass_brier_score": round(float(brier), 5),
        "top_label_ece_10_bins": round(float(ece), 5),
    }


def threshold_sweep(y_true, probabilities, classes, method):
    labels = np.asarray(classes)[probabilities.argmax(axis=1)]
    correct = labels == np.asarray(y_true)
    confidences = probabilities.max(axis=1)
    rows = []
    for threshold in THRESHOLDS:
        accepted = confidences >= threshold
        reviewed = ~accepted
        count = int(accepted.sum())
        incorrect_accepted = int((accepted & ~correct).sum())
        rows.append({
            "model": method,
            "threshold": threshold,
            "auto_accepted": count,
            "reviewed": int(reviewed.sum()),
            "review_rate_pct": round(float(100 * reviewed.mean()), 2),
            "wrong_auto_accepted": incorrect_accepted,
            "wrong_auto_accept_rate_pct": round(100 * incorrect_accepted / count, 2) if count else None,
        })
    return rows


def risk_coverage(y_true, probabilities, classes, method):
    """Compare at SAME accepted volume; better comparison than same numeric cutoff."""
    truth = np.asarray(y_true)
    prediction = np.asarray(classes)[probabilities.argmax(axis=1)]
    scores = probabilities.max(axis=1)
    ranked = np.argsort(-scores, kind="stable")
    rows = []
    for coverage in COVERAGES:
        take = max(1, int(round(len(truth) * coverage)))
        selected = ranked[:take]
        wrong = int((prediction[selected] != truth[selected]).sum())
        rows.append({
            "model": method,
            "target_coverage": coverage,
            "auto_accepted": take,
            "reviewed": len(truth) - take,
            "wrong_auto_accepted": wrong,
            "wrong_auto_accept_rate_pct": round(100 * wrong / take, 2),
        })
    return rows


def save_reliability_svg(bins_by_model, path: Path):
    """Portable SVG reliability diagram, no matplotlib dependency needed."""
    from html import escape

    w, h = 690, 495
    left, bottom, span = 77, 414, 345
    x = lambda value: left + float(value) * span
    y = lambda value: bottom - float(value) * span
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           '<rect width="100%" height="100%" fill="#fff"/>',
           '<text x="77" y="37" font-family="Arial" font-size="19" font-weight="bold">Validation reliability diagram</text>',
           '<text x="77" y="59" font-family="Arial" font-size="12" fill="#666">Top-label confidence vs empirical accuracy (10 bins)</text>']
    for tick in np.linspace(0, 1, 6):
        out.append(f'<line x1="{x(tick):.1f}" y1="{y(0):.1f}" x2="{x(tick):.1f}" y2="{y(1):.1f}" stroke="#eee"/>')
        out.append(f'<line x1="{x(0):.1f}" y1="{y(tick):.1f}" x2="{x(1):.1f}" y2="{y(tick):.1f}" stroke="#eee"/>')
        out.append(f'<text x="{x(tick):.1f}" y="433" text-anchor="middle" font-family="Arial" font-size="11" fill="#555">{tick:.1f}</text>')
        out.append(f'<text x="64" y="{y(tick)+4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{tick:.1f}</text>')
    out.append(f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" stroke="#777" stroke-dasharray="5,5" stroke-width="1.5"/>')
    legend = [("uncalibrated", "#b45309"), ("sigmoid_calibrated", "#1d4ed8")]
    for name, color in legend:
        bins = bins_by_model[name]
        pts = [(b["mean_confidence"], b["accuracy"]) for b in bins if b["count"]]
        if pts:
            polyline = " ".join(f'{x(px):.2f},{y(py):.2f}' for px,py in pts)
            out.append(f'<polyline points="{polyline}" stroke="{color}" stroke-width="2.5" fill="none"/>')
            for px,py in pts:
                out.append(f'<circle cx="{x(px):.2f}" cy="{y(py):.2f}" r="3.5" fill="{color}"/>')
    out.extend(['<text x="240" y="464" font-family="Arial" font-size="12">Mean top-label confidence</text>',
                '<text transform="translate(20 311) rotate(-90)" font-family="Arial" font-size="12">Observed accuracy</text>'])
    for i,(name,color) in enumerate(legend):
        yy=139+34*i
        out.append(f'<line x1="467" y1="{yy}" x2="491" y2="{yy}" stroke="{color}" stroke-width="3"/>')
        out.append(f'<text x="498" y="{yy+4}" font-family="Arial" font-size="13">{escape(name)}</text>')
    out.append('<text x="467" y="215" font-family="Arial" font-size="12" fill="#555">Dashed: ideal calibration</text>')
    out.append('</svg>')
    path.write_text("\n".join(out), encoding="utf-8")


def run(train_csv="data/raw/train.csv", output_dir="artifacts/calibration_experiment"):
    source = Path(train_csv)
    data = load_csv(source)
    fit_idx, cal_idx, val_idx = split_train_cal_validation(data)
    fit = data.iloc[fit_idx]
    cal = data.iloc[cal_idx]
    val = data.iloc[val_idx]
    # Critical: both methods use the exact same fitted base model and validation texts.
    base = build_model()
    base.fit(fit["text"].tolist(), fit["category"].tolist())
    calibrated = fit_sigmoid_calibrator(base, cal["text"].tolist(), cal["category"].tolist())
    classes = np.asarray(base.classes_)
    if not np.array_equal(classes, np.asarray(calibrated.classes_)):
        raise RuntimeError("Calibrated classifier class ordering changed")

    truth = val["category"].to_numpy()
    x_val = val["text"].tolist()
    raw_probs = base.predict_proba(x_val)
    cal_probs = calibrated.predict_proba(x_val)
    baseline_stats = evaluate(truth, raw_probs, classes)
    calibration_stats = evaluate(truth, cal_probs, classes)
    raw_bins,_ = top_label_bins(truth, raw_probs, classes)
    cal_bins,_ = top_label_bins(truth, cal_probs, classes)

    outputs = Path(output_dir)
    outputs.mkdir(parents=True, exist_ok=True)
    # New artifact, never overwrite the existing artifacts/baseline.joblib.
    joblib.dump(calibrated, outputs / "calibrated_sigmoid_experimental.joblib")
    joblib.dump(base, outputs / "same_fit_uncalibrated.joblib")
    pd.DataFrame([{"model": "uncalibrated", **row} for row in raw_bins] +
                 [{"model": "sigmoid_calibrated", **row} for row in cal_bins]).to_csv(
                     outputs / "reliability_bins.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(threshold_sweep(truth, raw_probs, classes, "uncalibrated")+
                 threshold_sweep(truth, cal_probs, classes, "sigmoid_calibrated")).to_csv(
                     outputs / "threshold_sweep.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(risk_coverage(truth, raw_probs, classes, "uncalibrated")+
                 risk_coverage(truth, cal_probs, classes, "sigmoid_calibrated")).to_csv(
                     outputs / "risk_coverage.csv", index=False, encoding="utf-8-sig")
    save_reliability_svg({"uncalibrated": raw_bins, "sigmoid_calibrated": cal_bins},
                         outputs / "reliability_diagram.svg")
    summary = {
        "purpose": "Exploratory in-domain sigmoid calibration evaluation (not production-ready)",
        "seed": SEED,
        "input_csv": str(source),
        "rows": {"total": len(data), "model_fit": len(fit), "calibration": len(cal), "validation": len(val)},
        "uncalibrated_same_fit": baseline_stats,
        "sigmoid_calibrated": calibration_stats,
        "data_leakage_policy": "Fit/cal/validation rows disjoint. Official test.csv NOT read.",
        "important_warnings": [
            "Validation data were already inspected in previous threshold experiments; results are exploratory.",
            "Existing baseline.joblib was fit on 80% of original training split; this paired experiment uses 64% for BOTH compared models. Do not mistake a cross-sample comparison for calibration's causal effect.",
            "Calibration on BANKING77 in-domain data does not establish out-of-domain rejection, safety, or real-world accuracy.",
            "Do not point the existing API at this calibrated artifact yet: its score_note currently says uncalibrated, and the existing 0.40 threshold should not be reused unchanged.",
            "Untrusted joblib files can execute code; load only local artifacts that you generated yourself.",
        ],
    }
    (outputs / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== CALIBRATION EXPERIMENT (VALIDATION ONLY) ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nArtifacts saved to: {outputs.resolve()}")
    return summary


def main():
    p = argparse.ArgumentParser(description="Reproducible validation-only calibration experiment")
    p.add_argument("--train-csv", default="data/raw/train.csv")
    p.add_argument("--output-dir", default="artifacts/calibration_experiment")
    args = p.parse_args()
    run(args.train_csv, args.output_dir)


if __name__ == "__main__":
    main()
