"""Tests for calibration experiment; tiny synthetic data, not BANKING77 results."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from trustdesk_calibration_experiment import (
    split_train_cal_validation,
    fit_sigmoid_calibrator,
    top_label_bins,
    evaluate,
    run,
)
from trustdesk.train import build_model


def demo_data():
    samples = {
        "card_stolen": ["my card got stolen", "someone stole my payment card", "report missing stolen card"],
        "change_pin": ["reset card pin", "change my pin code", "forgot pin number"],
        "refund": ["refund hasn't arrived", "refund still missing", "where is my refund"],
    }
    return pd.DataFrame([{"text": f"{s} message example {i}", "category": label}
                         for label, phrases in samples.items() for s in phrases for i in range(6)])


def test_data_split_no_leak_and_reproducible():
    data = demo_data()
    a = split_train_cal_validation(data)
    b = split_train_cal_validation(data)
    assert all(np.array_equal(i, j) for i,j in zip(a,b))
    assert len(set(a[0]) | set(a[1]) | set(a[2])) == len(data)
    assert set(a[0]).isdisjoint(a[1])
    assert set(a[0]).isdisjoint(a[2])
    assert set(a[1]).isdisjoint(a[2])


def test_calibration_probabilities_are_valid_and_metrics_finite():
    df = demo_data()
    fit, cal, val = split_train_cal_validation(df)
    x_fit = df.iloc[fit]["text"].tolist()
    y_fit = df.iloc[fit]["category"].tolist()
    base = build_model()
    base.fit(x_fit, y_fit)
    frozen_before = base.predict_proba(df.iloc[val]["text"].tolist()).copy()
    model = fit_sigmoid_calibrator(base, df.iloc[cal]["text"].tolist(), df.iloc[cal]["category"].tolist())
    frozen_after = base.predict_proba(df.iloc[val]["text"].tolist())
    assert np.allclose(frozen_before, frozen_after), "Calibration should not refit base model"
    scores = model.predict_proba(df.iloc[val]["text"].tolist())
    assert scores.shape == (len(val), len(base.classes_))
    assert np.allclose(scores.sum(axis=1), 1)
    assert np.all((scores >= 0) & (scores <= 1))
    result = evaluate(df.iloc[val]["category"].tolist(), scores, model.classes_)
    assert 0 <= result["accuracy"] <= 1
    assert 0 <= result["top_label_ece_10_bins"] <= 1
    assert result["multiclass_brier_score"] >= 0


def test_reliability_bins_include_confidence_one():
    bins,ece = top_label_bins(["A", "B"], np.array([[1.,0.],[0.2,0.8]]), ["A", "B"])
    assert sum(row["count"] for row in bins) == 2
    assert bins[-1]["count"] == 1
    assert ece == pytest.approx(0.1)


def test_complete_run_outputs(tmp_path: Path):
    source = tmp_path / "train.csv"
    demo_data().to_csv(source, index=False)
    output = tmp_path / "results"
    summary = run(source, output)
    assert summary["rows"]["total"] == 54
    assert sum(summary["rows"][key] for key in ("model_fit","calibration","validation")) == 54
    for name in ("calibrated_sigmoid_experimental.joblib", "same_fit_uncalibrated.joblib",
                 "summary.json", "reliability_diagram.svg", "reliability_bins.csv",
                 "threshold_sweep.csv", "risk_coverage.csv"):
        assert (output / name).exists(), name
