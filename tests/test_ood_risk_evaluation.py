"""No model artifact/dataset download required for these unit tests."""
from pathlib import Path
import numpy as np
import pandas as pd
from trustdesk_ood_risk_evaluation import (
    reject_audit, ood_discrimination, model_scores, risk_audit,
)
from trustdesk_risk_rules import screen_risk


class FakeModel:
    def predict_proba(self, texts):
        vals = {"bank": [0.9, 0.1], "other": [0.3, 0.7]}
        return np.array([vals[t] for t in texts])


def test_model_scores():
    got = model_scores(FakeModel(), ["bank", "other"])
    assert np.allclose(got, [0.9, 0.7])


def test_ood_threshold_exact():
    sweeps = reject_audit(np.array([0.8, 0.4]), np.array([0.1, 0.7]), thresholds=[0.5])
    assert sweeps[0]["id_false_reject_count"] == 1
    assert sweeps[0]["ood_detected_count"] == 1
    assert sweeps[0]["ood_missed_count"] == 1


def test_ood_auc_perfect_ranking():
    scores = ood_discrimination(np.array([0.8, 0.9]), np.array([0.1, 0.2]))
    assert scores["ood_positive_auroc"] == 1.0


def test_risk_rules_flag_and_negative():
    assert screen_risk("My card was stolen yesterday.")["flagged"]
    assert screen_risk("Where is my monthly account statement?")["flagged"] is False


def test_rule_audit_counts():
    frame = pd.DataFrame([
        {"scenario": "risk", "text": "My card was stolen.", "risk_expected": "true"},
        {"scenario": "nonrisk", "text": "Can I order a physical card?", "risk_expected": "false"},
        {"scenario": "edge", "text": "An intruder got my money.", "risk_expected": "true"},
    ])
    stats, rows = risk_audit(frame)
    assert (stats["true_positive"], stats["true_negative"], stats["false_negative"]) == (1, 1, 1)
    assert len(rows) == 3


def test_empty_invalid_scores_rejected():
    import pytest
    with pytest.raises(ValueError):
        reject_audit(np.array([float('nan')]), np.array([0.5]))
