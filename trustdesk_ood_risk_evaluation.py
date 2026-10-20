"""TrustDesk AI v0.3: exploratory OOD proxy + separate high-priority rule audit.

Run from project root with BANKING77 train.csv and prior v0.2 local model files.
Does not import/read BANKING77 test.csv, overwrite artifacts or change API.
OOD examples are authored synthetic probes; NOT a representative benchmark.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from trustdesk.data import load_csv
from trustdesk_calibration_experiment import split_train_cal_validation
from trustdesk_risk_rules import screen_risk

SCORE_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
MODEL_PATHS = {
    "uncalibrated_same_fit": "artifacts/calibration_experiment/same_fit_uncalibrated.joblib",
    "sigmoid_calibrated": "artifacts/calibration_experiment/calibrated_sigmoid_experimental.joblib",
}


def model_scores(model, texts: list[str]) -> np.ndarray:
    """Return max probability, after checking output shape/finite/probability sums."""
    probabilities = np.asarray(model.predict_proba(texts), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(texts) or probabilities.shape[1] < 2:
        raise ValueError("Invalid predict_proba shape")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Invalid model probability output")
    return probabilities.max(axis=1)


def reject_audit(id_conf: np.ndarray, ood_conf: np.ndarray, thresholds=SCORE_THRESHOLDS) -> list[dict]:
    """Threshold T: max-score < T is *flagged*; no production acceptance implied."""
    a, b = np.asarray(id_conf, dtype=float), np.asarray(ood_conf, dtype=float)
    if not len(a) or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("ID and OOD scores must be nonempty and finite")
    rows = []
    for threshold in thresholds:
        id_flag = a < threshold
        ood_flag = b < threshold
        rows.append({
            "threshold": float(threshold),
            "id_count": int(len(a)),
            "ood_count": int(len(b)),
            "id_false_reject_count": int(id_flag.sum()),
            "id_false_reject_pct": round(float(100 * id_flag.mean()), 2),
            "ood_detected_count": int(ood_flag.sum()),
            "ood_detection_recall_pct": round(float(100 * ood_flag.mean()), 2),
            "ood_missed_count": int((~ood_flag).sum()),
            "ood_miss_pct": round(float(100 * (~ood_flag).mean()), 2),
        })
    return rows


def ood_discrimination(id_conf, ood_conf) -> dict:
    """OOD-positive AUROC/AP; AP depends on authored  OOD prevalence."""
    conf = np.r_[id_conf, ood_conf]
    truth = np.r_[np.zeros(len(id_conf), dtype=int), np.ones(len(ood_conf), dtype=int)]
    anomaly_score = -conf
    return {
        "ood_positive_auroc": round(float(roc_auc_score(truth, anomaly_score)), 5),
        "ood_positive_average_precision": round(float(average_precision_score(truth, anomaly_score)), 5),
        "evaluation_ood_fraction_pct": round(100 * len(ood_conf) / len(conf), 2),
    }


def risk_audit(risk_data: pd.DataFrame):
    required = {"text", "risk_expected", "scenario"}
    if not required.issubset(risk_data.columns):
        raise ValueError(f"Risk CSV needs {required}")
    if risk_data.empty or risk_data["text"].isna().any():
        raise ValueError("Empty or invalid risk input")
    labels = risk_data["risk_expected"].astype(str).str.strip().str.lower()
    if not labels.isin({"true", "false", "1", "0"}).all():
        raise ValueError("risk_expected must contain true/false")
    expected = labels.isin({"true", "1"}).to_numpy()
    screenings = [screen_risk(text) for text in risk_data["text"].astype(str)]
    predicted = np.asarray([r["flagged"] for r in screenings], dtype=bool)
    tp = int(np.sum(predicted & expected))
    fp = int(np.sum(predicted & ~expected))
    fn = int(np.sum(~predicted & expected))
    tn = int(np.sum(~predicted & ~expected))
    details = risk_data[["text", "scenario"]].copy()
    details["risk_expected"] = expected
    details["rule_flagged"] = predicted
    details["matched_rules"] = [";".join(r["matched_rules"]) for r in screenings]
    details["correct"] = predicted == expected
    summary = {
        "curated_examples": int(len(predicted)),
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "recall_pct": round(100 * tp / (tp+fn), 2) if tp+fn else None,
        "precision_pct": round(100 * tp / (tp+fp), 2) if tp+fp else None,
        "notice": "Keyword rules are illustrative only, not a safety-validated fraud or security detector.",
    }
    return summary, details


def run(
    train_csv="data/raw/train.csv",
    ood_csv="data/evaluation/ood_curated_v1.csv",
    risk_csv="data/evaluation/risk_curated_v1.csv",
    model_paths=None,
    output_dir="artifacts/ood_risk_evaluation",
):
    models = model_paths if model_paths is not None else MODEL_PATHS
    # The provided v0.2 split: do not inspect or read official test.csv.
    train = load_csv(train_csv)
    _fit_idx, _cal_idx, val_idx = split_train_cal_validation(train)
    validation = train.iloc[val_idx]
    probes = pd.read_csv(ood_csv)
    if not {"text", "probe_group"}.issubset(probes.columns) or probes.empty or probes["text"].isna().any():
        raise ValueError("OOD file must have nonempty text and probe_group columns")
    if probes["text"].astype(str).str.strip().eq("").any():
        raise ValueError("Empty OOD example")
    # Make it harder to accidentally measure samples copied from the original train file.
    known_train = set(train["text"].astype(str).str.strip().str.casefold())
    collisions = [text for text in probes["text"].astype(str) if text.strip().casefold() in known_train]
    if collisions:
        raise ValueError(f"OOD probes overlap training CSV text: {collisions[:3]}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_summaries = {}
    all_threshold_rows = []
    all_ood_rows = []
    for name, filename in models.items():
        # Only load LOCAL model artifacts created by your own experiment.
        model = joblib.load(filename)
        id_scores = model_scores(model, validation["text"].tolist())
        ood_scores = model_scores(model, probes["text"].astype(str).tolist())
        discrimination = ood_discrimination(id_scores, ood_scores)
        sweep = reject_audit(id_scores, ood_scores)
        for row in sweep:
            all_threshold_rows.append({"model": name, **row})
        for (_, p), confidence in zip(probes.iterrows(), ood_scores):
            all_ood_rows.append({"model": name, "probe_group": str(p["probe_group"]),
                                 "text": str(p["text"]), "max_probability": round(float(confidence), 6)})
        model_summaries[name] = {
            "id_validation_count": int(len(id_scores)), "authored_ood_count": int(len(ood_scores)),
            **discrimination,
            "mean_id_max_probability": round(float(id_scores.mean()), 5),
            "mean_ood_max_probability": round(float(ood_scores.mean()), 5),
        }

    risk_table = pd.read_csv(risk_csv, keep_default_na=False)
    risk_summary, risk_details = risk_audit(risk_table)
    summary = {
        "purpose": "Exploratory OOD proxy and separate keyword-risk audit, NOT a production policy",
        "sources": {
            "id": "Previously inspected BANKING77 training split (v0.2 2001-row validation)",
            "ood": "Manually authored synthetic English non-banking probes, not independent benchmark",
            "risk": "Manually authored synthetic English risk/non-risk examples",
            "official_test_csv_read": False,
        },
        "ood_model_comparison": model_summaries,
        "risk_rule_audit": risk_summary,
        "limitations": [
            "OOD is NOT a learned class; max-probability score is just a weak proxy for novelty.",
            "Authors chose OOD examples; selection biases and lexical shortcuts can inflate apparent performance.",
            "AP/precision depend strongly on OOD prevalence, here unlike a real service.",
            "The BANKING77 validation data have been repeatedly inspected; results are exploratory only.",
            "Calibrated class probabilities do not imply reliable OOD detection.",
            "Security-sensitive requests require independently reviewed business rules and human oversight.",
            "Risk rules can be fooled by synonyms, misspellings, negation, other languages and adversarial wording.",
            "Do NOT use either experiment to authorize transfers, freeze cards or approve/refuse real bank requests.",
        ],
    }
    pd.DataFrame(all_threshold_rows).to_csv(output / "threshold_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_ood_rows).to_csv(output / "ood_probe_scores.csv", index=False, encoding="utf-8-sig")
    risk_details.to_csv(output / "risk_rule_predictions.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== OOD & RISK EXPLORATORY EVALUATION ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nThreshold sweep (higher OOD detected is good; higher ID false rejection is costly):")
    print(pd.DataFrame(all_threshold_rows).to_string(index=False))
    print(f"\nOutputs saved to: {output.resolve()}")
    return summary


def main():
    p = argparse.ArgumentParser(description="v0.3 OOD & risk audit (offline; no API changes)")
    p.add_argument("--train-csv", default="data/raw/train.csv")
    p.add_argument("--ood-csv", default="data/evaluation/ood_curated_v1.csv")
    p.add_argument("--risk-csv", default="data/evaluation/risk_curated_v1.csv")
    p.add_argument("--output-dir", default="artifacts/ood_risk_evaluation")
    args = p.parse_args()
    run(args.train_csv, args.ood_csv, args.risk_csv, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
