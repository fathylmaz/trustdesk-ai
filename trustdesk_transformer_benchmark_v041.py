"""TrustDesk AI v0.4.1: same-split BANKING77 DistilBERT vs TF-IDF benchmark.

Experimental only: uses repeatedly inspected validation split, never reads test.csv.
Use solely with the pretrained model ID and locally generated baseline artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, top_k_accuracy_score

from trustdesk.data import load_csv
from trustdesk_calibration_experiment import SEED, split_train_cal_validation

DEFAULT_MODEL = "distilbert/distilbert-base-uncased"


def label_mapping(labels: list[str]) -> tuple[dict[int, str], dict[str, int]]:
    """Stable label IDs independent from accidental pandas input ordering."""
    unique = sorted(set(labels))
    if len(unique) != len(labels) or not unique:
        raise ValueError("labels must be nonempty, unique class names")
    return {i: name for i, name in enumerate(unique)}, {name: i for i, name in enumerate(unique)}


def evaluate_probs(y_true, probabilities, classes) -> dict:
    """Classification quality; no deployment threshold or OOD inference."""
    truth = np.asarray(y_true)
    classes = np.asarray(classes)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(truth), len(classes)):
        raise ValueError("probability matrix shape mismatch")
    if len(truth) == 0 or len(classes) < 3:
        raise ValueError("expected nonempty validation and at least 3 classes")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("invalid probability rows")
    # PyTorch softmax returns float32; small rounding errors (around 1e-7)
    # can trigger scikit-learn's stricter float64 log_loss normalization warning.
    # Re-normalize in float64 ONLY for evaluation, preserving class ranking.
    probabilities = probabilities / row_sums
    predicted = classes[np.argmax(probabilities, axis=1)]
    top3 = float(top_k_accuracy_score(truth, probabilities, labels=classes, k=3)) if len(classes) > 3 else 1.0
    return {
        "count": int(len(truth)),
        "accuracy": round(float(accuracy_score(truth, predicted)), 5),
        "macro_f1": round(float(f1_score(truth, predicted, labels=classes, average="macro", zero_division=0)), 5),
        "top3_accuracy": round(top3, 5),
        "multiclass_log_loss": round(float(log_loss(truth, probabilities, labels=classes)), 5),
        "wrong_count": int((predicted != truth).sum()),
    }


def prepare_data(csv_file: str):
    data = load_csv(csv_file)
    fit_idx, cal_idx, val_idx = split_train_cal_validation(data)
    fit = data.iloc[fit_idx].copy()
    val = data.iloc[val_idx].copy()
    classes = sorted(data.category.unique().tolist())
    # Exact v0.2 split; separate calibration rows are deliberately not used.
    if (len(fit), len(cal_idx), len(val)) != (6401, 1601, 2001) and len(data) == 10003:
        raise RuntimeError("Unexpected original BANKING77 split sizes")
    return data, fit, val, cal_idx, classes


def check_gpu():
    try:
        import torch
    except ImportError:
        return {"torch_installed": False, "cuda_available": False, "instruction": "Install CUDA-enabled PyTorch from https://pytorch.org/get-started/locally/"}
    available = bool(torch.cuda.is_available())
    result = {"torch_installed": True, "torch_version": torch.__version__,
              "cuda_available": available, "compiled_cuda": torch.version.cuda}
    if available:
        result["gpu_name"] = torch.cuda.get_device_name(0)
        result["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    return result


def make_dataset(tokenizer, frame, label2id, max_length):
    import torch
    encoded = tokenizer(frame.text.tolist(), truncation=True, padding="max_length",
                        max_length=max_length, return_tensors="pt")
    ids = torch.tensor([label2id[x] for x in frame.category.tolist()], dtype=torch.long)

    class EncodedDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(ids)
        def __getitem__(self, index):
            return {**{k: v[index] for k, v in encoded.items()}, "labels": ids[index]}

    return EncodedDataset()


def predict_probabilities(model, loader, device):
    import torch
    chunks = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            scores = model(**inputs).logits
            chunks.append(torch.softmax(scores.float(), dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def run(args):
    # Import transformers only when the training step is requested: --check-only needs no model downloads.
    random.seed(SEED)
    np.random.seed(SEED)
    data, fit, val, cal_idx, classes = prepare_data(args.train_csv)
    id2label, label2id = label_mapping(classes)
    baseline_path = Path(args.baseline)
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Same-fit v0.2 baseline not found: {baseline_path}. First run v0.2 calibration experiment.")
    # Trust only the joblib artifacts generated on your own computer.
    baseline = joblib.load(baseline_path)
    if list(baseline.classes_) != classes:
        raise RuntimeError("Same-fit baseline class ordering differs from current training CSV")
    val_texts = val.text.tolist()
    truth = val.category.tolist()
    baseline_probs = baseline.predict_proba(val_texts)
    baseline_metrics = evaluate_probs(truth, baseline_probs, classes)
    info = {
        "status": "checks_only" if args.check_only else "experiment_complete",
        "purpose": "Exploratory in-domain DistilBERT vs same-fit TF-IDF benchmark",
        "seed": SEED,
        "data_sha256": hashlib.sha256(Path(args.train_csv).read_bytes()).hexdigest(),
        "rows": {"fit": len(fit), "calibration_unused": len(cal_idx), "validation": len(val)},
        "classes": len(classes),
        "baseline_same_fit": baseline_metrics,
        "device_information": check_gpu(),
        "official_test_csv_read": False,
        "caveats": [
            "Validation was examined during previous v0.1-v0.3 experiments: exploratory, not independent final assessment.",
            "Both models share the 6,401-row fit split, but DistilBERT uses external self-supervised pretraining.",
            "The 1,601 calibration rows and official test.csv were not used.",
            "Neither classifier is independently validated for OOD, banking safety, or real customer data.",
            "Validation evaluations each epoch are exploratory and should not be treated as a pristine test.",
        ],
    }
    if args.check_only:
        print("\n=== TRANSFORMER BENCHMARK PRECHECK ===")
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return info

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as exc:
        raise RuntimeError("Missing PyTorch/Transformers. Read KURULUM_VE_ACIKLAMALAR.md first.") from exc
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch. Do NOT start slow CPU training unintentionally. Check torch installation and nvidia-smi. To explicitly train on CPU use --device cpu.")
    device = torch.device(args.device)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=False, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        DEFAULT_MODEL, num_labels=len(classes), id2label=id2label, label2id=label2id,
        trust_remote_code=False, use_safetensors=True,
    ).to(device)
    train_loader = DataLoader(make_dataset(tokenizer, fit, label2id, args.max_length),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(make_dataset(tokenizer, val, label2id, args.max_length),
                            batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    info["configuration"] = {
        "model_id": DEFAULT_MODEL, "epochs": args.epochs, "batch_size": args.batch_size,
        "gradient_accumulation": args.grad_accum, "max_length": args.max_length,
        "learning_rate": args.learning_rate, "device": str(device),
        "torch_version": torch.__version__, "transformers_version": __import__("transformers").__version__,
    }
    info["epoch_history"] = []
    start_time = time.time()
    print(f"Training on {device}. Epochs={args.epochs}; micro_batch={args.batch_size}; grad_accum={args.grad_accum}", flush=True)
    print("DistilBERT classification head is newly initialized; that warning is EXPECTED.", flush=True)
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        n_batches = len(train_loader)
        for j, batch in enumerate(train_loader):
            remainder_in_group = n_batches - (j // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, remainder_in_group)
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            (loss / group_size).backward()
            loss_total += float(loss.detach().item())
            if (j + 1) % args.grad_accum == 0 or j + 1 == n_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if (j + 1) % args.log_every == 0 or j + 1 == n_batches:
                print(f"Epoch {epoch+1}/{args.epochs} batch {j+1}/{n_batches} loss={loss_total/(j+1):.4f}", flush=True)
        probs = predict_probabilities(model, val_loader, device)
        score = evaluate_probs(truth, probs, classes)
        info["epoch_history"].append({"epoch": epoch + 1, "train_mean_loss": round(loss_total / n_batches, 5), **score})
        print(f"End of epoch {epoch+1}: validation accuracy={score['accuracy']}, macro_f1={score['macro_f1']}", flush=True)

    # The last epoch is reported; no best-on-validation checkpoint selection.
    info["transformer_last_epoch"] = info["epoch_history"][-1]
    info["elapsed_seconds"] = round(time.time() - start_time, 1)
    output = Path(args.output_dir)
    model_folder = output / "distilbert_model"
    model_folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_folder, safe_serialization=True)
    tokenizer.save_pretrained(model_folder)
    preds = np.asarray(classes)[probs.argmax(axis=1)]
    errors = pd.DataFrame({
        "text": val_texts,
        "actual_label": truth,
        "baseline_predicted": np.asarray(classes)[baseline_probs.argmax(axis=1)],
        "transformer_predicted": preds,
        "transformer_score_uncalibrated": probs.max(axis=1).round(6),
        "baseline_score_uncalibrated": baseline_probs.max(axis=1).round(6),
    })
    errors["transformer_correct"] = errors.actual_label == errors.transformer_predicted
    errors["baseline_correct"] = errors.actual_label == errors.baseline_predicted
    errors.to_csv(output / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    errors.loc[~errors.transformer_correct].to_csv(output / "transformer_errors.csv", index=False, encoding="utf-8-sig")
    (output / "benchmark_summary.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== TRANSFORMER BENCHMARK (EXPLORATORY VALIDATION ONLY) ===")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"\nSaved outputs: {output.resolve()}")
    return info


def main():
    p = argparse.ArgumentParser(description="v0.4 BANKING77 DistilBERT controlled baseline benchmark")
    p.add_argument("--train-csv", default="data/raw/train.csv")
    p.add_argument("--baseline", default="artifacts/calibration_experiment/same_fit_uncalibrated.joblib")
    p.add_argument("--output-dir", default="artifacts/transformer_benchmark")
    p.add_argument("--check-only", action="store_true", help="Check split, baseline, torch GPU detection; no model download")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-length", type=int, default=96)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--log-every", type=int, default=100)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.grad_accum < 1 or args.max_length < 8 or args.learning_rate <= 0 or args.log_every < 1:
        p.error("All training arguments must be positive; max-length must be at least 8")
    try:
        run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
