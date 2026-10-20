"""Check required files and Hugging Face DistilBERT class mapping without loading models.

Run from anywhere: python scripts/check_deploy_inputs.py
Requires only Python's standard library; never deserializes joblib or model weights.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CLASSES = 77
REQUIRED_FILES = (
    "pyproject.toml",
    "README.md",
    "src/trustdesk/inference.py",
    "trustdesk_unified_api_v06.py",
    "trustdesk_transformer_api_v05.py",
    "frontend/package.json",
    "frontend/package-lock.json",
    "frontend/src/App.jsx",
    "artifacts/baseline.joblib",
    "artifacts/transformer_benchmark_4epochs/distilbert_model/config.json",
    "artifacts/transformer_benchmark_4epochs/distilbert_model/model.safetensors",
    "artifacts/transformer_benchmark_4epochs/distilbert_model/tokenizer.json",
)
CONFIG_REL = "artifacts/transformer_benchmark_4epochs/distilbert_model/config.json"


def inspect_class_mapping(config: object) -> tuple[bool, str]:
    """Check label mapping and optional num_labels with useful diagnostics."""
    if not isinstance(config, dict):
        return False, "config.json JSON nesnesi (object) olmali."

    id2label = config.get("id2label")
    label2id = config.get("label2id")
    declared = config.get("num_labels")

    # Saved Hugging Face configs commonly omit num_labels. In that case, the
    # number of classes is derived from the id2label mapping.
    if declared is not None and (type(declared) is not int or declared != EXPECTED_CLASSES):
        return False, f"num_labels alani beklenenden farkli: {declared!r} (beklenen: {EXPECTED_CLASSES})."
    if not isinstance(id2label, dict) or len(id2label) != EXPECTED_CLASSES:
        found = len(id2label) if isinstance(id2label, dict) else "yok"
        return False, f"id2label esleme sayisi: {found}; beklenen: {EXPECTED_CLASSES}."
    if not isinstance(label2id, dict) or len(label2id) != EXPECTED_CLASSES:
        found = len(label2id) if isinstance(label2id, dict) else "yok"
        return False, f"label2id esleme sayisi: {found}; beklenen: {EXPECTED_CLASSES}."

    expected_ids = {str(i) for i in range(EXPECTED_CLASSES)}
    if set(id2label) != expected_ids:
        return False, "id2label indeksleri 0..76 araligiyla eslesmiyor."

    labels = list(id2label.values())
    if not all(isinstance(label, str) and label for label in labels) or len(set(labels)) != EXPECTED_CLASSES:
        return False, "id2label etiketleri benzersiz ve bos olmayan dizeler olmali."

    if set(label2id) != set(labels):
        return False, "id2label ve label2id etiketleri tutarsiz."
    for idx_text, label in id2label.items():
        mapped_idx = label2id.get(label)
        if type(mapped_idx) is not int or mapped_idx != int(idx_text):
            return False, f"Etiket-indeks tutarsizligi: {label!r}."

    suffix = "num_labels JSON icinde yok; etiket eslemesinden dogrulandi" if declared is None else "num_labels alani da uyumlu"
    return True, f"{EXPECTED_CLASSES} sinif dogrulandi ({suffix})."


def main(root: Path | None = None) -> int:
    root = ROOT if root is None else root
    absent = [path for path in REQUIRED_FILES if not (root / path).is_file()]
    if absent:
        print("EKSIK DOSYALAR:")
        for path in absent:
            print("  -", path)
        return 1

    try:
        config = json.loads((root / CONFIG_REL).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print("CONFIG JSON HATASI:", exc)
        return 1

    ok, explanation = inspect_class_mapping(config)
    if not ok:
        print("MODEL METADATA KONTROLU BASARISIZ:", explanation)
        return 1

    print("BASARILI: Gerekli dosyalar mevcut.")
    print("BASARILI:", explanation)
    print("NOT: Bu kontrol model agirliklarini yuklemez, gercek inference veya Docker testini yapmaz.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
