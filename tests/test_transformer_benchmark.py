"""v0.4 fast, offline unit tests (no Hugging Face download, no GPU)."""
import numpy as np
import pytest

from trustdesk_transformer_benchmark import label_mapping, evaluate_probs


def test_sorted_class_ids():
    id2label, label2id = label_mapping(["z", "a", "b"])
    assert id2label == {0: "a", 1: "b", 2: "z"}
    assert label2id["z"] == 2


def test_reject_duplicate_class_labels():
    with pytest.raises(ValueError):
        label_mapping(["a", "a", "b"])


def test_model_metrics():
    truth = ["a", "b", "c", "a"]
    probs = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.1, 0.8],
        [0.3, 0.5, 0.2],
    ])
    m = evaluate_probs(truth, probs, ["a", "b", "c"])
    assert m["count"] == 4
    assert m["wrong_count"] == 1
    assert m["accuracy"] == 0.75
    assert m["top3_accuracy"] == 1.0


def test_reject_invalid_probability_rows():
    with pytest.raises(ValueError, match="invalid probability"):
        evaluate_probs(["a"], [[1., 1., 0.]], ["a", "b", "c"])


def test_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        evaluate_probs(["a"], [[0.1, 0.9]], ["a", "b", "c"])
