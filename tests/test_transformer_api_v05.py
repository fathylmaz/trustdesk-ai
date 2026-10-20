"""Offline unit tests; do NOT require 268-MB model or a GPU."""

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SCRIPT = Path(__file__).resolve().parents[1] / "trustdesk_transformer_api_v05.py"
spec = importlib.util.spec_from_file_location("trustdesk_transformer_api_v05", SCRIPT)
svc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = svc
spec.loader.exec_module(svc)


def test_labels_are_read_from_real_mapping_in_id_order():
    assert svc.checked_labels({"1": "second", "0": "first"}, 2) == ["first", "second"]


def test_generic_mapping_is_rejected():
    with pytest.raises(ValueError, match="genuine"):
        svc.checked_labels({0: "LABEL_0"}, 1)


def test_probability_ranking_and_rounding():
    result = svc.rank_probabilities([0.02, 0.8, 0.18], ["a", "b", "c"])
    assert [row["label"] for row in result] == ["b", "c", "a"]
    assert result[0]["score"] == 0.8


@pytest.mark.parametrize("probs", [[0.8, 0.8], [float("nan"), 0.5], [-0.1, 1.1]])
def test_invalid_probabilities_rejected(probs):
    with pytest.raises(ValueError):
        svc.rank_probabilities(probs, ["a", "b"])


def test_missing_local_weights_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing saved model"):
        svc.LocalTransformerPredictor(tmp_path, "cpu")


def test_api_contract_with_fake_predictor():
    class FakePredictor:
        device = "cpu"

        def classify(self, text):
            return {"label": "lost_or_stolen_card", "score": 0.4,
                    "top_predictions": [{"label": "lost_or_stolen_card", "score": 0.4}],
                    "automatic_routing_enabled": False,
                    "score_note": "Experimental uncalibrated softmax scores"}

    svc.app.state.predictor = FakePredictor()
    # Without lifespan context the 268 MB checkpoint isn't loaded.
    client = TestClient(svc.app)
    health = client.get("/health")
    assert health.status_code == 200 and health.json()["model_loaded"] is True
    response = client.post("/predict", json={"text": "My card was stolen"})
    assert response.status_code == 200
    assert response.json()["automatic_routing_enabled"] is False
    assert client.post("/predict", json={"text": "ok"}).status_code == 422
