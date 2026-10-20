"""Offline unit tests: no GPU, no checkpoint downloads, no real customer data."""
from threading import Lock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import trustdesk_unified_api_v06 as svc


class FakeBaseline:
    classes_ = np.array(["cancel_transfer", "terminate_account"])

    def predict_proba(self, texts):
        return np.array([[0.15, 0.85] for _ in texts])


class FakeTransformer:
    device = "cpu"

    def classify(self, text):
        return {"label": "terminate_account", "score": 0.65,
                "top_predictions": [{"label": "terminate_account", "score": 0.65},
                                    {"label": "cancel_transfer", "score": 0.35}],
                "automatic_routing_enabled": False,
                "score_note": "Experimental uncalibrated softmax scores."}


@pytest.fixture
def client():
    # Do not enter lifespan: real 268-MB checkpoints are not required for tests.
    svc.app.state.baseline = FakeBaseline()
    svc.app.state.transformer = FakeTransformer()
    svc.app.state.predict_lock = Lock()
    client = TestClient(svc.app)
    yield client
    svc.app.state.baseline = None
    svc.app.state.transformer = None


def test_health_and_models(client):
    data = client.get("/health").json()
    assert data["models_loaded"] is True
    assert data["automatic_routing_enabled"] is False
    ids = [row["id"] for row in client.get("/v1/models").json()["models"]]
    assert ids == ["tfidf", "distilbert"]


def test_predict_tfidf(client):
    response = client.post("/v1/predict", json={"model": "tfidf", "text": "Close my bank account"})
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "tfidf"
    assert data["label"] == "terminate_account"
    assert data["top_predictions"][0]["score"] == 0.85
    assert data["automatic_routing_enabled"] is False
    assert data["inference_ms"] >= 0


def test_predict_transformer(client):
    result = client.post("/v1/predict", json={"model": "distilbert", "text": "Close my bank account"})
    assert result.status_code == 200
    assert result.json()["score"] == 0.65
    assert "uncalibrated" in result.json()["score_note"].lower()


def test_compare_returns_both_models_without_routing(client):
    response = client.post("/v1/compare", json={"text": "I want to close my account"})
    assert response.status_code == 200
    result = response.json()
    assert [r["model"] for r in result["predictions"]] == ["tfidf", "distilbert"]
    assert all(not r["automatic_routing_enabled"] for r in result["predictions"])
    assert "Do not compare" in result["warning"]


@pytest.mark.parametrize("payload", [{"text": "hi"}, {"text": ""}, {}])
def test_invalid_text_rejected(client, payload):
    assert client.post("/v1/compare", json=payload).status_code == 422


def test_unknown_model_rejected(client):
    assert client.post("/v1/predict", json={"text": "Close my account", "model": "gpt"}).status_code == 422


def test_unloaded_models_give_503(client):
    svc.app.state.transformer = None
    assert client.post("/v1/compare", json={"text": "Close my account"}).status_code == 503


def test_cors_only_allows_local_react(client):
    good = client.options("/v1/compare", headers={"Origin": "http://localhost:5173",
                                               "Access-Control-Request-Method": "POST"})
    assert good.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.options("/v1/compare", headers={"Origin": "https://malicious.example",
                                              "Access-Control-Request-Method": "POST"})
    assert bad.headers.get("access-control-allow-origin") is None
