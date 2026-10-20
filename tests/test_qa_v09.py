"""Fast offline QA: fake models only; requires NO live server, GPU or checkpoints."""
from threading import Lock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import trustdesk_unified_api_v06 as svc
from trustdesk_qa_v09 import QAError, latency_summary, nearest_rank, validate_url, verify_prediction


class FakeBaseline:
    classes_ = np.array(["cancel_transfer", "terminate_account"])

    def predict_proba(self, texts):
        return np.array([[0.15, 0.85] for _ in texts])


class FakeTransformer:
    device = "cpu"

    def classify(self, text):
        return {
            "label": "terminate_account", "score": 0.65,
            "top_predictions": [
                {"label": "terminate_account", "score": 0.65},
                {"label": "cancel_transfer", "score": 0.35},
            ],
            "automatic_routing_enabled": False,
            "score_note": "Experimental uncalibrated softmax scores.",
        }


@pytest.fixture
def client():
    # Do not enter lifespan: no real model files are touched or loaded.
    svc.app.state.baseline = FakeBaseline()
    svc.app.state.transformer = FakeTransformer()
    svc.app.state.predict_lock = Lock()
    client = TestClient(svc.app)  # No context manager: does NOT start lifespan.
    yield client
    svc.app.state.baseline = None
    svc.app.state.transformer = None


def test_health_and_model_registry(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["models_loaded"] is True
    assert health.json()["automatic_routing_enabled"] is False
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert [m["id"] for m in response.json()["models"]] == ["tfidf", "distilbert"]


def test_comparison_schema_and_routing_disabled(client):
    response = client.post("/v1/compare", json={"text": "I want to cancel my bank account."})
    assert response.status_code == 200
    content = response.json()
    assert content["text"] == "I want to cancel my bank account."
    assert len(content["predictions"]) == 2
    for name, prediction in zip(("tfidf", "distilbert"), content["predictions"]):
        verify_prediction(prediction, name)
    assert "automate banking actions" in content["warning"].lower()


@pytest.mark.parametrize("model", ["tfidf", "distilbert"])
def test_single_model_prediction_contract(client, model):
    response = client.post("/v1/predict", json={"model": model, "text": "Close my account please"})
    assert response.status_code == 200
    verify_prediction(response.json(), model)


@pytest.mark.parametrize("bad_payload", [
    {}, {"text": ""}, {"text": "hi"}, {"text": "x" * 2001}, {"text": 1234}
])
def test_invalid_compare_input_returns_422(client, bad_payload):
    assert client.post("/v1/compare", json=bad_payload).status_code == 422


def test_invalid_single_model_name_returns_422(client):
    assert client.post("/v1/predict", json={"model": "gpt", "text": "Close my account"}).status_code == 422


def test_missing_models_returns_503(client):
    svc.app.state.transformer = None
    assert client.post("/v1/compare", json={"text": "Close my account"}).status_code == 503


def test_development_only_cors_origins(client):
    allowed = client.options("/v1/compare", headers={
        "Origin": "http://127.0.0.1:5173", "Access-Control-Request-Method": "POST"})
    assert allowed.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"
    forbidden = client.options("/v1/compare", headers={
        "Origin": "https://malicious.example", "Access-Control-Request-Method": "POST"})
    assert forbidden.headers.get("access-control-allow-origin") is None


def test_local_api_url_only():
    assert validate_url("http://127.0.0.1:8002/") == "http://127.0.0.1:8002"
    assert validate_url("http://localhost:8002") == "http://localhost:8002"


@pytest.mark.parametrize("url", [
    "https://127.0.0.1:8002", "http://example.com:8002", "http://192.168.1.5:8002",
    "http://127.0.0.1:8002/v1", "http://127.0.0.1:8002?foo=1", "http://localhost",
])
def test_nonlocal_or_invalid_url_rejected(url):
    with pytest.raises(QAError):
        validate_url(url)


def test_percentile_nearest_rank_and_latency_summary():
    assert nearest_rank([7, 1, 5, 3, 9], 50) == 5
    assert nearest_rank([7, 1, 5, 3, 9], 95) == 9
    assert latency_summary([10, 20, 30, 40], [1, 2, 3, 4]) == {
        "requests": 4, "round_trip_ms_mean": 25.0, "round_trip_ms_p50": 20,
        "round_trip_ms_p95": 40, "model_inference_ms_p50": 2,
        "model_inference_ms_p95": 4,
    }


def test_percentile_invalid_inputs():
    with pytest.raises(ValueError):
        nearest_rank([], 95)
    with pytest.raises(ValueError):
        nearest_rank([1], 0)
    with pytest.raises(ValueError):
        latency_summary([1], [1, 2])


def test_prediction_contract_rejects_auto_routing():
    fake = {
        "model": "tfidf", "label": "example", "score": 0.9,
        "top_predictions": [{"label": "example", "score": 0.9}],
        "inference_ms": 1.0, "automatic_routing_enabled": True,
        "score_note": "Not calibrated and not production-ready",
    }
    with pytest.raises(QAError, match="auto-routing"):
        verify_prediction(fake, "tfidf")


def test_prediction_contract_rejects_bad_ranking():
    fake = {
        "model": "distilbert", "label": "example", "score": 0.7,
        "top_predictions": [{"label": "example", "score": 0.7}, {"label": "other", "score": 0.8}],
        "inference_ms": 1.0, "automatic_routing_enabled": False,
        "score_note": "Not calibrated and not production-ready",
    }
    with pytest.raises(QAError, match="sorted"):
        verify_prediction(fake, "distilbert")
