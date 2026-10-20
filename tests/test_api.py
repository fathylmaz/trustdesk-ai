from fastapi.testclient import TestClient
from trustdesk.api import app
from trustdesk.train import train_and_evaluate
from test_pipeline import make_fake_data


def test_api(tmp_path, monkeypatch):
    train_path, test_path = make_fake_data(tmp_path)
    train_and_evaluate(train_path, test_path, tmp_path / "out")
    monkeypatch.setenv("TRUSTDESK_MODEL_PATH", str(tmp_path / "out" / "baseline.joblib"))
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        response = client.post("/predict", json={"text": "my payment card was stolen"})
        assert response.status_code == 200
        assert "top_predictions" in response.json()
        assert client.post("/predict", json={"text": "x"}).status_code == 422
