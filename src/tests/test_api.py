"""Test the HTTP API contracts and validation behavior."""

from fastapi.testclient import TestClient

from src.serving.main import app
from src.tests.test_predict import CLIENT_EXEMPLE


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_valid_customer():
    response = client.post("/predict", json=CLIENT_EXEMPLE)
    body = response.json()

    assert response.status_code == 200
    assert 0 <= body["churn_probability"] <= 1
    assert body["churn_label"] in {0, 1}
    assert body["risk_level"] in {"Low", "Medium", "High"}


def test_predict_rejects_invalid_customer():
    invalid_contract = {**CLIENT_EXEMPLE, "Contract": "banana"}
    missing_field = {key: value for key, value in CLIENT_EXEMPLE.items() if key != "tenure"}
    negative_tenure = {**CLIENT_EXEMPLE, "tenure": -1}

    assert client.post("/predict", json=invalid_contract).status_code == 422
    assert client.post("/predict", json=missing_field).status_code == 422
    assert client.post("/predict", json=negative_tenure).status_code == 422


def test_batch_contract():
    response = client.post(
        "/predict/batch", json={"customers": [CLIENT_EXEMPLE, CLIENT_EXEMPLE]}
    )

    assert response.status_code == 200
    assert response.json()["total"] == 2
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_interpret_contract():
    response = client.post("/interpret?top_n=3", json=CLIENT_EXEMPLE)
    body = response.json()

    assert response.status_code == 200
    assert body["top_features"]
    assert len(body["top_features"]) == 3
    assert any(abs(value) > 1e-6 for value in body["top_features"].values())
    assert isinstance(body["summary"], str)
    assert len(body["explanation"]) == 3
    assert {
        "feature",
        "label",
        "value",
        "shap",
        "direction",
    }.issubset(body["explanation"][0])


def test_unavailable_model_returns_503(monkeypatch):
    import src.serving.main as serving_main

    monkeypatch.setattr(serving_main, "model", None)
    response = client.post("/predict", json=CLIENT_EXEMPLE)
    health_response = client.get("/health")

    assert response.status_code == 503
    assert health_response.status_code == 503
    assert health_response.json()["model_loaded"] is False
