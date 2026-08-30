from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


def test_predict_valid_payload():
    payload = {
        "age": 45, "bmi": 27.5, "blood_pressure": 125, "glucose": 125,
        "insulin": 85, "cholesterol": 205, "hba1c": 6.1, "sugar": 115,
    }
    resp = client.post("/api/v1/predict", json=payload)
    # 200 if model trained, 503 if not trained yet - both are valid states
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        body = resp.json()
        assert 0.0 <= body["risk_score"] <= 1.0
        assert body["risk_band"] in ("Low", "Moderate", "High")


def test_predict_rejects_out_of_range():
    payload = {
        "age": 999, "bmi": 27.5, "blood_pressure": 125, "glucose": 125,
        "insulin": 85, "cholesterol": 205, "hba1c": 6.1, "sugar": 115,
    }
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422
