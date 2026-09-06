from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


VALID_CUSTOMER = {
    "LIMIT_BAL": 50000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 35,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 10000,
    "BILL_AMT2": 9000,
    "BILL_AMT3": 8000,
    "BILL_AMT4": 7000,
    "BILL_AMT5": 6000,
    "BILL_AMT6": 5000,
    "PAY_AMT1": 1000,
    "PAY_AMT2": 1000,
    "PAY_AMT3": 1000,
    "PAY_AMT4": 1000,
    "PAY_AMT5": 1000,
    "PAY_AMT6": 1000,
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_valid_customer():
    response = client.post(
        "/predict",
        json=VALID_CUSTOMER,
    )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_probability" in data
    assert "risk_band" in data
    assert "decision_at_0_30" in data

    assert 0.0 <= data["predicted_probability"] <= 1.0
    assert data["decision_at_0_30"] in [0, 1]


def test_predict_missing_feature():
    customer = VALID_CUSTOMER.copy()
    customer.pop("AGE")

    response = client.post(
        "/predict",
        json=customer,
    )

    assert response.status_code == 422


def test_predict_invalid_feature_type():
    customer = VALID_CUSTOMER.copy()
    customer["AGE"] = "thirty-five"

    response = client.post(
        "/predict",
        json=customer,
    )

    assert response.status_code == 422