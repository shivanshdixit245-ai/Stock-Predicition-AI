import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint_returns_ok():
    """Verify health check returns status ok and a timestamp."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data

def test_predict_returns_404_when_no_model():
    """Verify 404 with clean message when model doesn't exist."""
    ticker = "NONEXIST" # Valid format but no model
    response = client.get(f"/tickers/{ticker}/predict")
    assert response.status_code == 404
    assert response.json()["detail"] == f"No trained model found for {ticker}. Call POST /train first."

def test_invalid_ticker_rejected():
    """Verify ticker "INVALID123" returns HTTP 422."""
    ticker = "INVALID123"
    response = client.get(f"/tickers/{ticker}/data")
    assert response.status_code == 422
    assert "Invalid ticker format" in response.json()["detail"]

def test_train_endpoint_returns_run_id():
    """Verify train endpoint returns a run_id and status."""
    ticker = "AAPL"
    payload = {
        "ticker": ticker,
        "start_date": "2022-01-01",
        "end_date": "2023-12-31",
        "use_sentiment": True,
        "use_regime": True
    }
    response = client.post(f"/tickers/{ticker}/train", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data
    assert data["status"] == "training_complete"
