import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

# IMPORTANT: Every test function MUST include 'client' in the parentheses
# so that pytest can inject the fixture.
def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_high_urgency_prediction(client):
    payload = {
        "case_id": "test-1",
        "description": "This is a disaster. I am going to sue the company for everything they have.",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["urgency_score"] == "HIGH"

def test_invalid_payload(client):
    # Tests the Pydantic 'Contract' validation
    payload = {"case_id": "test-2", "description": "Too short"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_full_pipeline_integration(client):
    """
    Verifies that raw data can pass through the Refinery and reach the Oracle.
    """
    from data_pipeline.processor import get_reproducible_pipeline, clean_data_efficiently
    import pandas as pd

    # 1. Create 'Dirty' Data
    raw_data = pd.DataFrame([{
        "case_id": "integration-1",
        "case_age_days": 10,
        "claim_amount": 50000,
        "case_type": "Liability",
        "jurisdiction": "ny " # Includes whitespace to test cleaning
    }])

    # 2. Run through Refinery
    clean_df = clean_data_efficiently(raw_data)
    pipeline = get_reproducible_pipeline()
    features = pipeline.fit_transform(clean_df)

    # 3. Assert Oracle-compatible output
    assert features.shape[1] > 0
    assert clean_df['is_high_value'].iloc[0] == 1