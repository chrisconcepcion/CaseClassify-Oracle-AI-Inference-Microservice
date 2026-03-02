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