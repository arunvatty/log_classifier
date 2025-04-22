from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_api_predict_important():
    response = client.post("/predict", json={"log": "Unhandled exception occurred"})
    assert response.status_code == 200
    assert response.json()["prediction"] == "Important"

def test_api_predict_not_important():
    response = client.post("/predict", json={"log": "Scheduled task executed successfully"})
    assert response.status_code == 200
    assert response.json()["prediction"] == "Not Important"
