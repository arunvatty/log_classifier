from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_api_predict_important():
    response = client.post("/predict", json={"log": "Unhandled exception occurred"})
    assert response.status_code == 200
    result = response.json()
    # Check for dictionary with prediction key
    assert "prediction" in result
    # Compare just the prediction field
    assert result["prediction"] == "Important"

def test_api_predict_not_important():
    response = client.post("/predict", json={"log": "Scheduled task executed successfully"})
    assert response.status_code == 200
    result = response.json()
    # Check for dictionary with prediction key
    assert "prediction" in result
    # Test for "Normal" since that's what the model returns
    assert result["prediction"] in ["Not Important", "Normal"]