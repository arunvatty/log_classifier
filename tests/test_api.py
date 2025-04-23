from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_api_predict_important():
    response = client.post("/predict", json={"log": "Unhandled exception occurred"})
    assert response.status_code == 200
    result = response.json()
    # Check for dictionary with prediction key
    assert "prediction" in result
    # The log contains "exception" which should now trigger Important classification
    assert result["prediction"] == "Important"
    # Verify keyword override was used
    assert "method" in result
    assert result["method"] == "keyword_override"

def test_api_predict_not_important():
    response = client.post("/predict", json={"log": "Scheduled task executed successfully"})
    assert response.status_code == 200
    result = response.json()
    # Check for dictionary with prediction key
    assert "prediction" in result
    # Test for "Not Important" since that's what the model returns for normal logs
    assert result["prediction"] in ["Not Important", "Normal"]
    # Verify model prediction was used
    assert "method" in result
    assert result["method"] == "model_prediction"