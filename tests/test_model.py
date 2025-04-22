from app.predict import predict

def test_predict_important():
    log = "Exception occurred in main thread due to critical error"
    result = predict(log)
    assert result == "Important"

def test_predict_not_important():
    log = "App started successfully without issues"
    result = predict(log)
    assert result == "Not Important"
