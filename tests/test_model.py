import pytest
import os
import torch
import joblib
from app.predict import predict  # Update this import to match your actual predict function path

# Skip tests if model files don't exist - useful for CI
need_model_files = pytest.mark.skipif(
    not os.path.exists("model/lstm_model.pt") or not os.path.exists("model/tokenizer.pkl"),
    reason="Model files not found. Tests requiring model files will be skipped."
)

def test_imports():
    """Test that critical libraries can be imported"""
    import torch
    import nltk
    import fastapi
    import pandas
    import sklearn
    assert True  # If we get here, imports worked

@need_model_files
def test_model_load():
    """Test that the model can be loaded"""
    from train.train_model import LSTMClassifier  # Fix: Updated import path
    
    # Load vocabulary
    vocab = joblib.load("model/tokenizer.pkl")
    vocab_size = len(vocab)
    
    # Initialize model
    model = LSTMClassifier(vocab_size=vocab_size)
    
    # Load model weights
    model.load_state_dict(torch.load("model/lstm_model.pt"))
    
    # Set to evaluation mode
    model.eval()
    
    assert isinstance(model, LSTMClassifier)

@need_model_files
def test_predict_function():
    """Test that the predict function works with a sample log"""
    sample_log = "Application started successfully"
    result = predict(sample_log)
    
    # Check that result is a dictionary with the expected fields
    assert isinstance(result, dict)
    assert "prediction" in result
    assert "probability" in result
    assert isinstance(result["probability"], float)
    assert 0 <= result["probability"] <= 1