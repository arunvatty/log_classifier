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
    from train_model import LSTMClassifier  # Update this import to your model class
    
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
    
    # Check that result is a boolean or a probability (0-1)
    assert isinstance(result, (bool, float, int))
    if isinstance(result, float):
        assert 0 <= result <= 1