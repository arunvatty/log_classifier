import torch
import joblib

# Import the same model class
from simple_train import LSTMClassifier

def diagnose_model():
    """Run diagnostics on the trained model"""
    # Load vocabulary
    vocab = joblib.load("model/tokenizer.pkl")
    vocab_size = len(vocab)
    
    # Initialize model with current architecture
    model = LSTMClassifier(vocab_size=vocab_size)
    
    # Load the model weights
    model.load_state_dict(torch.load("model/lstm_model.pt"))
    
    # Run diagnostic calculations
    lstm_norm = calculate_lstm_norm(model)
    
    print(f"LSTM Weight Norm: {lstm_norm:.4f}")
    
    return {"lstm_norm": lstm_norm}

def calculate_lstm_norm(model):
    """Calculate the norm of LSTM weights"""
    with torch.no_grad():
        # Calculate norm of LSTM weights
        lstm_weights = []
        for name, param in model.named_parameters():
            if 'lstm' in name and 'weight' in name:
                lstm_weights.append(param.norm().item())
        
        return sum(lstm_weights) / len(lstm_weights) if lstm_weights else 0