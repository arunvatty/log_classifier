# predict.py

import torch
import joblib
import nltk
from nltk.tokenize import word_tokenize
import re
from train.train_model import LSTMClassifier  # Import your model class

# Make sure necessary NLTK packages are downloaded
nltk.download("punkt", quiet=True)

def clean_log(log):
    """Clean log text by removing timestamps and non-alphabetic characters"""
    log = re.sub(r"\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+", "", log)  # Remove timestamps
    log = re.sub(r"[^a-zA-Z ]", " ", log)  # Remove non-alphabetic
    return log.lower().strip()

def tokenize_and_encode(log_text, vocab, max_len=50):
    """Tokenize and encode a single log entry using the vocabulary"""
    # Clean and tokenize
    cleaned_log = clean_log(log_text)
    tokens = word_tokenize(cleaned_log)
    
    # Encode using vocabulary
    unk_idx = vocab["<UNK>"]
    pad_idx = vocab["<PAD>"]
    
    # Convert tokens to indices
    token_ids = [vocab.get(token, unk_idx) for token in tokens[:max_len]]
    
    # Pad sequence
    padded_ids = token_ids + [pad_idx] * (max_len - len(token_ids))
    
    return padded_ids

def load_model():
    """Load the trained model and vocabulary"""
    # Load vocabulary
    vocab = joblib.load("model/tokenizer.pkl")
    vocab_size = len(vocab)
    
    # Initialize model with correct vocabulary size
    model = LSTMClassifier(vocab_size=vocab_size)
    
    # Load trained weights
    model.load_state_dict(torch.load("model/lstm_model.pt"))
    model.eval()  # Set to evaluation mode
    
    return model, vocab

def predict(log_text, model=None, vocab=None):
    """Predict if a log entry is important"""
    # Load model and vocabulary if not provided
    if model is None or vocab is None:
        model, vocab = load_model()
    
    # Tokenize and encode the input log
    encoded_log = tokenize_and_encode(log_text, vocab)
    
    # Convert to tensor
    input_tensor = torch.tensor([encoded_log], dtype=torch.long)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probability = output.item()
        prediction = 1 if probability >= 0.5 else 0
    
    return {
        "prediction": "Important" if prediction == 1 else "Normal",
        "probability": probability,
        "encoded_length": sum(1 for id in encoded_log if id != 0)  # Count non-padding tokens
    }

def predict_batch(log_texts):
    """Predict importance for multiple log entries"""
    # Load model and vocabulary once
    model, vocab = load_model()
    
    results = []
    for log in log_texts:
        result = predict(log, model, vocab)
        results.append(result)
    
    return results

# Example usage
if __name__ == "__main__":
    # Single prediction example
    sample_log = "03-17 16:13:46.765  2227  2794 W KeyguardUpdateMonitor: android.util.AndroidRuntimeException: Must execute in UI"
    result = predict(sample_log)
    print(f"Log: {sample_log}")
    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.4f}")
    
    # Batch prediction example
    sample_logs = [
        "03-17 16:13:45.466  1702 17632 W ActivityManager: java.lang.ClassCastException: android.os.BinderProxy cannot be cast to com.android.server.am.ActivityRecord$Token",
        "04-15 12:35:12.345 Info: Application started successfully",
        "04-15 12:36:45.678 Warning: High memory usage detected (85%)"
    ]
    
    print("\nBatch predictions:")
    results = predict_batch(sample_logs)
    for i, (log, result) in enumerate(zip(sample_logs, results)):
        print(f"\nLog {i+1}: {log}")
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']:.4f}")