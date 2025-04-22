import json
import re
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from collections import Counter
import joblib
import os

nltk.download("punkt")
nltk.download("punkt_tab")

def clean_log(log):
    """Clean log text by removing timestamps and non-alphabetic characters"""
    log = re.sub(r"\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+", "", log)  # Remove timestamps
    log = re.sub(r"[^a-zA-Z ]", " ", log)  # Remove non-alphabetic
    return log.lower().strip()

def tokenize_and_build_vocab(logs, min_freq=2):
    """Tokenize logs and build vocabulary from the most frequent words"""
    # Clean and tokenize each log
    tokenized_logs = [word_tokenize(clean_log(log)) for log in logs]
    
    # Count word frequencies across all logs
    all_tokens = [token for log_tokens in tokenized_logs for token in log_tokens]
    word_counts = Counter(all_tokens)
    
    # Build vocabulary with words appearing at least min_freq times
    vocab = {}
    vocab["<PAD>"] = 0  # Padding token
    vocab["<UNK>"] = 1  # Unknown token
    
    # Add words to vocabulary with indices starting from 2
    idx = 2
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return tokenized_logs, vocab

def encode_logs(tokenized_logs, vocab, max_len=50):
    """Encode tokenized logs using vocabulary indices"""
    unk_idx = vocab["<UNK>"]
    pad_idx = vocab["<PAD>"]
    
    encoded = []
    for tokens in tokenized_logs:
        # Map each token to its vocabulary index, use <UNK> for OOV tokens
        ids = [vocab.get(token, unk_idx) for token in tokens[:max_len]]
        
        # Pad sequences to max_len
        padded = ids + [pad_idx] * (max_len - len(ids))
        encoded.append(padded)
    
    return encoded


def preprocess_data(path="data/labeled_logs.json", test_size=0.2):
    """Main preprocessing function that handles the entire pipeline"""
    # Load data
    with open(path, "r") as f:
        data = json.load(f)
    
    logs = [item["log"] for item in data]
    labels = [1 if item["label"] == "Important" else 0 for item in data]
    
    # Step 1: Clean and tokenize logs, build vocabulary
    tokenized_logs, vocab = tokenize_and_build_vocab(logs, min_freq=2)
    print(f"Built vocabulary with {len(vocab)} entries")
    
    # Step 2: Encode logs using vocabulary
    encoded_logs = encode_logs(tokenized_logs, vocab)
    
    # Step 3: Verify no token exceeds vocabulary size
    max_token = max(max(seq) for seq in encoded_logs)
    if max_token >= len(vocab):
        raise ValueError(f"Token index {max_token} >= vocab size {len(vocab)}")
    
    # Step 4: Create dataframe
    df = pd.DataFrame(encoded_logs)
    df["label"] = labels
    
    # Step 5: Train/val split
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)
    
    # Step 6: Save vocabulary and processed data
    os.makedirs("model", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    joblib.dump(vocab, "model/tokenizer.pkl")
    
    return train_df, val_df


if __name__ == "__main__":
    train_df, val_df = preprocess_data()
    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
