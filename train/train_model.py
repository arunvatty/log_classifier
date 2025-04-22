import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
from train.preprocess import preprocess_data

# Define model matching your architecture in train_model.py
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3, output_dim=1):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.vocab_size = vocab_size
        
    def attention_net(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context
        
    def forward(self, x):
        x = self.embedding(x)
        lstm_output, _ = self.lstm(x)
        context = self.attention_net(lstm_output)
        output = self.fc1(context)
        output = torch.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = self.sigmoid(output)
        return output

def train_model():
    """Simple function to train a new model and save it"""
    print("Training new model from scratch...")
    
    # Preprocess data
    train_df, val_df = preprocess_data()
    print(f"Loaded {len(train_df)} training samples")
    
    # Load vocabulary
    vocab = joblib.load("model/tokenizer.pkl")
    vocab_size = len(vocab)
    print(f"Loaded vocabulary with {vocab_size} entries")
    
    # Prepare data
    X_train = torch.tensor(train_df.drop("label", axis=1).values, dtype=torch.long)
    y_train = torch.tensor(train_df["label"].values, dtype=torch.float32).unsqueeze(1)
    
    # Create data loader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    
    # Initialize model
    model = LSTMClassifier(vocab_size=vocab_size)
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(5):
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = (outputs >= 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)
        
        # Print progress
        print(f"Epoch {epoch+1}/5: Loss={total_loss/len(train_loader):.4f}, Acc={correct/total:.4f}")
    
    # Save model
    torch.save(model.state_dict(), "model/lstm_model.pt")
    print("✅ Model saved to model/lstm_model.pt")
    
    return model

if __name__ == "__main__":
    train_model()