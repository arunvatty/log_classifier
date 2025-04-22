import torch
import torch.nn as nn
import joblib

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, output_dim=1):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return self.sigmoid(out)

def load_model_and_vocab():
    vocab = joblib.load("model/tokenizer.pkl")
    model = LSTMClassifier(len(vocab))
    model.load_state_dict(torch.load("model/lstm_model.pt", map_location=torch.device("cpu")))
    model.eval()
    return model, vocab
