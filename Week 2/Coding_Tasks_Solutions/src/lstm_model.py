# Importing necessary libraries
import torch
import torch.nn as nn

# Defining the LSTM-based sentiment classifier
class LSTMClassifier(nn.Module):
    # Constructor to initialize layers
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_layers=1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    # Forward pass
    def forward(self, x):
        emb = self.embedding(x)
        _, (h, _) = self.lstm(emb)
        out = self.fc(h[-1])
        return self.sigmoid(out).squeeze()