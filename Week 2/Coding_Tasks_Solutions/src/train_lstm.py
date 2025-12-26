# Importing necessary libraries
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from dataset import Vocabulary, TextDataset
from lstm_model import LSTMClassifier

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
MAX_LEN = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Loading train and test data
train_df = pd.read_csv("D:\Projects\Winter_in_Data_Science\WiDS-2025-Speech_to_text_engine\Week 2\Coding_Tasks_Solutions\data\imdb_train.csv")
test_df = pd.read_csv("D:\Projects\Winter_in_Data_Science\WiDS-2025-Speech_to_text_engine\Week 2\Coding_Tasks_Solutions\data\imdb_test.csv")

# Building vocabulary
vocab = Vocabulary(max_size=20000)
vocab.build_vocab(train_df["text"])

# Creating datasets and dataloaders
train_ds = TextDataset(
    train_df["text"],
    train_df["label"],
    vocab,
    MAX_LEN
)

# Creating datasets and dataloaders
test_ds = TextDataset(
    test_df["text"],
    test_df["label"],
    vocab,
    MAX_LEN
)

# DataLoaders
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# Model, loss function, and optimizer
model = LSTMClassifier(len(vocab.word2idx)).to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
all_preds, all_labels = [], []

# Evaluation loop
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        preds = model(x).cpu()
        all_preds.extend((preds > 0.5).int().tolist())
        all_labels.extend(y.tolist())

# Calculating metrics
print("Accuracy:", accuracy_score(all_labels, all_preds))
print("F1 Score:", f1_score(all_labels, all_preds))