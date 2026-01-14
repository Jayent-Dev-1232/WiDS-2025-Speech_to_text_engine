import torch
from torch.utils.data import DataLoader
from dataset import SpeechDataset
from cnn_model import KeywordCNN
import torch.nn as nn

dataset = SpeechDataset("data/speech_commands")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = KeywordCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    total_loss = 0
    for x, y in loader:
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")