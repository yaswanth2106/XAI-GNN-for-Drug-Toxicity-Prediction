import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from data import train_loader, test_loader
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from model import GNN

device = "cpu"
model = GNN(in_channels=train_loader.dataset[0].x.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data).view(-1)
        y = data.y[:, 0].float()

        loss = F.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def test():
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data).view(-1)

            preds = (torch.sigmoid(out) > 0.5).float().cpu()
            labels = data.y[:, 0].float().cpu()

            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    return accuracy_score(y_true, y_pred)


for epoch in range(1, 21):
    loss = train()
    acc = test()
    print(f"Epoch {epoch:02d} | Loss {loss:.4f} | Accuracy {acc:.4f}")


MODEL_PATH = "gnn_tox21.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "in_channels": 9
}, "gnn_tox21.pth")


print("✅ Model saved")

